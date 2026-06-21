import copy
from dataclasses import dataclass
import json
from pathlib import Path
import queue
import re
import sys
import threading
import time
from typing import Any, Union

from Levenshtein import distance
from loguru import logger
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, HttpUrl
import requests
import sounddevice as sd  # type: ignore
from sounddevice import CallbackFlags
import yaml

from .ASR.asr import AudioTranscriber as ParakeetASR
from .ASR import VAD
from .TTS import tts_kokoro
from .utils import spoken_text_converter as stc
from .utils.resources import resource_path

# Type alias for ASR models
ASRModel = ParakeetASR

logger.remove(0)
logger.add(sys.stderr, level="SUCCESS")


class PersonalityPrompt(BaseModel):
    system: str | None = None
    user: str | None = None
    assistant: str | None = None

    def to_chat_message(self) -> dict[str, str]:
        """Convert the prompt to a chat message format.

        Returns:
            dict[str, str]: A single chat message dictionary

        Raises:
            ValueError: If the prompt does not contain exactly one non-null field
        """
        for field, value in self.model_dump(exclude_none=True).items():
            return {"role": field, "content": value}
        raise ValueError("PersonalityPrompt must have exactly one non-null field")


class GladosConfig(BaseModel):
    completion_url: HttpUrl
    model: str
    api_key: str | None = None
    interruptible: bool = True
    wake_word: str | None = None
    voice: str
    announcement: str | None = None
    personality_preprompt: list[PersonalityPrompt]
    api_type: str = "openai"  # "openai" or "gemini"
    asr_model: str = "parakeet"  # "parakeet" or "zipformer"
    rag_enabled: bool = True
    rag_top_k: int = 3
    manual_trigger_required: bool = False
    asr_corrections: dict = {}

    @classmethod
    def from_yaml(cls, path: str | Path, key_to_config: tuple[str, ...] = ("Glados",)) -> "GladosConfig":
        """
        Load a GladosConfig instance from a YAML configuration file.

        Parameters:
            path: Path to the YAML configuration file
            key_to_config: Tuple of keys to navigate nested configuration

        Returns:
            GladosConfig: Configuration object with validated settings

        Raises:
            ValueError: If the YAML content is invalid
            OSError: If the file cannot be read
            pydantic.ValidationError: If the configuration is invalid
        """
        path = Path(path)

        # Try different encodings
        for encoding in ["utf-8", "utf-8-sig"]:
            try:
                data = yaml.safe_load(path.read_text(encoding=encoding))
                break
            except UnicodeDecodeError:
                if encoding == "utf-8-sig":
                    raise

        # Navigate through nested keys
        config = data
        for key in key_to_config:
            config = config[key]

        return cls(**cls.model_validate(config).dict())

    def to_chat_messages(self) -> list[dict[str, str]]:
        """Convert personality preprompt to chat message format."""
        return [prompt.to_chat_message() for prompt in self.personality_preprompt]


@dataclass
class AudioMessage:
    audio: NDArray[np.float32]
    text: str
    is_eos: bool = False


class Glados:
    PAUSE_TIME: float = 0.05  # Time to wait between processing loops
    SAMPLE_RATE: int = 16000  # Sample rate for input stream
    VAD_SIZE: int = 32  # Milliseconds of sample for Voice Activity Detection (VAD)
    VAD_THRESHOLD: float = 0.8  # Threshold for VAD detection
    BUFFER_SIZE: int = 800  # Milliseconds of buffer BEFORE VAD detection
    PAUSE_LIMIT: int = 640  # Milliseconds of pause allowed before processing
    SIMILARITY_THRESHOLD: int = 2  # Threshold for wake word similarity

    NEUROTOXIN_RELEASE_ALLOWED: bool = False  # preparation for function calling, see issue #13
    DEFAULT_PERSONALITY_PREPROMPT: tuple[dict[str, str], ...] = (
        {
            "role": "system",
            "content": "You are a helpful AI assistant. You are here to assist the user in their tasks.",
        },
    )

    def __init__(
        self,
        asr_model: ASRModel,
        tts_model: tts_kokoro.Synthesizer,
        vad_model: VAD,
        completion_url: str,
        model: str,
        api_key: str | None = None,
        interruptible: bool = True,
        wake_word: str | None = None,
        personality_preprompt: tuple[dict[str, str], ...] = DEFAULT_PERSONALITY_PREPROMPT,
        announcement: str | None = None,
        api_type: str = "openai",
        rag_enabled: bool = True,
        rag_top_k: int = 3,
        manual_trigger_required: bool = False,
        use_mic_speaker: bool = True,
    ) -> None:
        """
        Initialize the Glados voice assistant with configuration parameters.

        This method sets up the voice recognition system, including voice activity detection (VAD),
        automatic speech recognition (ASR), text-to-speech (TTS), and language model processing.
        The initialization configures various components and starts background threads for
        processing LLM responses and TTS output.

        Args:
            voice_model (str): Path to the voice model for text-to-speech synthesis.
            speaker_id (int | None): Identifier for the specific speaker voice, if applicable.
            completion_url (str): URL endpoint for language model completions.
            model (str): Identifier for the language model being used.
            api_key (str | None, optional): Authentication key for the language model API. Defaults to None.
            wake_word (str | None, optional): Activation word to trigger voice assistant. Defaults to None.
            personality_preprompt (list[dict[str, str]], optional): Initial context or personality
                configuration for the language model. Defaults to DEFAULT_PERSONALITY_PREPROMPT.
            announcement (str | None, optional): Initial announcement to be spoken upon initialization.
                Defaults to None.
            interruptible (bool, optional): Whether the assistant's speech can be interrupted.
                Defaults to True.
        """
        self.completion_url = completion_url
        self.model = model
        self.wake_word = wake_word
        self.api_type = api_type
        self._vad_model = vad_model
        self._tts = tts_model
        self._asr_model = asr_model
        self._stc = stc.SpokenTextConverter()

        # warm up onnx ASR model
        self._asr_model.transcribe_file(resource_path("data/0.wav"))

        # Set up headers based on API type
        if api_type == "gemini":
            self.prompt_headers = {
                "Content-Type": "application/json",
            }
            # Automatically format standard Gemini URL to use the configured model
            if "generativelanguage.googleapis.com" in str(completion_url):
                constructed_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
            else:
                constructed_url = str(completion_url)

            if api_key:
                separator = "&" if "?" in constructed_url else "?"
                self.completion_url = f"{constructed_url}{separator}key={api_key}"
            else:
                self.completion_url = constructed_url
        else:
            # Default OpenAI-style headers
            self.prompt_headers = {
                "Authorization": (f"Bearer {api_key}" if api_key else "Bearer your_api_key_here"),
                "Content-Type": "application/json",
            }
            self.completion_url = str(completion_url)

        # RAG configuration
        self.rag_enabled = rag_enabled
        self.rag_top_k = rag_top_k
        self.rag_store = None
        if self.rag_enabled:
            try:
                from .utils.rag_store import RagStore
                self.rag_store = RagStore()
            except Exception as e:
                logger.error(f"Failed to initialize RagStore: {e}")
                self.rag_enabled = False

        self.manual_trigger_required = manual_trigger_required
        self.use_mic_speaker = use_mic_speaker

        # Web UI Event Callbacks
        self.on_state_change = None
        self.on_transcript = None
        self.on_assistant = None
        self.on_assistant_chunk = None
        self.on_rag = None
        self.on_log = None
        self.on_latency_profile = None
        self.on_audio_out = None
        
        self.current_turn_latency = {
            "vad": 32.0,
            "asr": 0.0,
            "rag": 0.0,
            "llm": 0.0,
            "tts": 0.0,
            "prompt_tokens": 0,
            "response_tokens": 0,
            "payload_size_kb": 0.0
        }
        self.is_first_sentence = False

        # Initialize sample queues and state flags
        self._samples: list[NDArray[np.float32]] = []
        self._sample_queue: queue.Queue[tuple[NDArray[np.float32], bool]] = queue.Queue()
        self._buffer: queue.Queue[NDArray[np.float32]] = queue.Queue(maxsize=self.BUFFER_SIZE // self.VAD_SIZE)
        self._incoming_buffer = np.array([], dtype=np.float32)
        self._recording_started = False
        self._gap_counter = 0
        # Manual trigger for noisy environments: when True, start recording even if VAD is low
        self.manual_trigger = False

        self._messages: list[dict[str, str]] = list(personality_preprompt)

        self.llm_queue: queue.Queue[str] = queue.Queue()
        self.tts_queue: queue.Queue[str] = queue.Queue()
        self.audio_queue: queue.Queue[AudioMessage] = queue.Queue()

        self.processing = False
        self.interruptible = interruptible

        self.currently_speaking = threading.Event()
        self.shutdown_event = threading.Event()

        llm_thread = threading.Thread(target=self.process_llm)
        llm_thread.start()

        tts_thread = threading.Thread(target=self.process_tts_thread)
        tts_thread.start()

        audio_thread = threading.Thread(target=self.process_audio_thread)
        audio_thread.start()

        if announcement:
            audio = self._tts.synthesize_audio(announcement)
            logger.success(f"TTS text: {announcement}")
            if self.use_mic_speaker:
                sd.play(audio, self._tts.sample_rate)
                if not self.interruptible:
                    sd.wait()
            elif self.on_audio_out:
                self.on_audio_out(audio, announcement)

        def audio_callback_for_sd_input_stream(
            indata: np.dtype[np.float32],
            frames: int,
            time: sd.CallbackStop,
            status: CallbackFlags,
        ) -> None:
            """
            Callback function for processing audio input from a sounddevice input stream.

            This method is responsible for handling incoming audio samples, performing voice activity detection (VAD),
            and queuing the processed audio data for further analysis.

            Parameters:
                indata (np.ndarray): Input audio data from the sounddevice stream
                frames (int): Number of audio frames in the current chunk
                time (sd.CallbackStop): Timing information for the audio callback
                status (CallbackFlags): Status flags for the audio callback

            Returns:
                None

            Notes:
                - Copies and squeezes the input data to ensure single-channel processing
                - Applies voice activity detection to determine speech presence
                - Puts processed audio samples and VAD confidence into a thread-safe queue
                - Ignores input while TTS is speaking to prevent feedback loops
            """
            # Ignore input while the assistant is speaking to prevent feedback loops
            if self.currently_speaking.is_set():
                return

            data = np.array(indata).copy().squeeze()  # Reduce to single channel if necessary
            vad_value = self._vad_model(np.expand_dims(data, 0))
            vad_confidence = vad_value > self.VAD_THRESHOLD
            self._sample_queue.put((data, bool(vad_confidence)))

        if self.use_mic_speaker:
            self.input_stream = sd.InputStream(
                samplerate=self.SAMPLE_RATE,
                channels=1,
                callback=audio_callback_for_sd_input_stream,
                blocksize=int(self.SAMPLE_RATE * self.VAD_SIZE / 1000),
            )
        else:
            self.input_stream = None

    def push_audio_chunk(self, data: np.ndarray) -> None:
        """Called by external server to push audio chunks"""
        if self.currently_speaking.is_set():
            return
            
        data = data.copy().squeeze()
        self._incoming_buffer = np.concatenate((self._incoming_buffer, data))
        
        chunk_size = int(self.SAMPLE_RATE * self.VAD_SIZE / 1000)
        while len(self._incoming_buffer) >= chunk_size:
            chunk = self._incoming_buffer[:chunk_size]
            self._incoming_buffer = self._incoming_buffer[chunk_size:]
            
            vad_value = self._vad_model(np.expand_dims(chunk, 0))
            vad_confidence = vad_value > self.VAD_THRESHOLD
            self._sample_queue.put((chunk, bool(vad_confidence)))

    def _update_state(self, state: str) -> None:
        """Helper to update internal state and trigger UI callbacks."""
        logger.debug(f"State transition: {state}")
        if self.on_state_change:
            try:
                self.on_state_change(state)
            except Exception as e:
                logger.error(f"Error in on_state_change callback: {e}")

    @property
    def messages(self) -> list[dict[str, str]]:
        """
        Retrieve the current list of conversation messages.

        Returns:
            list[dict[str, str]]: A list of message dictionaries representing the conversation history.
        """
        return self._messages

    @classmethod
    def from_config(cls, config: GladosConfig, use_mic_speaker: bool = True) -> "Glados":
        """
        Create a Glados instance from a GladosConfig configuration object.

        Parameters:
            config (GladosConfig): Configuration object containing Glados initialization parameters

        Returns:
            Glados: A new Glados instance configured with the provided settings
        """
        # Choose ASR model based on configuration
        asr_model = ParakeetASR(asr_corrections=config.asr_corrections)
        vad_model = VAD()

        # Choose TTS model based on voice configuration
        voice = config.voice if config.voice != "glados" else "af_alloy"
        assert voice in tts_kokoro.get_voices(), f"Voice '{voice}' not available"
        tts_model = tts_kokoro.Synthesizer(voice=voice)

        return cls(
            asr_model=asr_model,
            tts_model=tts_model,
            vad_model=vad_model,
            completion_url=config.completion_url,
            model=config.model,
            api_key=config.api_key,
            interruptible=config.interruptible,
            wake_word=config.wake_word,
            announcement=config.announcement,
            personality_preprompt=tuple(config.to_chat_messages()),
            api_type=config.api_type,
            rag_enabled=config.rag_enabled,
            rag_top_k=config.rag_top_k,
            manual_trigger_required=config.manual_trigger_required,
            use_mic_speaker=use_mic_speaker,
        )

    @classmethod
    def from_yaml(cls, path: str) -> "Glados":
        """
        Create a Glados instance from a configuration file.

        Parameters:
            path (str): Path to the YAML configuration file containing Glados settings.

        Returns:
            Glados: A new Glados instance configured with settings from the specified YAML file.

        Example:
            glados = Glados.from_yaml('config/default.yaml')
        """
        return cls.from_config(GladosConfig.from_yaml(path))

    def start_manual_listening(self) -> None:
        """
        Set the manual trigger to start recording on the next audio sample.

        This is intended for noisy environments where automatic VAD may produce false
        positives. Calling this method will cause the pre-activation buffer to start
        recording on the next incoming sample and is cleared once recording begins.
        """
        logger.info("Manual listening triggered by user input")
        self.manual_trigger = True

    def start_listen_event_loop(self) -> None:
        """
        Start the voice assistant's listening event loop, continuously processing audio input.

        This method initializes the audio input stream and enters an infinite loop to handle incoming audio samples.
        The loop retrieves audio samples and their voice activity detection (VAD) confidence from a queue and processes
        each sample using the `_handle_audio_sample` method.

        Behavior:
        - Starts the audio input stream
        - Logs successful initialization of audio modules
        - Enters an infinite listening loop
        - Retrieves audio samples from a queue
        - Processes each audio sample with VAD confidence
        - Handles keyboard interrupts by stopping the input stream and setting a shutdown event

        Raises:
            KeyboardInterrupt: Allows graceful termination of the listening loop
        """
        if self.input_stream:
            self.input_stream.start()
        logger.success("Audio Modules Operational")
        logger.success("Listening...")
        self._update_state("IDLE")
        # Loop forever, but is 'paused' when new samples are not available
        try:
            while True:
                sample, vad_confidence = self._sample_queue.get()
                self._handle_audio_sample(sample, vad_confidence)
        except KeyboardInterrupt:
            self.shutdown_event.set()
            if self.input_stream:
                self.input_stream.stop()

    def _handle_audio_sample(self, sample: NDArray[np.float32], vad_confidence: bool) -> None:
        """
        Handles the processing of each audio sample.

        If the recording has not started, the sample is added to the circular buffer.

        If the recording has started, the sample is added to the samples list, and the pause
        limit is checked to determine when to process the detected audio.

        Args:
            sample (np.ndarray): The audio sample to process.
            vad_confidence (bool): Whether voice activity is detected in the sample.
        """
        # Don't process any audio samples while the assistant is speaking
        if self.currently_speaking.is_set():
            return
            
        if not self._recording_started:
            self._manage_pre_activation_buffer(sample, vad_confidence)
        else:
            self._process_activated_audio(sample, vad_confidence)

    def _manage_pre_activation_buffer(self, sample: NDArray[np.float32], vad_confidence: bool) -> None:
        """
        Manages the pre-activation audio buffer and handles voice activity detection.

        This method maintains a circular buffer of audio samples before voice activation,
        discarding the oldest sample when the buffer is full. When voice activity is detected,
        it stops the audio stream and prepares for audio processing.

        Args:
            sample (np.ndarray): The current audio sample to be added to the buffer.
            vad_confidence (bool): Indicates whether voice activity is detected in the sample.

        Side Effects:
            - Modifies the internal circular buffer
            - Stops the audio stream when voice is detected
            - Disables processing on LLM and TTS threads
            - Prepares samples for recording when voice is detected
        """
        if self._buffer.full():
            self._buffer.get()  # Discard the oldest sample to make room for new ones
        self._buffer.put(sample)

        # Begin recording if user manually triggered it, OR if manual trigger is not required and VAD detected speech
        should_trigger = self.manual_trigger or (not self.manual_trigger_required and vad_confidence)

        if should_trigger:
            if not self.interruptible and self.currently_speaking.is_set():
                logger.info("Interruption is disabled, and the assistant is currently speaking, ignoring new input.")
                return

            sd.stop()  # Stop the audio stream to prevent overlap
            self.processing = False  # Turns off processing on threads for the LLM and TTS!!!
            self._samples = list(self._buffer.queue)
            self._recording_started = True
            self._update_state("LISTENING")
            # Reset manual trigger
            self.manual_trigger = False
        else:
            # If VAD fired but manual trigger is required, ignore the VAD event
            if vad_confidence:
                logger.debug("VAD detected speech but manual trigger is required; ignoring VAD.")

    def _process_activated_audio(self, sample: NDArray[np.float32], vad_confidence: bool) -> None:
        """
        Process audio samples, tracking speech pauses to capture complete utterances.

        This method accumulates audio samples and monitors voice activity detection (VAD) confidence to determine
        when a complete speech segment has been captured. It appends incoming samples to the internal buffer and
        tracks silent gaps to trigger audio processing.

        Parameters:
            sample (np.ndarray): A single audio sample from the input stream
            vad_confidence (bool): Indicates whether voice activity is currently detected

        Side Effects:
            - Appends audio samples to self._samples
            - Increments or resets self._gap_counter
            - Triggers audio processing via self._process_detected_audio() when pause limit is reached
        """

        self._samples.append(sample)

        if not vad_confidence:
            self._gap_counter += 1
            if self._gap_counter >= self.PAUSE_LIMIT // self.VAD_SIZE:
                self._process_detected_audio()
        else:
            self._gap_counter = 0

    def _wakeword_detected(self, text: str) -> bool:
        """
        Check if the detected text contains a close match to the wake word using Levenshtein distance.

        This method helps handle variations in wake word detection by calculating the minimum edit distance
        between detected words and the configured wake word. It accounts for potential misheard
        variations during speech recognition.

        Parameters:
            text (str): The transcribed text to check for wake word similarity

        Returns:
            bool: True if a word in the text is sufficiently similar to the wake word, False otherwise

        Raises:
            AssertionError: If the wake word is not configured (None)

        Notes:
            - Uses Levenshtein distance to measure text similarity
            - Compares each word in the text against the wake word
            - Considers a match if the distance is below a predefined similarity threshold
        """
        assert self.wake_word is not None, "Wake word should not be None"

        words = text.split()
        closest_distance = min([distance(word.lower(), self.wake_word) for word in words])
        return bool(closest_distance < self.SIMILARITY_THRESHOLD)

    def reset(self) -> None:
        """
        Reset the voice recording state and clear all audio buffers.

        This method performs the following actions:
        - Logs a debug message indicating the reset process
        - Stops the current recording by setting `_recording_started` to False
        - Clears the collected audio samples
        - Resets the gap counter used for detecting speech pauses
        - Empties the thread-safe audio buffer queue

        Note:
            Uses a mutex lock to safely clear the shared buffer queue to prevent
            potential race conditions in multi-threaded audio processing.
        """
        logger.debug("Resetting recorder...")
        self._recording_started = False
        self._samples.clear()
        self._gap_counter = 0
        with self._buffer.mutex:
            self._buffer.queue.clear()

    def _process_detected_audio(self) -> None:
        """
        Process detected audio and generate a response after speech pause.

        Transcribes audio samples and handles wake word detection and LLM processing. Manages the
        audio input stream and processing state throughout the interaction.

        Args:
            None

        Returns:
            None

        Side Effects:
            - Stops the input audio stream
            - Performs automatic speech recognition (ASR)
            - Potentially sends text to LLM queue
            - Resets audio recording state
            - Restarts input audio stream

        Raises:
            No explicit exceptions raised
        """
        self._update_state("THINKING")
        logger.debug("Detected pause after speech. Processing...")
        audio_duration_s = sum(len(s) for s in self._samples) / self.SAMPLE_RATE
        logger.success(f"VAD triggered. Processing audio chunk of length {audio_duration_s:.2f}s")
        self.current_turn_latency["vad"] = audio_duration_s * 1000 # in ms
        self._is_voice_turn = True
        
        start_time = time.perf_counter()
        detected_text = self.asr(self._samples)
        asr_duration_ms = (time.perf_counter() - start_time) * 1000
        logger.success(f"ASR execution completed in {asr_duration_ms:.3f}ms")
        self.current_turn_latency["asr"] = asr_duration_ms

        if detected_text:
            logger.success(f"ASR text: '{detected_text}'")
            raw_text = getattr(self._asr_model, "last_raw_transcription", detected_text)
            if self.on_transcript:
                try:
                    self.on_transcript(raw_text, detected_text)
                except Exception as e:
                    logger.error(f"Error in on_transcript callback: {e}")

            if self.wake_word and not self._wakeword_detected(detected_text):
                logger.info(f"Required wake word {self.wake_word=} not detected.")
                self._update_state("IDLE")
            else:
                self.llm_queue.put(detected_text)
                self.processing = True
                # Set speaking flag early to prevent immediate feedback
                self.currently_speaking.set()
                logger.debug("Speaking event set early - preparing TTS response")
        else:
            self._update_state("IDLE")

        self.reset()

    def asr(self, samples: list[NDArray[np.float32]]) -> str:
        """
        Perform automatic speech recognition (ASR) on the provided audio samples.

        Parameters:
            samples (list[np.dtype[np.float32]]): A list of numpy arrays containing audio samples to be transcribed.

        Returns:
            str: The transcribed text from the input audio samples.

        Notes:
            - Concatenates multiple audio samples into a single continuous audio array
            - Uses the pre-configured ASR model to transcribe the audio
        """
        audio = np.concatenate(samples)

        # Normalize audio to [-0.5, 0.5] range to prevent clipping and ensure consistent levels
        audio = audio / np.max(np.abs(audio)) / 2

        detected_text = self._asr_model.transcribe(audio)
        return detected_text

    def _convert_messages_for_gemini(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """
        Convert OpenAI-style messages to Gemini API format.
        
        Args:
            messages: List of OpenAI-style message dictionaries
            
        Returns:
            List of Gemini-style content dictionaries
        """
        gemini_contents = []
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            # Map roles to Gemini format
            if role == "system":
                # System messages become user messages with special formatting in Gemini
                gemini_contents.append({
                    "role": "user",
                    "parts": [{"text": f"System: {content}"}]
                })
            elif role == "user":
                gemini_contents.append({
                    "role": "user", 
                    "parts": [{"text": content}]
                })
            elif role == "assistant":
                gemini_contents.append({
                    "role": "model",
                    "parts": [{"text": content}]
                })
                
        return gemini_contents

    def _create_request_data(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """
        Create request data based on API type.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Request data dictionary formatted for the appropriate API
        """
        if self.api_type == "gemini":
            return {
                "contents": self._convert_messages_for_gemini(messages),
                "generationConfig": {
                    "temperature": 0.7,
                    "topK": 40,
                    "topP": 0.95,
                    "maxOutputTokens": 1024,
                },
                "safetySettings": [
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    },
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH", 
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    }
                ]
            }
        else:
            # Default OpenAI format
            return {
                "model": self.model,
                "stream": True,
                "messages": messages,
            }

    def percentage_played(self, total_samples: int) -> tuple[bool, int]:
        """
        Monitor audio playback progress and return completion status with interrupt detection.

        Streams audio samples through PortAudio and actively tracks the number of samples
        that have been played. The playback can be interrupted by setting self.processing
        to False or self.shutdown_event. Uses a non-blocking callback system with a completion event for
        synchronization.

        Args:
            total_samples: Number of audio samples to be played in total. For example,
                for 1 second of 48kHz audio, this would be 48000.

        Returns:
            A tuple containing:
            - bool: True if playback was interrupted, False if completed normally
            - int: Percentage of samples played (0-100), calculated as
            (played_samples / total_samples * 100)

        Raises:
            sd.PortAudioError: If the audio stream encounters initialization or
                playback errors
            RuntimeError: If stream management fails during execution

        Examples:
            For 1 second of audio at 48kHz:
            >>> interrupted, progress = audio.percentage_played(48000)
            >>> print(f"Interrupted: {interrupted}, Progress: {progress}%")
            Interrupted: False, Progress: 100%

        Implementation Details:
            - Uses a stream callback system to track sample count in real-time
            - Handles interruption via self.processing flag
            - Implements timeout based on audio duration plus 1 second buffer
            - Caps progress percentage at 100 even if more samples are processed
        """
        interrupted = False
        progress = 0
        completion_event = threading.Event()

        if not self.use_mic_speaker:
            # In web mode, we don't use sd.OutputStream. We wait synchronously 
            # while allowing for interruptions from the main thread.
            start_time = time.time()
            expected_duration = total_samples / self._tts.sample_rate
            
            while time.time() - start_time < expected_duration:
                if self.processing is False or self.shutdown_event.is_set():
                    interrupted = True
                    break
                time.sleep(0.01)
                progress = int((time.time() - start_time) * self._tts.sample_rate)
                
            percentage_played = min(int(progress / total_samples * 100), 100)
            return interrupted, percentage_played

        def stream_callback(
            outdata: NDArray[np.float32], frames: int, time: dict[str, Any], status: sd.CallbackFlags
        ) -> tuple[NDArray[np.float32], sd.CallbackStop | None]:
            nonlocal progress, interrupted
            progress += frames
            if self.processing is False or self.shutdown_event.is_set():
                interrupted = True
                completion_event.set()
                return outdata, sd.CallbackStop
            if progress >= total_samples:
                completion_event.set()
            return outdata, None

        try:
            stream = sd.OutputStream(
                callback=stream_callback,
                samplerate=self._tts.sample_rate,
                channels=1,
                finished_callback=completion_event.set,
            )
            with stream:
                # Wait with timeout to allow for interruption
                completion_event.wait(timeout=total_samples / self._tts.sample_rate + 1)

        except (sd.PortAudioError, RuntimeError):
            logger.debug("Audio stream already closed or invalid")

        percentage_played = min(int(progress / total_samples * 100), 100)
        return interrupted, percentage_played

    def process_llm(self) -> None:
        """
        Process text through the Language Model (LLM) and generate conversational responses.

        This method runs in a continuous loop, retrieving detected text from a queue and sending it to an LLM server.
        It streams the response, processes each chunk, and sends processed sentences to the text-to-speech (TTS) queue.

        Key Behaviors:
        - Continuously polls the LLM queue for detected text
        - Sends text to LLM server with streaming enabled
        - Processes response chunks in real-time
        - Breaks sentences at punctuation marks
        - Handles interruptions and processing flags
        - Adds end-of-stream token to TTS queue after processing

        Exceptions:
        - Handles empty queue timeouts
        - Catches and logs errors during line processing
        - Stops processing if shutdown event is set or processing flag is False

        Side Effects:
        - Modifies `self.messages` by appending user messages
        - Puts processed sentences into `self.tts_queue`
        - Logs debug and error information

        Note:
        - Uses a timeout mechanism to prevent blocking
        - Supports graceful interruption of LLM processing
        """
        while not self.shutdown_event.is_set():
            try:
                detected_text = self.llm_queue.get(timeout=0.1)
                
                # Setup voice vs text turn check
                is_voice = getattr(self, "_is_voice_turn", False)
                if is_voice:
                    self._is_voice_turn = False
                else:
                    self.current_turn_latency["vad"] = 0.0
                    self.current_turn_latency["asr"] = 0.0
                
                self.current_turn_latency["rag"] = 0.0
                self.current_turn_latency["llm"] = 0.0
                self.current_turn_latency["tts"] = 0.0
                self.current_turn_latency["prompt_tokens"] = 0
                self.current_turn_latency["response_tokens"] = 0
                self.current_turn_latency["payload_size_kb"] = 0.0
                self.is_first_sentence = True

                self.messages.append({"role": "user", "content": detected_text})

                # Perform RAG query context injection if enabled
                messages_for_llm = self.messages
                if self.rag_enabled and self.rag_store:
                    logger.debug(f"Querying RagStore for context on query: '{detected_text}'")
                    try:
                        start_time = time.perf_counter()
                        retrieved = self.rag_store.search(detected_text, top_k=self.rag_top_k)
                        duration_ms = (time.perf_counter() - start_time) * 1000
                        logger.success(f"NumPy cosine similarity search completed in {duration_ms:.3f}ms")
                        self.current_turn_latency["rag"] = duration_ms
                        
                        if retrieved:
                            context_str = "\n".join([f"- From {c['source']}: {c['text']}" for c in retrieved])
                            logger.success(f"Retrieved {len(retrieved)} context chunks from RagStore.")
                            
                            if self.on_rag:
                                try:
                                    self.on_rag(retrieved)
                                except Exception as e:
                                    logger.error(f"Error in on_rag callback: {e}")
                            
                            # Create a copy so we do not mutate the history list itself
                            messages_for_llm = copy.deepcopy(self.messages)
                            if messages_for_llm and messages_for_llm[-1]["role"] == "user":
                                messages_for_llm[-1]["content"] = (
                                    f"Context from Mentat Archives:\n{context_str}\n\n"
                                    f"Query: {detected_text}"
                                )
                    except Exception as e:
                        logger.error(f"Error executing RAG search: {e}")

                data = self._create_request_data(messages_for_llm)
                logger.debug(f"starting request on {messages_for_llm=}")
                logger.debug("Performing request to LLM server...")

                # Measure prompt token size (characters)
                prompt_chars = sum(len(msg.get("content", "")) for msg in messages_for_llm)
                self.current_turn_latency["prompt_tokens"] = prompt_chars

                # Perform the request and process the stream
                try:
                    start_llm = time.perf_counter()
                    with requests.post(
                        self.completion_url,
                        headers=self.prompt_headers,
                        json=data,
                        stream=(self.api_type != "gemini"),  # Don't stream for Gemini
                    ) as response:
                        llm_latency_ms = (time.perf_counter() - start_llm) * 1000
                        self.current_turn_latency["llm"] = llm_latency_ms

                        if response.status_code != 200:
                            logger.error(f"LLM API returned status code {response.status_code}: {response.text}")
                            self.currently_speaking.clear()
                            self.processing = False
                            self.tts_queue.put("<EOS>")
                            continue

                        if self.api_type == "gemini":
                            # Handle non-streaming Gemini response
                            try:
                                result = response.json()
                                if "error" in result:
                                    logger.error(f"Gemini API returned error: {result['error']}")
                                elif "candidates" in result and len(result["candidates"]) > 0:
                                    content = result["candidates"][0].get("content", {})
                                    parts = content.get("parts", [])
                                    if parts and len(parts) > 0:
                                        full_text = parts[0].get("text", "")
                                        if full_text:
                                            # Record response characters
                                            response_chars = len(full_text)
                                            self.current_turn_latency["response_tokens"] = response_chars
                                            self.current_turn_latency["payload_size_kb"] = (prompt_chars + response_chars) / 1024.0
                                            
                                            if self.on_assistant_chunk:
                                                try:
                                                    self.on_assistant_chunk(full_text)
                                                except Exception as e:
                                                    logger.error(f"Error in on_assistant_chunk callback: {e}")
                                            # Process the complete response
                                            self._process_complete_response(full_text)
                            except Exception as e:
                                logger.error(f"Error processing Gemini response: {e}")
                        else:
                            # Handle streaming response (OpenAI/Ollama format)
                            sentence = []
                            
                            # Check if the response is actually non-streaming (full JSON response)
                            content_type = response.headers.get('content-type', '')
                            if 'text/event-stream' not in content_type:
                                # Non-streaming response - parse as complete JSON
                                try:
                                    result = response.json()
                                    if "choices" in result and len(result["choices"]) > 0:
                                        full_text = result["choices"][0].get("message", {}).get("content", "")
                                        if full_text:
                                            # Record response characters
                                            response_chars = len(full_text)
                                            self.current_turn_latency["response_tokens"] = response_chars
                                            self.current_turn_latency["payload_size_kb"] = (prompt_chars + response_chars) / 1024.0
                                            
                                            if self.on_assistant_chunk:
                                                try:
                                                    self.on_assistant_chunk(full_text)
                                                except Exception as e:
                                                    logger.error(f"Error in on_assistant_chunk callback: {e}")
                                            self._process_complete_response(full_text)
                                except Exception as e:
                                    logger.error(f"Error processing non-streaming OpenAI response: {e}")
                            else:
                                # Streaming response - process line by line
                                accumulated_response = []
                                for line in response.iter_lines():
                                    if self.processing is False:
                                        break  # If the stop flag is set from new voice input, halt processing
                                    if line:  # Filter out empty keep-alive new lines
                                        try:
                                            cleaned_line = self._clean_raw_bytes(line)
                                            if cleaned_line:  # Add check for empty cleaned line
                                                chunk = self._process_chunk(cleaned_line)
                                                if chunk:
                                                    accumulated_response.append(chunk)
                                                    if self.on_assistant_chunk:
                                                        try:
                                                            self.on_assistant_chunk(chunk)
                                                        except Exception as e:
                                                            logger.error(f"Error in on_assistant_chunk callback: {e}")
                                                    sentence.append(chunk)
                                                    # If there is a pause token, send the sentence to the TTS queue
                                                    if (
                                                        chunk
                                                        in [
                                                            ".",
                                                            "!",
                                                            "?",
                                                            ":",
                                                            "?",
                                                            "?!",
                                                            "\n",
                                                            "\n\n",
                                                        ]
                                                        and sentence[-2].isdigit() is False
                                                    ):  # Don't split on numbers!
                                                        logger.info(f"Chunk: {chunk}")
                                                        self._process_sentence(sentence)
                                                        sentence = []
                                        except Exception as e:
                                            logger.error(f"Error processing line: {e}")
                                            continue

                                if self.processing and sentence:
                                    self._process_sentence(sentence)
                                
                                # Record response characters
                                full_text = "".join(accumulated_response)
                                response_chars = len(full_text)
                                self.current_turn_latency["response_tokens"] = response_chars
                                self.current_turn_latency["payload_size_kb"] = (prompt_chars + response_chars) / 1024.0
                        self.tts_queue.put("<EOS>")  # Add end of stream token to the queue
                except Exception as e:
                    logger.error(f"Error communicating with LLM API: {e}")
                    self.currently_speaking.clear()
                    self.processing = False
                    self.tts_queue.put("<EOS>")
            except queue.Empty:
                time.sleep(self.PAUSE_TIME)

    def _process_sentence(self, current_sentence: list[str]) -> None:
        """
        Process a sentence for text-to-speech by cleaning and formatting the input text.

        This method handles text preprocessing for the TTS system, removing special formatting
        and cleaning up the text before adding it to the TTS queue.

        Args:
            current_sentence (list[str]): A list of text fragments to be processed.

        Notes:
            - Removes text enclosed in asterisks (*) and parentheses ()
            - Replaces newlines with periods
            - Removes extra whitespace and colons
            - Only adds non-empty sentences to the TTS queue
        """
        sentence = "".join(current_sentence)
        sentence = re.sub(r"\*.*?\*|\(.*?\)", "", sentence)
        sentence = sentence.replace("\n\n", ". ").replace("\n", ". ").replace("  ", " ").replace(":", " ")
        if sentence:
            self.tts_queue.put(sentence)

    def _process_complete_response(self, full_text: str) -> None:
        """
        Process a complete response from Gemini API by breaking it into sentences.
        
        Args:
            full_text (str): The complete response text from Gemini API
        """
        # Split the response into sentences for TTS processing
        import re
        sentences = re.split(r'[.!?:;]+', full_text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and self.processing:
                # Clean and format the sentence
                sentence = re.sub(r"\*.*?\*|\(.*?\)", "", sentence)
                sentence = sentence.replace("\n\n", ". ").replace("\n", ". ").replace("  ", " ").replace(":", " ")
                if sentence:
                    self.tts_queue.put(sentence)

    def _clean_raw_bytes(self, line: bytes) -> dict[str, Any] | None:
        """
        Cleans the raw bytes from the server and converts to a standardized format.

        Args:
            line (bytes): The raw bytes from the server

        Returns:
            dict or None: Parsed JSON response, or None if parsing fails
        """
        try:
            # Handle OpenAI format
            if line.startswith(b"data: "):
                json_str = line.decode("utf-8")[6:]  # Remove 'data: ' prefix
                if json_str.strip() == "[DONE]":
                    return None
                parsed_json: dict[str, Any] = json.loads(json_str)
                return parsed_json
            # Handle Ollama format or Gemini format
            else:
                parsed_json = json.loads(line.decode("utf-8"))
                if isinstance(parsed_json, dict):
                    return parsed_json
                return None
        except Exception as e:
            logger.warning(f"Failed to parse server response: {e}")
            return None

    def _process_chunk(self, line: dict[str, Any]) -> str | None:
        """
        Process a single chunk of text from the LLM server, extracting content from different response formats.

        This method handles text chunks from different LLM server response formats: OpenAI, Ollama, and Gemini.
        It safely extracts the text content, handling potential missing or malformed data.

        Args:
            line (dict[str, Any]): A dictionary containing the LLM server response chunk.

        Returns:
            str | None: The extracted text content, or None if no content is found or an error occurs.

        Raises:
            No explicit exceptions are raised; errors are logged and result in returning None.
        """
        if not line or not isinstance(line, dict):
            return None

        try:
            # Handle OpenAI format
            if "choices" in line:
                choice = line.get("choices", [{}])[0]
                # Try streaming format first (delta), then non-streaming format (message)
                content = choice.get("delta", {}).get("content") or choice.get("message", {}).get("content")
                return content if content else None
            # Handle Gemini format
            elif "candidates" in line:
                candidates = line.get("candidates", [])
                if candidates and len(candidates) > 0:
                    content = candidates[0].get("content", {})
                    parts = content.get("parts", [])
                    if parts and len(parts) > 0:
                        text = parts[0].get("text", "")
                        return text if text else None
                return None
            # Handle Ollama format
            else:
                content = line.get("message", {}).get("content")
                return content if content else None
        except Exception as e:
            logger.error(f"Error processing chunk: {e}")
            return None

    def process_tts_thread(self) -> None:
        """
        Processes text-to-speech (TTS) generation and playback in a dedicated thread.

        This method continuously retrieves generated text from the TTS queue and converts it to spoken audio.
        It manages the lifecycle of TTS output, including handling interruptions, tracking playback
        progress, and updating conversation messages.

        The method runs until the shutdown event is triggered and handles several key scenarios:
        - Generating speech audio from text
        - Playing audio through the default sound device
        - Detecting and handling audio interruptions
        - Tracking and logging TTS performance metrics
        - Managing conversation message history

        Attributes:
            assistant_text (list[str]): Accumulates text generated by the assistant for current response
            system_text (list[str]): Stores text logged when TTS is interrupted
            finished (bool): Indicates completion of TTS generation
            interrupted (bool): Signals whether TTS playback was interrupted

        Raises:
            queue.Empty: When no text is available in the TTS queue within the specified timeout
        """
        while not self.shutdown_event.is_set():
            try:
                generated_text = self.tts_queue.get(timeout=self.PAUSE_TIME)

                if generated_text == "<EOS>":
                    self.audio_queue.put(AudioMessage(np.array([]), "", is_eos=True))
                elif not generated_text:
                    logger.warning("Empty string sent to TTS")
                else:
                    start = time.time()
                    spoken_text = self._stc.text_to_spoken(generated_text)
                    audio = self._tts.synthesize_audio(spoken_text)
                    tts_duration_ms = (time.time() - start) * 1000
                    logger.info(
                        f"TTS Complete, inference: {tts_duration_ms / 1000:.2f}s, "
                        f"length: {len(audio) / self._tts.sample_rate:.2f}s"
                    )

                    if self.is_first_sentence:
                        self.current_turn_latency["tts"] = tts_duration_ms
                        self.is_first_sentence = False
                        if self.on_latency_profile:
                            try:
                                self.on_latency_profile(dict(self.current_turn_latency))
                            except Exception as e:
                                logger.error(f"Error in on_latency_profile callback: {e}")

                    if len(audio):
                        self.audio_queue.put(AudioMessage(audio, spoken_text))

            except queue.Empty:
                pass

    def process_audio_thread(self) -> None:
        """Process audio from the TTS queue and play it through the default sound device

        This method continuously retrieves audio messages from the audio queue and plays them through the default sound
        device. It manages the lifecycle of audio output, including handling interruptions, tracking playback progress,
        and updating conversation messages.

        Attributes:
            assistant_text (list[str]): Accumulates text generated by the assistant for current response
            system_text (list[str]): Stores text logged when TTS is interrupted

        Raises:
            queue.Empty: When no audio is available in the audio queue within the specified timeout.
        """
        assistant_text: list[str] = []
        system_text: list[str] = []

        while not self.shutdown_event.is_set():
            try:
                audio_msg = self.audio_queue.get(timeout=self.PAUSE_TIME)

                if audio_msg.is_eos:
                    logger.debug("Processing end of stream")
                    # End of stream - append complete message
                    if assistant_text:
                        logger.debug(f"Appending assistant message: {' '.join(assistant_text)}")
                        self.messages.append({"role": "assistant", "content": " ".join(assistant_text)})
                    assistant_text = []
                    
                    # Add a small delay to prevent immediate microphone pickup
                    time.sleep(0.2)
                    
                    self.currently_speaking.clear()
                    self._update_state("IDLE")
                    logger.debug("Speaking event cleared - ASR/VAD re-enabled")
                    continue

                if len(audio_msg.audio):
                    # Set speaking flag immediately when we start playing audio
                    if not self.currently_speaking.is_set():
                        self.currently_speaking.set()
                        logger.debug("Speaking event set - ASR/VAD disabled")
                    self._update_state("SPEAKING")
                    if self.on_assistant:
                        try:
                            self.on_assistant(audio_msg.text)
                        except Exception as e:
                            logger.error(f"Error in on_assistant callback: {e}")
                    
                    if self.use_mic_speaker:
                        sd.play(audio_msg.audio, self._tts.sample_rate)
                    elif self.on_audio_out:
                        try:
                            self.on_audio_out(audio_msg.audio, audio_msg.text)
                        except Exception as e:
                            logger.error(f"Error in on_audio_out callback: {e}")
                    
                    total_samples = len(audio_msg.audio)

                    logger.success(f"TTS text: {audio_msg.text}")

                    interrupted, percentage_played = self.percentage_played(total_samples)

                    if interrupted:
                        clipped_text = self.clip_interrupted_sentence(audio_msg.text, percentage_played)
                        logger.success(f"TTS interrupted at {percentage_played}%: {clipped_text}")

                        system_text = copy.deepcopy(assistant_text)
                        system_text.append(clipped_text)

                        # Add interrupted message
                        self.messages.append({"role": "assistant", "content": " ".join(system_text)})
                        assistant_text = []

                        self.currently_speaking.clear()
                        self._update_state("IDLE")
                        logger.debug("Speaking event cleared after interruption - ASR/VAD re-enabled")

                        # Clear remaining audio queue
                        with self.audio_queue.mutex:
                            self.audio_queue.queue.clear()
                    else:
                        assistant_text.append(audio_msg.text)

            except queue.Empty:
                pass

    def clip_interrupted_sentence(self, generated_text: str, percentage_played: float) -> str:
        """
        Clips the generated text based on the percentage of audio played before interruption.

        Truncates the text proportionally to the percentage of audio played and appends an
        interruption marker if the text was cut short.

        Args:
            generated_text (str): The complete text generated by the language model.
            percentage_played (float): Percentage of audio played before interruption (0-100).

        Returns:
            str: Truncated text with an optional interruption marker.

        Example:
            >>> assistant.clip_interrupted_sentence("Hello world how are you today", 50)
            "Hello world<INTERRUPTED>"
        """
        tokens = generated_text.split()
        words_to_print = round((percentage_played / 100) * len(tokens))
        text = " ".join(tokens[:words_to_print])

        # If the TTS was cut off, make that clear
        if words_to_print < len(tokens):
            text = text + "<INTERRUPTED>"
        return text


def start() -> None:
    """Set up the LLM server and start GlaDOS.

    This function reads the configuration file, initializes the Glados voice assistant,
    and starts the listening event loop.

    Raises:
        FileNotFoundError: If the configuration file is not found.
        yaml.YAMLError: If there is an error parsing the YAML configuration file.
    """
    glados_config = GladosConfig.from_yaml("glados_config.yaml")
    glados = Glados.from_config(glados_config)
    glados.start_listen_event_loop()


if __name__ == "__main__":
    start()
