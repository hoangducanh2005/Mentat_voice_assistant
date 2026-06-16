const workletCode = `
class RecorderProcessor extends AudioWorkletProcessor {
  process(inputs, outputs, parameters) {
    const input = inputs[0];
    if (input.length > 0) {
      const channelData = input[0];
      // Send data to main thread
      this.port.postMessage(channelData);
    }
    return true;
  }
}
registerProcessor('recorder-processor', RecorderProcessor);
`;

const workletUrl = URL.createObjectURL(new Blob([workletCode], { type: 'application/javascript' }));

let audioContext;
let ws;
let ttsSampleRate = 24000;
let isRecording = false;
let mediaStream;

const statusEl = document.getElementById('status');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const transcriptEl = document.getElementById('transcript');
const responseEl = document.getElementById('response');

function connect() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${window.location.host}/ws/audio`);
    ws.binaryType = 'arraybuffer';

    ws.onopen = () => {
        statusEl.textContent = 'Trạng thái: Đã kết nối tới Server';
        startBtn.disabled = false;
    };

    ws.onmessage = async (event) => {
        if (typeof event.data === 'string') {
            const data = JSON.parse(event.data);
            if (data.type === 'config') {
                ttsSampleRate = data.tts_sample_rate;
            } else if (data.type === 'state') {
                statusEl.textContent = `Trạng thái: ${data.state}`;
            } else if (data.type === 'transcript') {
                transcriptEl.textContent = data.clean;
            } else if (data.type === 'audio_text') {
                responseEl.textContent = data.text;
            }
        } else if (event.data instanceof ArrayBuffer) {
            playAudio(event.data);
        }
    };

    ws.onclose = () => {
        statusEl.textContent = 'Trạng thái: Mất kết nối (Đang thử lại...)';
        startBtn.disabled = true;
        stopBtn.disabled = true;
        setTimeout(connect, 3000);
    };
}

const audioQueue = [];
let isPlaying = false;
let playContext;

async function playAudio(arrayBuffer) {
    if (!playContext) {
        playContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: ttsSampleRate });
    }
    
    // Resume context if it was suspended (browser policy)
    if (playContext.state === 'suspended') {
        await playContext.resume();
    }
    
    const float32Array = new Float32Array(arrayBuffer);
    const audioBuffer = playContext.createBuffer(1, float32Array.length, ttsSampleRate);
    audioBuffer.copyToChannel(float32Array, 0);
    
    audioQueue.push(audioBuffer);
    
    if (!isPlaying) {
        playNextInQueue();
    }
}

function playNextInQueue() {
    if (audioQueue.length === 0) {
        isPlaying = false;
        return;
    }
    
    isPlaying = true;
    const buffer = audioQueue.shift();
    const source = playContext.createBufferSource();
    source.buffer = buffer;
    source.connect(playContext.destination);
    
    source.onended = () => {
        playNextInQueue();
    };
    
    source.start();
}

async function startRecording() {
    try {
        // Request exactly 16000Hz for ASR compatibility
        audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
        
        // Resume context
        if (audioContext.state === 'suspended') {
            await audioContext.resume();
        }

        mediaStream = await navigator.mediaDevices.getUserMedia({ 
            audio: {
                channelCount: 1,
                sampleRate: 16000
            } 
        });
        
        await audioContext.audioWorklet.addModule(workletUrl);
        const source = audioContext.createMediaStreamSource(mediaStream);
        const processor = new AudioWorkletNode(audioContext, 'recorder-processor');
        
        processor.port.onmessage = (e) => {
            if (isRecording && ws.readyState === WebSocket.OPEN) {
                const float32Array = e.data; 
                ws.send(float32Array.buffer);
            }
        };
        
        source.connect(processor);
        processor.connect(audioContext.destination);
        
        isRecording = true;
        startBtn.disabled = true;
        stopBtn.disabled = false;
        statusEl.textContent = 'Trạng thái: Đang nghe... (Hãy nói gì đó)';
        
        // If we also need to resume playContext
        if (playContext && playContext.state === 'suspended') {
            await playContext.resume();
        }
        
    } catch (e) {
        alert('Lỗi truy cập micro: ' + e.message);
        console.error(e);
    }
}

function stopRecording() {
    isRecording = false;
    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
    }
    if (audioContext) {
        audioContext.close();
    }
    startBtn.disabled = false;
    stopBtn.disabled = true;
    statusEl.textContent = 'Trạng thái: Đã dừng thu âm';
}

startBtn.onclick = startRecording;
stopBtn.onclick = stopRecording;

// Initialize WebSocket connection
connect();
