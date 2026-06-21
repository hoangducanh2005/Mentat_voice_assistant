# Dune Mentat Voice Assistant

[Demo 01: Mentat voice assistant demo](https://youtu.be/ovSiHtjRS1c)
[Demo 02: Mentat voice assistant demo](https://www.youtube.com/watch?v=UqgPc0JgK1I)

## Overview
The **Dune Mentat Voice Assistant** is an offline-first, low-latency voice assistant that responds entirely in English. It simulates the highly logical and analytical personality of a **Mentat**—a biological human computer from Frank Herbert's *Dune* universe.

This project combines highly optimized local machine learning models with a **lightweight Retrieval-Augmented Generation (RAG) pipeline** designed to run efficiently even on resource-constrained hardware like a Raspberry Pi. 

## Key Features
*   **VAD (Voice Activity Detection)**: Real-time filtering of ambient noise and speech boundary detection using `Silero VAD`.
*   **ASR (Automatic Speech Recognition)**: Rapid transcription of spoken English into text using `Nemo-Parakeet TDT-CTC` (ONNX).
*   **Lightweight RAG Engine**: Performs semantic search on Dune wiki archives using a pure **NumPy-based Cosine Similarity** vector store. It completely avoids heavy vector databases like ChromaDB or FAISS.
*   **LLM Engine**: Merges retrieved context with the user's query and leverages APIs (Gemini/OpenAI) to generate lore-accurate, Mentat-style responses.
*   **Text Normalizer**: Automatically converts complex text formatting (numbers, symbols, percentages, decimals) into fully pronounceable words for the TTS engine.
*   **TTS (Text-to-Speech)**: High-quality, natural-sounding voice synthesis using `Kokoro v1.0` (ONNX).
*   **TUI & Web Dashboard**: Includes a beautiful terminal UI (built with Textual) and a Cockpit web dashboard for real-time monitoring of system states and logs.

## Quick Start

### 1. Installation
Initialize a Python 3.12 environment using `uv` (recommended for speed) and install dependencies:
```bash
uv venv --python 3.12
# Activate the environment (e.g., .venv\Scripts\activate on Windows or source .venv/bin/activate on Unix)
uv pip install -e .[cpu]
```

### 2. Download Models
Download all required ML models (ASR, VAD, TTS, Phonemizer) locally:
```bash
uv run glados download
```

### 3. Run the Assistant
You can run the assistant in three different modes:

**Terminal UI (TUI) Mode:** (Recommended)
Displays a beautiful, retro sci-fi terminal interface.
```bash
uv run glados tui
```

**Web Dashboard Mode:**
Starts the engine and a web-based telemetry dashboard accessible at `http://127.0.0.1:8000`.
```bash
uv run glados dashboard
```

**Headless Mode:**
Starts the assistant purely in the background CLI.
```bash
uv run glados start
```

## Configuration
Edit the `configs/glados_config.yaml` file to adjust the LLM provider, customize the TTS voice, set wake words, and fine-tune the Mentat persona preprompts.

## Project Structure & Architecture
The system operates as a continuous listen-compute-speak loop:
1. **Microphone Input** -> **VAD** isolates active speech.
2. **ASR** transcribes the raw audio waveform.
3. **RAG** embeds the query and searches the pre-computed `data/vectors.npz` NumPy file.
4. **LLM Engine** processes the combined query + context.
5. **Text Normalizer** transforms raw text (e.g., `$50`) into spoken words (e.g., `fifty dollars`).
6. **TTS Engine** synthesizes the audio and plays it back.
