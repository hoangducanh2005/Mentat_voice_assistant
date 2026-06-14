// WebSocket connection manager
let socket = null;
let reconnectTimer = null;
const socketUrl = `ws://${window.location.host}/ws`;

// Audio Waveform Animation variables
const canvas = document.getElementById('oscilloscope');
const ctx = canvas.getContext('2d');
let animationFrameId = null;
let currentState = 'IDLE';

// Animation configs per state
const stateConfigs = {
    'IDLE': {
        waveCount: 1,
        amplitude: 6,
        frequency: 0.012,
        speed: 0.04,
        color: '#8b9bb4',
        glow: 'rgba(139, 155, 180, 0.2)'
    },
    'LISTENING': {
        waveCount: 3,
        amplitude: 28,
        frequency: 0.025,
        speed: 0.16,
        color: '#00e5ff',
        glow: 'rgba(0, 229, 255, 0.4)'
    },
    'THINKING': {
        waveCount: 2,
        amplitude: 12,
        frequency: 0.04,
        speed: 0.08,
        color: '#d27d2d',
        glow: 'rgba(210, 125, 45, 0.4)'
    },
    'SPEAKING': {
        waveCount: 2,
        amplitude: 22,
        frequency: 0.018,
        speed: 0.12,
        color: '#39ff14',
        glow: 'rgba(57, 255, 20, 0.4)'
    }
};

let wavePhase = 0;

// Setup Clock
function updateClock() {
    const now = new Date();
    const hrs = String(now.getHours()).padStart(2, '0');
    const mins = String(now.getMinutes()).padStart(2, '0');
    const secs = String(now.getSeconds()).padStart(2, '0');
    const ms = String(now.getMilliseconds()).padStart(3, '0');
    document.getElementById('system-time').textContent = `${hrs}:${mins}:${secs}.${ms}`;
    requestAnimationFrame(updateClock);
}
requestAnimationFrame(updateClock);

// Canvas Waveform Animation Loop
function drawWave() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Auto-resize canvas buffer if size changes
    const rect = canvas.getBoundingClientRect();
    if (canvas.width !== rect.width || canvas.height !== rect.height) {
        canvas.width = rect.width;
        canvas.height = rect.height;
    }

    const config = stateConfigs[currentState] || stateConfigs['IDLE'];
    const centerY = canvas.height / 2;
    const width = canvas.width;
    
    wavePhase += config.speed;

    // Speeches can modulate the amplitude randomly to make it look alive
    let currentAmplitude = config.amplitude;
    if (currentState === 'SPEAKING') {
        currentAmplitude = config.amplitude * (0.4 + Math.random() * 0.6);
    }

    // Draw overlapping waves
    for (let w = 0; w < config.waveCount; w++) {
        ctx.beginPath();
        ctx.lineWidth = w === 0 ? 2.5 : 1.2;
        ctx.strokeStyle = config.color;
        
        // Add transparency to secondary waves
        if (w > 0) {
            ctx.strokeStyle = config.color + '66'; // 40% opacity hex
        }

        const offsetPhase = wavePhase + (w * Math.PI / 2);
        
        for (let x = 0; x < width; x++) {
            // Calculate a horizontal envelope to taper waves at edges
            const envelope = Math.sin((x / width) * Math.PI);
            const y = centerY + Math.sin(x * config.frequency + offsetPhase) * currentAmplitude * envelope;
            
            if (x === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
        
        ctx.shadowBlur = w === 0 ? 8 : 0;
        ctx.shadowColor = config.color;
        ctx.stroke();
    }
    
    ctx.shadowBlur = 0; // Reset
    animationFrameId = requestAnimationFrame(drawWave);
}

// Word highlight utility comparing raw vs corrected
function getHighlightedText(raw, corrected) {
    if (!raw || !corrected) return corrected || '';
    
    // Normalize and extract words for comparison
    const rawClean = raw.toLowerCase().replace(/[^\w\s']/g, '');
    const rawWords = rawClean.split(/\s+/).filter(Boolean);
    
    const correctedWords = corrected.split(/\s+/);
    
    return correctedWords.map(word => {
        const cleanWord = word.toLowerCase().replace(/[^\w']/g, '');
        if (cleanWord && !rawWords.includes(cleanWord)) {
            return `<span class="corrected-word">${word}</span>`;
        }
        return word;
    }).join(' ');
}

// Append log helper
function addLogLine(log) {
    const terminal = document.getElementById('terminal-body');
    const line = document.createElement('div');
    line.className = `log-line level-${log.level.toLowerCase()}`;
    
    // Structured format
    line.innerHTML = `<span style="color:#7f8c8d">[${log.time}]</span> <span style="color:var(--accent-amber)">${log.name}:${log.line}</span> - ${log.message}`;
    
    terminal.appendChild(line);
    
    // Keep max 200 logs
    while (terminal.childElementCount > 200) {
        terminal.removeChild(terminal.firstChild);
    }
    
    terminal.scrollTop = terminal.scrollHeight;
}

// Append system log helper
function addSystemLog(msg) {
    const time = new Date().toLocaleTimeString();
    addLogLine({
        time: time,
        level: 'SYSTEM',
        name: 'UI_WEB',
        line: 0,
        message: msg
    });
}

// RAG Cards Renderer
function renderRagContext(chunks) {
    const listContainer = document.getElementById('rag-list');
    listContainer.innerHTML = '';
    
    if (!chunks || chunks.length === 0) {
        listContainer.innerHTML = '<div class="no-rag-data">No active retrieval context loaded.</div>';
        return;
    }
    
    chunks.forEach(chunk => {
        const card = document.createElement('div');
        card.className = 'rag-card';
        
        const scorePct = (chunk.score * 100).toFixed(1);
        
        // Extract file name from absolute path if applicable
        let sourceName = chunk.source;
        if (sourceName.includes('/') || sourceName.includes('\\')) {
            sourceName = sourceName.split(/[/\\]/).pop();
        }

        card.innerHTML = `
            <div class="rag-card-header">
                <span class="rag-source" title="${chunk.source}">${sourceName}</span>
                <span class="rag-score">${scorePct}% MATCH</span>
            </div>
            <div class="rag-snippet">"${chunk.text}"</div>
        `;
        listContainer.appendChild(card);
    });
}

// WebSocket setup
function connectWebSocket() {
    addSystemLog('Establishing telemetry socket link...');
    socket = new WebSocket(socketUrl);
    
    socket.onopen = () => {
        addSystemLog('Telemetry socket link operational.');
        if (reconnectTimer) {
            clearInterval(reconnectTimer);
            reconnectTimer = null;
        }
    };
    
    socket.onclose = () => {
        addSystemLog('Telemetry link terminated. Retrying in 2 seconds...');
        if (!reconnectTimer) {
            reconnectTimer = setInterval(connectWebSocket, 2000);
        }
    };
    
    socket.onerror = (err) => {
        console.error('Socket error:', err);
    };

    let firstChunk = true;
    
    socket.onmessage = (event) => {
        try {
            const msg = JSON.parse(event.data);
            
            switch(msg.type) {
                case 'state':
                    updateState(msg.data);
                    break;
                    
                case 'transcript':
                    document.getElementById('raw-transcript').textContent = msg.data.corrected || msg.data.raw || 'No input...';
                    break;
                    
                case 'assistant_chunk':
                    const assistantEl = document.getElementById('assistant-text');
                    if (firstChunk) {
                        assistantEl.textContent = '';
                        firstChunk = false;
                    }
                    assistantEl.textContent += msg.data;
                    break;

                case 'assistant_sentence':
                    // If we receive a full sentence (e.g. from TTS synthesis thread), we can also backup display it
                    // but the chunk-based stream is faster and covers character streaming.
                    break;
                    
                case 'rag':
                    renderRagContext(msg.data);
                    break;
                    
                case 'log':
                    addLogLine(msg.data);
                    // Update telemetry gauges if performance info is logged
                    parseTelemetryFromLog(msg.data.message);
                    break;
            }
        } catch(e) {
            console.error('Error parsing WebSocket message:', e);
        }
    };

    // Helper to clear assistant chunk buffer on thinking state
    function updateState(state) {
        currentState = state;
        document.body.className = `industrial-dark state-${state.toLowerCase()}`;
        document.getElementById('status-indicator').className = `status-indicator`;
        document.getElementById('status-text').textContent = state;
        
        if (state === 'THINKING') {
            document.getElementById('assistant-text').textContent = 'Generating response...';
            firstChunk = true;
        }
    }
}

// Telemetry values parsing from standard Loguru output
function parseTelemetryFromLog(message) {
    // 1. ASR Execution time
    if (message.includes('ASR execution completed in')) {
        const match = message.match(/ASR execution completed in ([\d.]+)ms/);
        if (match) {
            const ms = parseFloat(match[1]);
            document.getElementById('telemetry-asr-val').textContent = `${ms.toFixed(1)} ms`;
            // Normal scale: 0 to 1000ms
            const pct = Math.min((ms / 1000) * 100, 100);
            document.getElementById('telemetry-asr-bar').style.width = `${pct}%`;
        }
    }
    // 2. RAG execution time
    else if (message.includes('NumPy cosine similarity search completed in')) {
        const match = message.match(/NumPy cosine similarity search completed in ([\d.]+)ms/);
        if (match) {
            const ms = parseFloat(match[1]);
            document.getElementById('telemetry-rag-val').textContent = `${ms.toFixed(1)} ms`;
            // Normal scale: 0 to 100ms
            const pct = Math.min((ms / 100) * 100, 100);
            document.getElementById('telemetry-rag-bar').style.width = `${pct}%`;
        }
    }
    // 3. Audio chunk length
    else if (message.includes('VAD triggered. Processing audio chunk of length')) {
        const match = message.match(/Processing audio chunk of length ([\d.]+)s/);
        if (match) {
            const sec = parseFloat(match[1]);
            document.getElementById('telemetry-audio-val').textContent = `${sec.toFixed(2)} s`;
            // Normal scale: 0 to 8s
            const pct = Math.min((sec / 8) * 100, 100);
            document.getElementById('telemetry-audio-bar').style.width = `${pct}%`;
        }
    }
}

// Setup Event Listeners
document.getElementById('btn-trigger').addEventListener('click', () => {
    if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({ type: 'manual_trigger' }));
        addSystemLog('Operator triggered ASR recording.');
    }
});

document.getElementById('btn-reset').addEventListener('click', () => {
    if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({ type: 'reset' }));
        addSystemLog('Operator requested assistant reset.');
        document.getElementById('raw-transcript').textContent = 'Awaiting auditory signal...';
        document.getElementById('assistant-text').textContent = 'System ready. Awaiting instruction.';
        document.getElementById('rag-list').innerHTML = '<div class="no-rag-data">No active retrieval context loaded.</div>';
    }
});

// Boot operations
connectWebSocket();
drawWave();
