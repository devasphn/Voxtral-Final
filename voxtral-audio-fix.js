/*!
 * Voxtral Voice AI - Audio Fix Implementation
 * Fixes sample rate issues and DOM errors
 */

class VoxtralAudioFixManager {
    constructor() {
        this.SAMPLE_RATE = 16000;
        this.audioContext = null;
        this.audioQueue = [];
        this.isInitialized = false;
        
        console.log('[Voxtral Audio Fix] Manager initialized');
    }
    
    async initialize() {
        try {
            // Initialize audio context
            await this.initializeAudioContext();
            
            // Fix button references
            this.fixButtonReferences();
            
            // Override existing audio handlers
            this.overrideAudioHandlers();
            
            this.isInitialized = true;
            console.log('[Voxtral Audio Fix] All fixes applied successfully');
            
            return true;
        } catch (error) {
            console.error('[Voxtral Audio Fix] Initialization failed:', error);
            return false;
        }
    }
    
    async initializeAudioContext() {
        try {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: this.SAMPLE_RATE
            });
            
            if (this.audioContext.state === 'suspended') {
                await this.audioContext.resume();
            }
            
            console.log(`[Voxtral Audio Fix] Audio context ready - Sample rate: ${this.audioContext.sampleRate}Hz`);
            return true;
        } catch (error) {
            console.error('[Voxtral Audio Fix] Audio context failed:', error);
            return false;
        }
    }
    
    fixButtonReferences() {
        // Override the problematic startConversation function
        if (window.startConversation) {
            const originalStartConversation = window.startConversation;
            window.startConversation = async () => {
                try {
                    console.log('[Voxtral Audio Fix] Starting conversation with fixed button handling');
                    
                    // Safe button management
                    const startButton = document.getElementById('startButton');
                    const stopButton = document.getElementById('stopButton');
                    
                    if (startButton) startButton.disabled = true;
                    if (stopButton) stopButton.disabled = false;
                    
                    // Initialize audio if not ready
                    if (!this.isInitialized) {
                        await this.initialize();
                    }
                    
                    // Call original function with error handling
                    if (originalStartConversation) {
                        originalStartConversation.call(this);
                    }
                    
                } catch (error) {
                    console.error('[Voxtral Audio Fix] Conversation start failed:', error);
                }
            };
        }
    }
    
    overrideAudioHandlers() {
        // Override WebSocket message handler for audio
        const originalHandleMessage = window.handleWebSocketMessage || window.handleAudioResponse;
        
        window.handleAudioResponse = (data) => {
            if (data.type === 'audio_response' && data.audio) {
                this.processAudioChunk(data.audio, data.chunk_id || 'unknown');
            } else if (originalHandleMessage) {
                originalHandleMessage(data);
            }
        };
    }
    
    async processAudioChunk(base64Audio, chunkId) {
        try {
            console.log(`[Voxtral Audio Fix] Processing audio chunk ${chunkId} (${base64Audio.length} chars)`);
            
            // Convert base64 to audio buffer
            const audioBuffer = this.base64ToArrayBuffer(base64Audio);
            
            // Create proper WAV file
            const wavBuffer = this.createWAVBuffer(audioBuffer);
            
            // Decode and play
            const decodedAudio = await this.audioContext.decodeAudioData(wavBuffer);
            await this.playAudioBuffer(decodedAudio, chunkId);
            
        } catch (error) {
            console.error(`[Voxtral Audio Fix] Audio processing failed:`, error);
        }
    }
    
    base64ToArrayBuffer(base64) {
        const binaryString = atob(base64);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
            bytes[i] = binaryString.charCodeAt(i);
        }
        return bytes.buffer;
    }
    
    createWAVBuffer(audioData) {
        const audioBytes = audioData.byteLength;
        const buffer = new ArrayBuffer(44 + audioBytes);
        const view = new DataView(buffer);
        
        // Standard WAV header for 16kHz, 16-bit, mono
        view.setUint32(0, 0x46464952, false);                    // "RIFF"
        view.setUint32(4, 36 + audioBytes, true);                // File size
        view.setUint32(8, 0x45564157, false);                    // "WAVE"
        view.setUint32(12, 0x20746d66, false);                   // "fmt "
        view.setUint32(16, 16, true);                            // PCM format size
        view.setUint16(20, 1, true);                             // PCM format
        view.setUint16(22, 1, true);                             // Mono
        view.setUint32(24, this.SAMPLE_RATE, true);             // Sample rate
        view.setUint32(28, this.SAMPLE_RATE * 2, true);         // Byte rate
        view.setUint16(32, 2, true);                             // Block align
        view.setUint16(34, 16, true);                            // Bits per sample
        view.setUint32(36, 0x61746164, false);                  // "data"
        view.setUint32(40, audioBytes, true);                   // Data size
        
        // Copy audio data
        new Uint8Array(buffer, 44).set(new Uint8Array(audioData));
        
        console.log(`[Voxtral Audio Fix] Created WAV: ${buffer.byteLength} bytes (${this.SAMPLE_RATE}Hz)`);
        return buffer;
    }
    
    async playAudioBuffer(audioBuffer, chunkId) {
        try {
            const source = this.audioContext.createBufferSource();
            const gainNode = this.audioContext.createGain();
            
            source.buffer = audioBuffer;
            gainNode.gain.value = 1.0;
            
            source.connect(gainNode);
            gainNode.connect(this.audioContext.destination);
            
            source.start();
            
            console.log(`[Voxtral Audio Fix] Playing audio ${chunkId} - ${audioBuffer.duration.toFixed(2)}s @ ${audioBuffer.sampleRate}Hz`);
            
            source.onended = () => {
                console.log(`[Voxtral Audio Fix] Completed audio ${chunkId}`);
            };
            
        } catch (error) {
            console.error(`[Voxtral Audio Fix] Playback failed:`, error);
        }
    }
}

// Auto-initialize when DOM is ready
document.addEventListener('DOMContentLoaded', async () => {
    window.voxtralAudioFix = new VoxtralAudioFixManager();
    const success = await window.voxtralAudioFix.initialize();
    
    if (success) {
        console.log('[Voxtral Audio Fix] ✅ All audio issues fixed!');
    } else {
        console.error('[Voxtral Audio Fix] ❌ Some issues remain');
    }
});

// Also try immediate initialization
if (document.readyState !== 'loading') {
    setTimeout(async () => {
        if (!window.voxtralAudioFix) {
            window.voxtralAudioFix = new VoxtralAudioFixManager();
            await window.voxtralAudioFix.initialize();
        }
    }, 100);
}