"use strict";

class AlertSound {
    constructor({enabled, status, createContext, now = () => performance.now()}) {
        this.enabled = enabled;
        this.status = status;
        this.createContext = createContext;
        this.now = now;
        this.context = null;
        this.activeTone = null;
        this.lastBeep = -Infinity;
    }

    async enable() {
        if (!this.enabled()) return false;
        try {
            if (!this.context || this.context.state === "closed") {
                this.context = this.createContext();
                this.context.onstatechange = () => this.updateStatus();
            }
            // Called directly from Start, the checkbox, or Test sound (user gestures).
            await this.context.resume();
            this.updateStatus();
            return this.enabled() && this.context.state === "running";
        } catch {
            this.status("Sound unavailable. Use Test sound to retry.");
            return false;
        }
    }

    updateStatus() {
        this.status(!this.enabled() ? "Sound off" : this.context?.state === "running"
            ? "Sound ready · critical alerts every 5 seconds"
            : "Sound paused. Click Test sound to enable playback.");
    }

    play(critical = true) {
        if (!this.enabled()) return false;
        if (this.context?.state !== "running") {
            this.updateStatus();
            return false; // Never queue a stale alarm in a suspended context.
        }
        if (critical && this.now() - this.lastBeep < 5000) return false;
        let tone, gain;
        try {
            this.stop();
            const audio = this.context;
            tone = audio.createOscillator();
            gain = audio.createGain();
            tone.frequency.value = 740;
            gain.gain.setValueAtTime(.001, audio.currentTime);
            gain.gain.exponentialRampToValueAtTime(.12, audio.currentTime + .02);
            gain.gain.exponentialRampToValueAtTime(.001, audio.currentTime + .35);
            tone.connect(gain);
            gain.connect(audio.destination);
            tone.onended = () => {
                tone.disconnect();
                gain.disconnect();
                if (this.activeTone === tone) this.activeTone = null;
            };
            tone.start();
            tone.stop(audio.currentTime + .35);
            this.activeTone = tone;
            if (critical) this.lastBeep = this.now();
            return true;
        } catch {
            try { tone?.stop(); } catch { /* May not have started. */ }
            tone?.disconnect();
            gain?.disconnect();
            this.status("Sound unavailable. Use Test sound to retry.");
            return false;
        }
    }

    stop() {
        if (this.activeTone) {
            try { this.activeTone.stop(); } catch { /* Already ended. */ }
            this.activeTone = null;
        }
    }

    reset() {
        this.stop();
        this.lastBeep = -Infinity;
    }
}

if (typeof module !== "undefined") module.exports = AlertSound;
