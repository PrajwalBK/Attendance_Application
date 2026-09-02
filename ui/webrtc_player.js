/**
 * WebRTC Stream Client for go2rtc & Vision Attendance System
 * Connects to the local FastAPI WebRTC endpoint (/api/local/webrtc/{cameraId})
 * or directly to go2rtc (/api/ws or /api/whep).
 */

export class WebRTCStreamClient {
    constructor(cameraId, videoElement, options = {}) {
        this.cameraId = cameraId;
        this.videoElement = videoElement;
        this.apiBaseUrl = options.apiBaseUrl || "http://localhost:8000";
        this.pc = null;
        this.onStatusChange = options.onStatusChange || (() => {});
        this.autoReconnect = options.autoReconnect !== false;
        this.reconnectTimeout = null;
    }

    async connect() {
        this.disconnect();
        this.onStatusChange("connecting");

        try {
            // 1. Initialize WebRTC PeerConnection for ultra-low latency local streaming
            this.pc = new RTCPeerConnection({
                iceServers: []
            });

            // 2. Setup track handler to attach incoming video stream to the HTML5 video element
            this.pc.ontrack = (event) => {
                if (this.videoElement && event.streams && event.streams[0]) {
                    this.videoElement.srcObject = event.streams[0];
                    this.onStatusChange("playing");
                }
            };

            this.pc.onconnectionstatechange = () => {
                const state = this.pc.connectionState;
                this.onStatusChange(state);
                if (state === "failed" || state === "disconnected") {
                    if (this.autoReconnect) {
                        this._scheduleReconnect();
                    }
                }
            };

            // 3. Add receive-only transceiver for Video
            this.pc.addTransceiver("video", { direction: "recvonly" });

            // 4. Create WebRTC Offer
            const offer = await this.pc.createOffer();
            await this.pc.setLocalDescription(offer);

            // 5. Send SDP offer to FastAPI WebRTC endpoint
            const response = await fetch(`${this.apiBaseUrl}/api/local/webrtc/${this.cameraId}`, {
                method: "POST",
                headers: { "Content-Type": "application/sdp" },
                body: this.pc.localDescription.sdp
            });

            if (!response.ok) {
                const errText = await response.text();
                throw new Error(`WebRTC negotiation error (${response.status}): ${errText}`);
            }

            const answerSdp = await response.text();

            // 6. Set Remote Description (SDP Answer)
            await this.pc.setRemoteDescription(new RTCSessionDescription({
                type: "answer",
                sdp: answerSdp
            }));

        } catch (error) {
            console.error(`[WebRTC Client Cam ${this.cameraId}] Error:`, error);
            this.onStatusChange("error");
            if (this.autoReconnect) {
                this._scheduleReconnect();
            }
        }
    }

    _waitForIceGathering(pc) {
        if (pc.iceGatheringState === "complete") {
            return Promise.resolve();
        }
        return new Promise((resolve) => {
            const checkState = () => {
                if (pc.iceGatheringState === "complete") {
                    pc.removeEventListener("icegatheringstatechange", checkState);
                    resolve();
                }
            };
            pc.addEventListener("icegatheringstatechange", checkState);
            // Fallback timeout after 1.5s
            setTimeout(resolve, 1500);
        });
    }

    _scheduleReconnect() {
        if (this.reconnectTimeout) clearTimeout(this.reconnectTimeout);
        this.reconnectTimeout = setTimeout(() => {
            console.log(`[WebRTC Client Cam ${this.cameraId}] Attempting reconnect...`);
            this.connect();
        }, 3000);
    }

    disconnect() {
        if (this.reconnectTimeout) {
            clearTimeout(this.reconnectTimeout);
            this.reconnectTimeout = null;
        }
        if (this.pc) {
            this.pc.ontrack = null;
            this.pc.onconnectionstatechange = null;
            this.pc.close();
            this.pc = null;
        }
        if (this.videoElement) {
            this.videoElement.srcObject = null;
        }
        this.onStatusChange("stopped");
    }
}
