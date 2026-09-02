import React, { useEffect, useRef, useState } from 'react';
import { WebRTCStreamClient } from './webrtc_player';

/**
 * Reusable WebRTC Camera Player Component
 * Connects to go2rtc via FastAPI endpoint with automatic fallback to MJPEG.
 *
 * @param {Object} props
 * @param {number} props.cameraId - Camera index (0 or 1)
 * @param {string} [props.apiBaseUrl] - Base API URL (defaults to http://localhost:8000)
 * @param {boolean} [props.isActive] - Whether camera streaming is currently active
 * @param {string} [props.className] - CSS class name
 * @param {boolean} [props.fallbackMjpeg] - Fallback to MJPEG if WebRTC fails
 */
export const WebRTCPlayer = ({
    cameraId = 0,
    apiBaseUrl = "http://localhost:8000",
    isActive = true,
    className = "",
    fallbackMjpeg = true,
    style = {}
}) => {
    const videoRef = useRef(null);
    const clientRef = useRef(null);
    const [status, setStatus] = useState("disconnected");
    const [useFallback, setUseFallback] = useState(false);

    useEffect(() => {
        if (!isActive) {
            if (clientRef.current) {
                clientRef.current.disconnect();
                clientRef.current = null;
            }
            setStatus("stopped");
            return;
        }

        const client = new WebRTCStreamClient(cameraId, videoRef.current, {
            apiBaseUrl,
            autoReconnect: true,
            onStatusChange: (newStatus) => {
                setStatus(newStatus);
                if (newStatus === "error" && fallbackMjpeg) {
                    console.warn(`[WebRTCPlayer ${cameraId}] WebRTC failed, falling back to MJPEG.`);
                    setUseFallback(true);
                } else if (newStatus === "playing") {
                    setUseFallback(false);
                }
            }
        });

        clientRef.current = client;
        client.connect();

        return () => {
            client.disconnect();
            clientRef.current = null;
        };
    }, [cameraId, apiBaseUrl, isActive, fallbackMjpeg]);

    if (!isActive) {
        return (
            <div
                className={`camera-inactive-placeholder ${className}`}
                style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    backgroundColor: '#111827',
                    color: '#9CA3AF',
                    minHeight: '240px',
                    borderRadius: '8px',
                    ...style
                }}
            >
                <span>Camera {cameraId + 1} Offline</span>
            </div>
        );
    }

    if (useFallback) {
        return (
            <div className={`camera-mjpeg-container ${className}`} style={{ position: 'relative', ...style }}>
                <img
                    src={`${apiBaseUrl}/api/local/video_feed/${cameraId}`}
                    alt={`Camera ${cameraId + 1} Stream (MJPEG)`}
                    style={{ width: '100%', height: '100%', objectFit: 'cover', borderRadius: '8px' }}
                />
                <span
                    style={{
                        position: 'absolute',
                        bottom: '8px',
                        left: '8px',
                        backgroundColor: 'rgba(0,0,0,0.6)',
                        color: '#F59E0B',
                        fontSize: '11px',
                        padding: '2px 6px',
                        borderRadius: '4px'
                    }}
                >
                    MJPEG Fallback
                </span>
            </div>
        );
    }

    return (
        <div className={`camera-webrtc-container ${className}`} style={{ position: 'relative', ...style }}>
            <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                style={{
                    width: '100%',
                    height: '100%',
                    objectFit: 'cover',
                    borderRadius: '8px',
                    backgroundColor: '#000'
                }}
            />
            <span
                style={{
                    position: 'absolute',
                    bottom: '8px',
                    left: '8px',
                    backgroundColor: 'rgba(0,0,0,0.6)',
                    color: status === 'playing' ? '#10B981' : '#F59E0B',
                    fontSize: '11px',
                    padding: '2px 6px',
                    borderRadius: '4px'
                }}
            >
                {status === 'playing' ? 'WebRTC Live (<50ms)' : `Connecting... (${status})`}
            </span>
        </div>
    );
};

export default WebRTCPlayer;
