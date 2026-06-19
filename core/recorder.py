import multiprocessing
import cv2
import time
import os
import queue
import sys

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class RecorderWorker(multiprocessing.Process):
    """
    Dedicated Process for Video Recording to Disk.
    Separating I/O from AI and GUI prevents lagging.
    """
    def __init__(self, recording_queue, task_queues, recording_dir, chunk_duration=30):
        super().__init__()
        self.recording_queue = recording_queue
        self.task_queues = task_queues # List of 6 multiprocessing.Queues
        self.recording_dir = recording_dir
        self.chunk_duration = chunk_duration
        self.stop_event = multiprocessing.Event()
        self.daemon = True
        
    def stop(self):
        self.stop_event.set()

    def run(self):
        writers = {} # {cam_index: VideoWriter}
        current_paths = {} # {cam_index: filepath}
        try:
            print(f"[Recorder] Started. Saving to {self.recording_dir}")
            os.makedirs(self.recording_dir, exist_ok=True)
            
            start_times = {} # {cam_index: start_time}
            last_receive_time = time.time()
            
            while not self.stop_event.is_set():
                try:
                    # Data: (cam_index, frame_array, timestamp) or "FLUSH" command
                    data = self.recording_queue.get(timeout=1.0)
                except queue.Empty:
                    # AUTO-FLUSH: If idle for 10s and have open writers, flush them
                    if writers and (time.time() - last_receive_time > 10.0):
                        print("[Recorder] Idle Timeout. Flushing open files...")
                        self._flush_writers(writers, current_paths)
                        start_times.clear()
                    continue
                
                # Process Data or Command
                if data == "FLUSH":
                    print("[Recorder] Received FLUSH command.")
                    self._flush_writers(writers, current_paths)
                    start_times.clear()
                    continue

                if not isinstance(data, (list, tuple)) or len(data) < 3:
                    continue

                cam_index, frame, timestamp = data
                last_receive_time = time.time()
                
                # Check rotation (30s chunks)
                if cam_index not in writers or (time.time() - start_times.get(cam_index, 0) > self.chunk_duration):
                    self._start_new_chunk(cam_index, frame, writers, start_times, current_paths)
                    
                # Write Frame
                if writers.get(cam_index):
                    try:
                        writers[cam_index].write(frame)
                    except Exception as e:
                        print(f"[Recorder] Write Error: {e}")

        except KeyboardInterrupt:
            print("[Recorder] Interrupted via KeyboardInterrupt.")
        except Exception as e:
            print(f"[Recorder] FATAL CRASH: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Final Cleanup
            try:
                self._flush_writers(writers, current_paths)
            except: pass
            print("[Recorder] Stopped.")

    def _flush_writers(self, writers, current_paths):
        """Closes all open writers and hands off to AI"""
        for index in list(writers.keys()):
            w = writers.pop(index)
            if w:
                w.release()
                final_path = current_paths.pop(index, None)
                if final_path:
                    # Sanity Check: If file is 0 KB, don't even bother
                    if os.path.exists(final_path) and os.path.getsize(final_path) > 0:
                        print(f"[Recorder] Flushed: {final_path}")
                        if 0 <= index < len(self.task_queues):
                            try:
                                self.task_queues[index].put(final_path)
                            except: pass
                    else:
                        print(f"[Recorder] Warning: Flushed file is empty, skipping handoff: {final_path}")

    def _start_new_chunk(self, index, frame, writers, start_times, current_paths):
        # Close old
        if writers.get(index):
            writers[index].release()
            final_path = current_paths[index]
            if final_path:
                # Sanity Check: If file is 0 KB, don't even bother
                if os.path.exists(final_path) and os.path.getsize(final_path) > 0:
                    print(f"[Recorder] Saved & Finished: {final_path}")
                    # Handoff to AI Worker
                    if 0 <= index < len(self.task_queues):
                        try:
                            self.task_queues[index].put(final_path)
                            print(f"[Recorder] Handoff -> Camera {index} Queue")
                        except Exception as e:
                            print(f"[Recorder] Handoff Failed: {e}")
                else:
                    print(f"[Recorder] Warning: Saved file is empty, skipping handoff: {final_path}")
            
        # Init New
        h, w = frame.shape[:2]
        # Use precise timestamp with milliseconds
        ts = int(time.time() * 1000)
        filename = f"cam_{index}_{ts}.avi"
        final_path = os.path.join(self.recording_dir, filename)
        
        try:
            # XVID is much more reliable on Windows than MJPG
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            writers[index] = cv2.VideoWriter(final_path, fourcc, 30.0, (w, h))
            if not writers[index].isOpened():
                print(f"[Recorder] CRITICAL: VideoWriter failed to open for {final_path}")
            
            start_times[index] = time.time()
            current_paths[index] = final_path
        except Exception as e:
            print(f"[Recorder] Init Error: {e}")
