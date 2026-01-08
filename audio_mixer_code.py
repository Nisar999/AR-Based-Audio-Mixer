import cv2
import mediapipe as mp
import numpy as np
from pydub import AudioSegment
import threading
import time
import queue
import os
import subprocess
import tempfile

# Ask user to input audio path
audio_path = input("Enter path to your audio file (.mp3 or .wav): ").strip().strip('"')

# Validate file
if not os.path.isfile(audio_path):
    print("[ERROR] File not found!")
    exit(1)

# Load audio
try:
    audio = AudioSegment.from_file(audio_path)
except Exception as e:
    print(f"[ERROR] Could not load audio: {e}")
    exit(1)

# Initialize playback
play_proc = None
audio_queue = queue.Queue()

# Playback using ffplay subprocess
def play_audio_segment(segment):
    global play_proc
    if play_proc and play_proc.poll() is None:
        play_proc.terminate()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        segment.export(f.name, format="wav")
        play_proc = subprocess.Popen(
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", f.name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

# Background thread to play audio segments
def audio_worker():
    while True:
        seg = audio_queue.get()
        if seg is None:
            break
        play_audio_segment(seg)
        audio_queue.task_done()

audio_thread = threading.Thread(target=audio_worker, daemon=True)
audio_thread.start()

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

# Utility functions
def get_coords(landmarks, image_shape):
    h, w, _ = image_shape
    return np.array([[int(lm.x * w), int(lm.y * h)] for lm in landmarks])

def calc_distance(p1, p2):
    return np.linalg.norm(p1 - p2)

# Webcam setup
cap = cv2.VideoCapture(0)
last_time = time.time()
segment_duration_ms = 2000
start_ms = 0

print("[INFO] Waiting for MediaPipe to initialize...")
for _ in range(30):
    cap.read()
    time.sleep(0.03)

print("[INFO] Starting real-time audio mixing...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
        coords = [get_coords(h.landmark, frame.shape) for h in results.multi_hand_landmarks]
        left_hand, right_hand = sorted(coords, key=lambda x: x[0][0])

        left_y = left_hand[8][1]
        speed = np.interp(left_y, [0, frame.shape[0]], [1.5, 0.5])

        right_y = right_hand[8][1]
        freq_shift = np.interp(right_y, [0, frame.shape[0]], [800, 100])

        volume = np.clip(calc_distance(left_hand[8], right_hand[8]) / 300, 0.0, 1.0)

        now = time.time()
        if now - last_time >= 2:
            seg = audio[start_ms:start_ms + segment_duration_ms]
            seg = seg + (volume * 10 - 5)
            seg = seg._spawn(seg.raw_data, overrides={
                "frame_rate": int(seg.frame_rate * speed)
            }).set_frame_rate(44100)

            print(f"[INFO] Playing new segment from {start_ms} ms at speed {speed:.2f}x and volume {volume:.2f}")
            audio_queue.put(seg)

            start_ms = (start_ms + segment_duration_ms) % len(audio)
            last_time = now

        cv2.putText(frame, f"Speed: {speed:.2f}x", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(frame, f"Volume: {volume:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(frame, f"Freq: {int(freq_shift)} Hz", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        print(f"[INFO] Queue size: {audio_queue.qsize()}")

    cv2.imshow("Air DJ: Real Audio Track", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

# Cleanup
audio_queue.put(None)
audio_thread.join()
if play_proc and play_proc.poll() is None:
    play_proc.terminate()
cap.release()
cv2.destroyAllWindows()
print("[INFO] Exiting...")
