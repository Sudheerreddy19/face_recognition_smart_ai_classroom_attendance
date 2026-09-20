"""
esp32_stream_test.py – Phase 11: Test ESP32-CAM MJPEG stream.

PURPOSE
-------
Verify that the ESP32-CAM is reachable and streaming correctly BEFORE
switching the face service to use it.  Run this AFTER the laptop
pipeline (Phases 1-10) is fully confirmed working.

PREREQUISITES
-------------
1. Flash the ESP32-CAM with the standard AI Thinker CameraWebServer sketch.
2. Update WIFI_SSID and WIFI_PASSWORD in the Arduino sketch.
3. Note the IP address printed on the Serial Monitor.
4. Set in your .env:
       CAMERA_SOURCE=esp32
       ESP32_CAMERA_URL=http://<IP_ADDRESS>
5. Run this script to verify streaming before starting the face service.

WHAT THIS TEST DOES
-------------------
- Opens the MJPEG stream URL using cv2.VideoCapture(url).
- Reads frames for 10 seconds.
- Reports actual FPS and resolution.
- Press Q to exit early.

WHAT THIS TEST DOES NOT DO
--------------------------
- Does NOT perform face detection (use face_detection_test.py after this).
- Does NOT save frames.
- Does NOT write to any database.
- Does NOT mark attendance.

EXPECTED STREAM URL FORMATS
----------------------------
AI Thinker CameraWebServer firmware:
    MJPEG stream : http://<IP>/stream
    Single JPEG  : http://<IP>/capture

Run command (from project root, with .venv311 active):
    .venv311\\Scripts\\python esp32_stream_test.py --url http://192.168.1.100
"""

import argparse
import sys
import time

import cv2

# ── Argument parsing ──────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="ESP32-CAM stream test")
parser.add_argument(
    "--url",
    default="http://192.168.1.100",
    help="ESP32-CAM base URL (default: http://192.168.1.100)",
)
parser.add_argument(
    "--path",
    default="/stream",
    help="Stream path (default: /stream)",
)
args = parser.parse_args()

stream_url = f"{args.url.rstrip('/')}{args.path}"

print()
print("=" * 54)
print("  PHASE 11 — ESP32-CAM STREAM TEST")
print("=" * 54)
print(f"[INFO] OpenCV  : {cv2.__version__}")
print(f"[INFO] Stream  : {stream_url}")
print()
print("[INFO] Opening stream (may take 3-5 seconds) ...")

# ── Open stream ───────────────────────────────────────────────────────────────

cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print(f"[ERROR] Cannot open stream: {stream_url}")
    print()
    print("Troubleshooting:")
    print("  1. Check ESP32 is powered and connected to Wi-Fi.")
    print("  2. Verify the IP address (check Serial Monitor).")
    print("  3. Try opening the URL in a browser first.")
    print("  4. Ensure laptop and ESP32 are on the same Wi-Fi network.")
    sys.exit(1)

print("[INFO] Stream opened successfully")
print("[INFO] Press Q to exit")
print()

# ── Read frames ───────────────────────────────────────────────────────────────

frame_count = 0
start_time  = time.time()
fps         = 0.0
fps_timer   = start_time

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("[WARN] Failed to read frame — retrying ...")
        time.sleep(0.1)
        continue

    frame_count += 1

    # FPS
    now = time.time()
    elapsed = now - fps_timer
    if elapsed >= 1.0:
        fps = frame_count / (now - start_time)
        fps_timer = now

    h, w = frame.shape[:2]

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Res: {w}x{h}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, stream_url, (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    cv2.imshow("Phase 11 — ESP32-CAM Stream (Q to quit)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:
        break

# ── Summary ───────────────────────────────────────────────────────────────────

total_time = time.time() - start_time
cap.release()
cv2.destroyAllWindows()

print(f"[INFO] Frames received : {frame_count}")
print(f"[INFO] Duration        : {total_time:.1f}s")
print(f"[INFO] Average FPS     : {frame_count/total_time:.1f}")
print(f"[INFO] Resolution      : {w}x{h}")
print()
if frame_count > 0 and (frame_count / total_time) >= 5:
    print("[OK]  Stream is working correctly.")
    print("      Next step: set CAMERA_SOURCE=esp32 in .env and run")
    print("      face_detection_test.py to verify face detection on the stream.")
else:
    print("[WARN] Low frame rate — check Wi-Fi signal and ESP32 quality setting.")
