import cv2
import csv
import os
import numpy as np
from datetime import datetime
from flask import Flask, render_template, Response, jsonify

app = Flask(__name__)

# ======== Load dataset CSV ==========
dataset_file = "deteksi_wajah.csv"
dataset = []

with open(dataset_file, mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        if row["face_x"] and row["face_y"] and row["face_w"] and row["face_h"]:
            try:
                face = [int(row["face_x"]), int(row["face_y"]),
                        int(row["face_w"]), int(row["face_h"])]
                dataset.append(face)
            except ValueError:
                continue

if not dataset:
    print("Dataset kosong! Jalankan script rekam dulu.")
    exit()

mean_face = np.mean(dataset, axis=0)
print("Template wajah (mean bbox):", mean_face)

# Load Haarcascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ======== Variabel global status ========
last_status = {"recognized": False, "distance": None}

# Buat folder simpanan kalau belum ada
SAVE_DIR = "captures"
os.makedirs(SAVE_DIR, exist_ok=True)

# ======== Video generator ==========
def generate_frames():
    global last_status
    cap = cv2.VideoCapture(1)

    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        recognized = False
        distance_val = None

        for (x, y, w, h) in faces:
            detected_face = np.array([x, y, w, h])
            dist = np.linalg.norm(detected_face - mean_face)
            distance_val = float(dist)

            if dist < 300:
                recognized = True
                color = (0, 255, 0)
                label = f"Wajah Dikenali ({dist:.1f})"
            else:
                recognized = False
                color = (0, 0, 255)
                label = f"Tidak Dikenali ({dist:.1f})"

                # 🚨 Capture wajah tidak dikenali
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(SAVE_DIR, f"unknown_{timestamp}.jpg")

                # Tambahkan watermark timestamp
                watermark = f"Unknown {timestamp}"
                cv2.putText(frame, watermark, (10, frame.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.imwrite(filename, frame)
                print(f"[INFO] Disimpan: {filename}")

            # Gambar kotak + label
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # update status global
        last_status["recognized"] = recognized
        last_status["distance"] = distance_val

        # Encode ke JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ======== Routes Flask ==========
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ✅ Route baru untuk status JSON
@app.route('/status')
def status():
    return jsonify(last_status)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)