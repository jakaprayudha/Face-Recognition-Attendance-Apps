import cv2
import csv
import os
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
import os
from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, session
from db.connect import get_connection
from hashlib import sha256

app = Flask(__name__)

load_dotenv()
app.secret_key = os.getenv("SECRET_KEY")

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

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

SAVE_DIR = "captures"
os.makedirs(SAVE_DIR, exist_ok=True)

# ======== Variabel global status ========
last_status = {"recognized": False, "distance": None, "face_detected": False}

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
        face_detected = False

        for (x, y, w, h) in faces:
            face_detected = True
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

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        last_status.update({
            "recognized": recognized,
            "distance": distance_val,
            "face_detected": face_detected
        })

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = get_connection()
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT * FROM users WHERE username = %s", (username,))
        user = cur.fetchone()
        cur.close()
        conn.close()

        if user and user['password_hash'] == sha256(password.encode()).hexdigest():
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['role'] = user['role']
            session['full_name'] = user['full_name']
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error="Username atau password salah")

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', username=session['full_name'])

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    return jsonify(last_status)

# ======== API untuk Simpan Absensi ========
@app.route('/checkin', methods=['POST'])
def checkin():
    data = request.get_json()
    nama = data.get("nama", "Unknown")
    latitude = data.get("latitude")
    longitude = data.get("longitude")
    recognized = last_status.get("recognized")
    distance = last_status.get("distance")

    if not last_status.get("face_detected"):
        return jsonify({"status": "error", "message": "Tidak ada wajah terdeteksi!"})

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO absensi (nama, waktu_checkin, latitude, longitude, distance, recognized) VALUES (%s,%s,%s,%s,%s,%s)",
        (nama, datetime.now(), latitude, longitude, distance, recognized)
    )
    conn.commit()
    cur.close()
    conn.close()

    return jsonify({"status": "success", "message": "Absensi berhasil disimpan!"})

@app.route('/map')
def map_view():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('map.html')

@app.route('/data_absensi')
def data_absensi():
    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT * FROM absensi ORDER BY waktu_checkin DESC")
    data = cur.fetchall()
    cur.close()
    conn.close()
    return jsonify(data)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)