import cv2
import csv
import os
import time

# Load classifier haarcascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Buka webcam
cap = cv2.VideoCapture(1)  # ganti 1 kalau pakai kamera eksternal

# Nama file CSV
csv_file = "deteksi_wajah.csv"

# Buat file CSV & header kalau belum ada
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["frame_id", "face_x", "face_y", "face_w", "face_h",
                         "eye_x", "eye_y", "eye_w", "eye_h",
                         "mouth_x", "mouth_y", "mouth_w", "mouth_h"])

# Countdown sebelum mulai
for i in range(3, 0, -1):
    print(f"Mulai merekam dalam {i}...")
    ret, frame = cap.read()
    if ret:
        cv2.putText(frame, f"Mulai dalam {i}", (50, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        cv2.imshow('Countdown', frame)
        cv2.waitKey(1000)  # tunggu 1 detik
cv2.destroyWindow('Countdown')

print("Mulai rekam data ke CSV selama 5 detik...")

frame_id = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Variabel default kosong
    face_coords = [None, None, None, None]
    eye_coords = [None, None, None, None]
    mouth_coords = [None, None, None, None]

    # Deteksi wajah
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        face_coords = [x, y, w, h]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Deteksi mata
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        if len(eyes) > 0:
            (ex, ey, ew, eh) = eyes[0]  # ambil mata pertama
            eye_coords = [ex, ey, ew, eh]
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        # Deteksi mulut (pakai smile cascade)
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 20)
        for (sx, sy, sw, sh) in smiles:
            mouth_coords = [sx, sy, sw, sh]
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 255, 255), 2)
            break  # ambil satu mulut saja

        break  # ambil satu wajah saja

    # Simpan hasil ke CSV (hanya dalam 5 detik pertama)
    if time.time() - start_time <= 5:
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([frame_id] + face_coords + eye_coords + mouth_coords)
    else:
        print("Rekam selesai. Data tersimpan di", csv_file)
        break

    # Tampilkan hasil
    cv2.imshow('Face, Eyes, Smile Detection', frame)

    # Keluar manual dengan 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()