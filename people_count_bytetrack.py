from ultralytics import YOLO
import cv2
import time
import datetime

# =========================
# Konfigurasi
# =========================
model = YOLO("yolov8n.pt")

# Webcam: 0
# Kalau mau video file, ganti jadi: "video.mp4"
source = 0

cap = cv2.VideoCapture(source)

if not cap.isOpened():
    print("Webcam / video tidak bisa dibuka")
    exit()

# Counter
total_masuk = 0
total_keluar = 0
masuk_batch = 0
keluar_batch = 0

# Memory posisi object berdasarkan track_id
track_memory = {}
counted_ids = set()

# Logging tiap 10 detik
log_file = open("people_log_bytetrack_10s.txt", "a")
start_time = time.time()

print("Tekan 'q' untuk keluar")

# =========================
# Loop video
# =========================
while True:
    ret, frame = cap.read()

    if not ret:
        print("Frame tidak terbaca / video selesai")
        break

    height, width = frame.shape[:2]
    line_position = width // 2   # garis vertikal di tengah frame

    # Jalankan YOLO + ByteTrack
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )

    result = results[0]

    if result.boxes is not None and result.boxes.id is not None:
        boxes = result.boxes.xyxy
        track_ids = result.boxes.id
        classes = result.boxes.cls
        confidences = result.boxes.conf

        for box, track_id, cls, conf in zip(boxes, track_ids, classes, confidences):
            class_name = model.names[int(cls)]
            confidence = float(conf)

            # Hanya hitung object person
            if class_name != "person" or confidence < 0.4:
                continue

            x1, y1, x2, y2 = map(int, box)
            track_id = int(track_id)

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Gambar bounding box dan ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"ID {track_id} person {confidence:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            # Ambil posisi sebelumnya
            prev_cx = track_memory.get(track_id)

            if prev_cx is not None and track_id not in counted_ids:
                # Kiri ke kanan = masuk
                if prev_cx < line_position and cx >= line_position:
                    total_masuk += 1
                    masuk_batch += 1
                    counted_ids.add(track_id)

                # Kanan ke kiri = keluar
                elif prev_cx > line_position and cx <= line_position:
                    total_keluar += 1
                    keluar_batch += 1
                    counted_ids.add(track_id)

            # Update posisi terakhir
            track_memory[track_id] = cx

    # Gambar garis vertikal tengah
    cv2.line(
        frame,
        (line_position, 0),
        (line_position, height),
        (0, 0, 255),
        2
    )

    # Tampilkan counter
    cv2.putText(
        frame,
        f"MASUK kiri->kanan: {total_masuk}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"KELUAR kanan->kiri: {total_keluar}",
        (10, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    cv2.imshow("People Counter - YOLOv8 + ByteTrack", frame)

    # Logging tiap 10 detik
    now = time.time()
    if now - start_time >= 10:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = (
            f"{timestamp} - "
            f"MASUK kiri->kanan: {masuk_batch}, "
            f"KELUAR kanan->kiri: {keluar_batch}\n"
        )

        log_file.write(log_line)
        log_file.flush()
        print(log_line.strip())

        masuk_batch = 0
        keluar_batch = 0
        start_time = now

    # Tekan q utk keluar
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# =========================
# Cleanup
# =========================
cap.release()
log_file.close()
cv2.destroyAllWindows()

print("Program selesai")
