from ultralytics import YOLO
import cv2

# Load model hasil training
model = YOLO("weights4.pt")

# Buka webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Webcam tidak terbaca")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Gagal membaca frame")
        break

    # Deteksi objek
    results = model(frame, conf=0.5)

    # Gambar bounding box ke frame
    annotated_frame = results[0].plot()

    # Tampilkan hasil
    cv2.imshow("Deteksi Helm - YOLO", annotated_frame)

    # Tekan q untuk keluar
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
