from ultralytics import YOLO
import cv2

# Load model YOLOv8
model = YOLO("yolov8n.pt")

# Buka webcam laptop
cap = cv2.VideoCapture(0)

# Cek apakah webcam berhasil dibuka
if not cap.isOpened():
    print("Webcam tidak ditemukan!")
    exit()

print("Tekan tombol 'q' untuk keluar")

while True:
    # Ambil frame dari webcam
    ret, frame = cap.read()

    if not ret:
        print("Gagal membaca frame")
        break

    # Jalankan YOLO detection
    results = model(frame)

    # Tambahkan bounding box ke frame
    annotated_frame = results[0].plot()

    # Tampilkan hasil
    cv2.imshow("YOLO Webcam Detection", annotated_frame)

    # Tekan q untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam dan tutup window
cap.release()
cv2.destroyAllWindows()

print("Program selesai")
