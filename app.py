import cv2
import os
from models.face_dataset import FaceDataset
from models.face_model import FaceModel
from models.face_recognition import FaceRecognizer
from utils.image_utils import capture_face_image, save_temp_image
import face_recognition


def main():
    # Khởi tạo các thành phần
    face_dataset = FaceDataset()
    face_model = FaceModel()

    # Kiểm tra model
    if os.path.exists(face_model.model_path):
        print("Phát hiện model đã tồn tại, bạn muốn:")
        print("1. Sử dụng model hiện có")
        print("2. Tạo model mới từ dataset")
        choice = input("Nhập lựa chọn (1/2): ")

        if choice == '1':
            known_face_encodings, known_face_names = face_model.load_model()
        else:
            known_face_encodings, known_face_names = face_dataset.create_face_encodings()
            face_model.save_model(known_face_encodings, known_face_names)
    else:
        print("Không tìm thấy model, tạo model mới từ dataset...")
        known_face_encodings, known_face_names = face_dataset.create_face_encodings()
        face_model.save_model(known_face_encodings, known_face_names)

    print(f"Đã load {len(known_face_names)} khuôn mặt")

    # Khởi tạo recognizer
    recognizer = FaceRecognizer(known_face_encodings, known_face_names)

    # Mở webcam
    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    registering = False
    new_face_name = ""

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Không nhận được frame từ webcam")
            break

        if not registering:
            # Chế độ nhận diện
            face_locations, face_names = recognizer.recognize_faces(frame)
            frame = recognizer.draw_face_annotations(frame, face_locations, face_names)

            # Hiển thị hướng dẫn
            cv2.putText(frame, "Press 'r' to register new face", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            # Chế độ đăng ký khuôn mặt mới
            cv2.putText(frame, f"Registering: {new_face_name}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, "Press 'c' to capture", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, "Press 'q' to cancel", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Tìm và đánh dấu khuôn mặt
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)

            if face_locations:
                (top, right, bottom, left) = face_locations[0]
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        cv2.imshow('Face Recognition', frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            if registering:
                registering = False  # Thoát chế độ đăng ký
            else:
                break  # Thoát chương trình

        elif key == ord('r') and not registering:
            registering = True
            new_face_name = input("Nhập tên người dùng mới: ")

        elif key == ord('c') and registering:
            # Chụp ảnh và đăng ký khuôn mặt mới
            face_image = capture_face_image(video_capture)
            if face_image is not None:
                rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

                # Thêm vào dataset và cập nhật model
                new_encoding, image_path = face_dataset.add_new_face(new_face_name, rgb_face)
                if new_encoding is not None:
                    face_model.update_model(new_encoding, new_face_name)
                    print(f"Đã đăng ký khuôn mặt mới: {new_face_name}")

                # Cập nhật recognizer với dữ liệu mới
                known_face_encodings, known_face_names = face_model.load_model()
                recognizer = FaceRecognizer(known_face_encodings, known_face_names)

                registering = False

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()