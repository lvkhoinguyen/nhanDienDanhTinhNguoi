import cv2
import face_recognition
import numpy as np
import os
import pickle
from PIL import Image

def load_image_file(file, mode='RGB'):
    """
    Load image từ file và chuyển đổi sang mode mong muốn
    """
    im = Image.open(file)
    if im.mode == 'P':
        im = im.convert('RGBA')
    if mode:
        im = im.convert(mode)
    return np.array(im)


# Hàm load ảnh từ dataset và tạo encodings
def create_face_encodings(dataset_dir):
    known_face_encodings = []
    known_face_names = []

    for person_name in os.listdir(dataset_dir):
        person_dir = os.path.join(dataset_dir, person_name)

        if os.path.isdir(person_dir):
            for file in os.listdir(person_dir):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(person_dir, file)
                    try:
                        # Sử dụng hàm load_image_file đã sửa
                        image = load_image_file(image_path)
                        face_encodings = face_recognition.face_encodings(image)

                        for encoding in face_encodings:
                            known_face_encodings.append(encoding)
                            known_face_names.append(person_name)
                    except Exception as e:
                        print(f"Không thể xử lý ảnh {image_path}: {str(e)}")
                        continue

    return known_face_encodings, known_face_names

# Hàm lưu model vào file
def save_model(encodings, names, model_path='face_recognition_model.pkl'):
    model_data = {
        'encodings': encodings,
        'names': names
    }
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"Model đã được lưu vào {model_path}")


# Hàm tải model từ file
def load_model(model_path='face_recognition_model.pkl'):
    if not os.path.exists(model_path):
        return None, None

    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data['encodings'], model_data['names']


# Kiểm tra xem model đã tồn tại chưa
model_path = 'face_recognition_model.pkl'
if os.path.exists(model_path):
    print("Phát hiện model đã tồn tại, bạn muốn:")
    print("1. Sử dụng model hiện có")
    print("2. Tạo model mới từ dataset")
    choice = input("Nhập lựa chọn (1/2): ")

    if choice == '1':
        known_face_encodings, known_face_names = load_model(model_path)
    else:
        known_face_encodings, known_face_names = create_face_encodings('dataset')
        save_model(known_face_encodings, known_face_names, model_path)
else:
    print("Không tìm thấy model, tạo model mới từ dataset...")
    known_face_encodings, known_face_names = create_face_encodings('dataset')
    save_model(known_face_encodings, known_face_names, model_path)

print(f"Đã load {len(known_face_names)} khuôn mặt")


# Hàm nhận diện khuôn mặt từ webcam
def recognize_faces():
    video_capture = cv2.VideoCapture(0)

    # Kiểm tra webcam
    if not video_capture.isOpened():    
        print("Không thể mở webcam!")
        return

    # Đặt kích thước frame
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Không nhận được frame từ webcam")
            break

        # Chuyển đổi màu
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Tìm khuôn mặt
        face_locations = face_recognition.face_locations(rgb_frame)

        if not face_locations:
            cv2.imshow('Face Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        try:
            # Lấy đặc trưng khuôn mặt
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # So sánh với khuôn mặt đã biết
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                if True in matches:
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)

                    if matches[best_match_index] and face_distances[best_match_index] < 0.6:
                        name = known_face_names[best_match_index]

                # Vẽ khung và nhãn
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)

        except Exception as e:
            print(f"Lỗi khi xử lý khuôn mặt: {str(e)}")
            continue

        cv2.imshow('Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


# Chạy chương trình nhận diện
recognize_faces()