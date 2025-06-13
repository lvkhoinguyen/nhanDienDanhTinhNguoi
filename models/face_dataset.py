import os
import numpy as np
from PIL import Image
import face_recognition


class FaceDataset:
    def __init__(self, dataset_dir='dataset'):
        self.dataset_dir = dataset_dir
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

    def load_image_file(self, file, mode='RGB'):
        """Load image từ file và chuyển đổi sang mode mong muốn"""
        im = Image.open(file)
        if im.mode == 'P':
            im = im.convert('RGBA')
        if mode:
            im = im.convert(mode)
        return np.array(im)

    def create_face_encodings(self):
        """Tạo encodings từ dataset"""
        known_face_encodings = []
        known_face_names = []

        for person_name in os.listdir(self.dataset_dir):
            person_dir = os.path.join(self.dataset_dir, person_name)

            if os.path.isdir(person_dir):
                for file in os.listdir(person_dir):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_path = os.path.join(person_dir, file)
                        try:
                            image = self.load_image_file(image_path)
                            face_encodings = face_recognition.face_encodings(image)

                            for encoding in face_encodings:
                                known_face_encodings.append(encoding)
                                known_face_names.append(person_name)
                        except Exception as e:
                            print(f"Không thể xử lý ảnh {image_path}: {str(e)}")
                            continue

        return known_face_encodings, known_face_names

    def add_new_face(self, name, image_array):
        """Thêm khuôn mặt mới vào dataset"""
        person_dir = os.path.join(self.dataset_dir, name)
        if not os.path.exists(person_dir):
            os.makedirs(person_dir)

        # Đếm số ảnh hiện có để tạo tên file mới
        count = len(os.listdir(person_dir)) + 1
        image_path = os.path.join(person_dir, f"{name}_{count}.jpg")

        # Lưu ảnh
        image = Image.fromarray(image_array)
        image.save(image_path)

        # Tạo encoding cho ảnh mới
        face_encodings = face_recognition.face_encodings(image_array)
        if face_encodings:
            return face_encodings[0], image_path
        return None, image_path