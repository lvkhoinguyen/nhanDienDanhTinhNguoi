import pickle
import os


class FaceModel:
    def __init__(self, model_path='face_recognition_model.pkl'):
        self.model_path = model_path

    def save_model(self, encodings, names):
        """Lưu model vào file"""
        model_data = {
            'encodings': encodings,
            'names': names
        }
        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model đã được lưu vào {self.model_path}")

    def load_model(self):
        """Tải model từ file"""
        if not os.path.exists(self.model_path):
            return None, None

        with open(self.model_path, 'rb') as f:
            model_data = pickle.load(f)
        return model_data['encodings'], model_data['names']

    def update_model(self, new_encoding, new_name):
        """Cập nhật model với khuôn mặt mới"""
        encodings, names = self.load_model()
        if encodings is None:
            encodings = []
            names = []

        encodings.append(new_encoding)
        names.append(new_name)
        self.save_model(encodings, names)