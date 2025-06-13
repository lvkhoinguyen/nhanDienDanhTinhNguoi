import cv2
import face_recognition
import numpy as np


class FaceRecognizer:
    def __init__(self, known_face_encodings, known_face_names):
        self.known_face_encodings = known_face_encodings
        self.known_face_names = known_face_names

    def recognize_faces(self, frame):
        """Nhận diện khuôn mặt trong frame"""
        # Chuyển đổi màu
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Tìm khuôn mặt
        face_locations = face_recognition.face_locations(rgb_frame)
        face_names = []

        if face_locations:
            try:
                # Lấy đặc trưng khuôn mặt
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                for face_encoding in face_encodings:
                    # So sánh với khuôn mặt đã biết
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = "Unknown"

                    if True in matches:
                        face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)

                        if matches[best_match_index] and face_distances[best_match_index] < 0.6:
                            name = self.known_face_names[best_match_index]

                    face_names.append(name)

            except Exception as e:
                print(f"Lỗi khi xử lý khuôn mặt: {str(e)}")

        return face_locations, face_names

    def draw_face_annotations(self, frame, face_locations, face_names):
        """Vẽ khung và nhãn lên frame"""
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Vẽ khung và nhãn
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)

        return frame