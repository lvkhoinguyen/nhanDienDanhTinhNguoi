import face_recognition
import cv2
import os
import pickle
import numpy as np
from config import Config


class FaceRecognitionModel:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.model_path = os.path.join(Config.MODEL_DIR, 'face_recognizer.pkl')
        self.load_model()

    def load_dataset(self):
        """Load and encode training dataset"""
        self.known_face_encodings = []
        self.known_face_names = []

        if not os.path.exists(Config.TRAIN_DIR):
            os.makedirs(Config.TRAIN_DIR, exist_ok=True)
            return 0

        for person_name in os.listdir(Config.TRAIN_DIR):
            person_dir = os.path.join(Config.TRAIN_DIR, person_name)
            if os.path.isdir(person_dir):
                for img_file in os.listdir(person_dir):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(person_dir, img_file)
                        image = face_recognition.load_image_file(img_path)

                        # Convert to RGB for face_recognition
                        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                        # Get face encodings
                        face_encodings = face_recognition.face_encodings(
                            rgb_image,
                            num_jitters=Config.NUM_JITTERS
                        )

                        if face_encodings:
                            self.known_face_encodings.append(face_encodings[0])
                            self.known_face_names.append(person_name)

        self.save_model()
        return len(self.known_face_names)

    def save_model(self):
        """Save trained model to file"""
        with open(self.model_path, 'wb') as f:
            pickle.dump({
                'encodings': self.known_face_encodings,
                'names': self.known_face_names
            }, f)

    def load_model(self):
        """Load pre-trained model"""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data['encodings']
                    self.known_face_names = data['names']
                return True
            except:
                return False
        return False

    def recognize_faces(self, frame):
        """Recognize faces in a frame and return with confidence"""
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Find all faces
        face_locations = face_recognition.face_locations(
            rgb_frame,
            model=Config.FACE_DETECTION_MODEL
        )
        face_encodings = face_recognition.face_encodings(
            rgb_frame,
            face_locations,
            num_jitters=Config.NUM_JITTERS
        )

        # Recognize each face
        results = []
        for face_encoding, face_location in zip(face_encodings, face_locations):
            # Calculate face distances
            face_distances = face_recognition.face_distance(
                self.known_face_encodings,
                face_encoding
            )

            name = "Unknown"
            confidence = 0.0

            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                min_distance = face_distances[best_match_index]

                # Convert distance to percentage confidence
                confidence = max(0.0, min(100.0, (1.0 - min_distance) * 100))

                # Only accept if confidence >= minimum threshold
                if confidence >= Config.MIN_CONFIDENCE and min_distance < Config.TOLERANCE:
                    name = self.known_face_names[best_match_index]

            # Convert location to (top, right, bottom, left)
            top, right, bottom, left = face_location

            results.append({
                'name': name,
                'location': (top, right, bottom, left),
                'confidence': confidence
            })

        return results

    def detect_faces(self, frame):
        """Detect faces without recognition using face_recognition library"""
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Find face locations
        face_locations = face_recognition.face_locations(
            rgb_frame,
            model=Config.FACE_DETECTION_MODEL
        )

        return face_locations