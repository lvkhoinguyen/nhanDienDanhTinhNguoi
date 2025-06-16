import cv2
import os
import time
from tkinter import filedialog
from config import Config
from face_model import FaceRecognitionModel


class FaceController:
    def __init__(self):
        self.video_capture = None
        self.current_frame = None
        self.recognizing = False
        self.model = FaceRecognitionModel()

    def get_frame(self):
        """Get current frame from camera"""
        if self.video_capture and self.video_capture.isOpened():
            ret, frame = self.video_capture.read()
            if ret:
                self.current_frame = frame
                return frame
        return None

    def start_camera(self, camera_id=Config.DEFAULT_CAMERA_ID):
        """Start video capture"""
        if self.video_capture is None or not self.video_capture.isOpened():
            self.video_capture = cv2.VideoCapture(camera_id)
            return self.video_capture.isOpened()
        return True

    def stop_camera(self):
        """Stop video capture"""
        if self.video_capture and self.video_capture.isOpened():
            self.video_capture.release()
            self.video_capture = None

    def recognize_faces(self, frame):
        """Recognize faces in a frame with confidence"""
        return self.model.recognize_faces(frame)

    def register_face(self, name):
        """Register a new face by capturing multiple samples"""
        person_dir = os.path.join(Config.TRAIN_DIR, name)
        os.makedirs(person_dir, exist_ok=True)

        count = 0
        while count < Config.FACE_SAMPLES:
            frame = self.get_frame()
            if frame is None:
                return False

            # Detect faces
            face_locations = self.model.detect_faces(frame)

            if face_locations:
                # Get the largest face
                face_location = max(face_locations, key=lambda loc: (loc[1] - loc[3]) * (loc[2] - loc[0]))
                top, right, bottom, left = face_location

                # Save face image
                face_img = frame[top:bottom, left:right]
                img_path = os.path.join(person_dir, f"{name}_{count + 1}.jpg")
                cv2.imwrite(img_path, face_img)

                count += 1
                time.sleep(Config.CAPTURE_INTERVAL)

        # Retrain model with new data
        self.model.load_dataset()
        return True

    def train_model(self):
        """Train the face recognition model"""
        return self.model.load_dataset()

    def process_image_file(self):
        """Process an image file selected by user"""
        file_path = filedialog.askopenfilename(
            title="Chọn ảnh để nhận diện",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        if not file_path:
            return None, None

        # Load image
        image = cv2.imread(file_path)
        if image is None:
            return None, None

        # Recognize faces with confidence
        results = self.model.recognize_faces(image)

        # Draw results on image
        for result in results:
            top, right, bottom, left = result['location']
            name = result['name']
            confidence = result['confidence']

            # Choose color based on recognition result
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

            # Draw rectangle around face
            cv2.rectangle(image, (left, top), (right, bottom), color, 2)

            # Draw background for name
            cv2.rectangle(image, (left, bottom - 35), (right, bottom), color, cv2.FILLED)

            # Create display text
            display_text = f"{name} ({confidence:.1f}%)"

            # Display name and confidence ON THE IMAGE
            cv2.putText(image, display_text, (left + 6, bottom - 6),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)

        return image, results