import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import pickle
import time

class LiveFaceRecognitionWithDatabase:
    def __init__(self, database_path="../models/face_database_all_v2.pkl"):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # this is for face detection
        self.mtcnn = MTCNN(
            image_size=160, 
            margin=40,
            min_face_size=20,
            thresholds=[0.7, 0.8, 0.8], 
            factor=0.709, 
            post_process=True,
            device=self.device,
            keep_all=True
        )
        
        # this is for face recognition
        self.resnet = InceptionResnetV1(
            classify=False,
            pretrained='vggface2'
        ).to(self.device)
        self.resnet.eval()
        
        # for smoothing values
        self.smoothing_buffer = {}
        self.buffer_size = 5
        
        # load face database
        self.load_database(database_path)
        
    def load_database(self, database_path):
        """Load pre-computed face database"""
        try:
            with open(database_path, 'rb') as f:
                database = pickle.load(f)
                self.known_embeddings = database['embeddings']
                self.known_names = database['names']
                self.classes = database['classes']
            
            print(f"✅ Loaded database with {len(self.known_names)} face embeddings")
            print(f"📋 Known people: {self.classes}")
            
        except FileNotFoundError:
            print(f"❌ Database file not found: {database_path}")
            print("Please run create_database_from_cropped.py first!")
            self.known_embeddings = []
            self.known_names = []
            self.classes = []
    
    def get_embedding(self, face_tensor):
        """Extract embedding from face tensor"""
        if face_tensor is None:
            return None
            
        with torch.no_grad():
            face_tensor = face_tensor.to(self.device)
            if len(face_tensor.shape) == 3:
                face_tensor = face_tensor.unsqueeze(0)
            embedding = self.resnet(face_tensor)
            return embedding.cpu().numpy().flatten()
    
    def distance_to_probability(self, distance, threshold=0.85):
        """Convert distance to probability-like score (0-1)"""
        if distance >= threshold * 2:
            return 0.0  # Very poor match
        elif distance <= 0.1:
            return 1.0  # Perfect match
        else:
            # Convert distance to probability: closer to 0 = higher probability
            # Use exponential decay for more intuitive scaling
            prob = max(0.0, 1.0 - (distance / threshold))
            return min(1.0, prob)
    
    def smooth_values(self, face_id, distance, confidence):
        """Smooth distance and confidence values over time"""
        if face_id not in self.smoothing_buffer:
            self.smoothing_buffer[face_id] = {
                'distances': [],
                'confidences': []
            }
        
        buffer = self.smoothing_buffer[face_id]
        
        buffer['distances'].append(distance)
        buffer['confidences'].append(confidence)
        
        if len(buffer['distances']) > self.buffer_size:
            buffer['distances'] = buffer['distances'][-self.buffer_size:]
        if len(buffer['confidences']) > self.buffer_size:
            buffer['confidences'] = buffer['confidences'][-self.buffer_size:]
        
        # Return smoothed values
        smooth_distance = sum(buffer['distances']) / len(buffer['distances'])
        smooth_confidence = sum(buffer['confidences']) / len(buffer['confidences'])
        
        return smooth_distance, smooth_confidence
    
    def recognize_face(self, face_tensor, threshold=0.85):
        """Recognize face using the loaded database"""
        if face_tensor is None or len(self.known_embeddings) == 0:
            return "Unknown", 999.0
        
        embedding = self.get_embedding(face_tensor)
        if embedding is None:
            return "Unknown", 999.0
        
        # compute distances to all known embeddings
        distances = []
        for known_embedding in self.known_embeddings:
            distance = np.linalg.norm(embedding - known_embedding)
            distances.append(distance)
        
        min_distance = min(distances)
        if min_distance < threshold:
            best_match_idx = distances.index(min_distance)
            return self.known_names[best_match_idx], min_distance
        else:
            return "Unknown", min_distance
    
    def process_frame(self, frame, threshold=0.85):
        """Process frame for face recognition"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        results = []
        
        try:
            boxes, probs = self.mtcnn.detect(pil_image)
            
            if boxes is not None:
                face_tensors = self.mtcnn.extract(pil_image, boxes, save_path=None)
                
                for i, (box, face_tensor) in enumerate(zip(boxes, face_tensors)):
                    if face_tensor is not None:
                        name, distance = self.recognize_face(face_tensor, threshold)
                        left, top, right, bottom = box.astype(int)
                        
                        face_id = f"{left//50}_{top//50}"  # Grid-based ID for tracking
                        
                        # Smooth the distance values
                        detection_confidence = probs[i] if probs is not None else 0.0
                        smooth_distance, _ = self.smooth_values(face_id, distance, detection_confidence)
                        
                        results.append({
                            'box': (left, top, right, bottom),
                            'name': name,
                            'distance': smooth_distance
                        })
        
        except Exception as e:
            print(f"Error processing frame: {e}")
        
        return results
    
    def draw_results(self, frame, results):
        """Draw recognition results on frame"""
        for result in results:
            left, top, right, bottom = result['box']
            name = result['name']
            distance = result['distance']
            
            # Color coding based on distance thresholds - more meaningful than probability
            if name == "Unknown":
                color = (0, 0, 255)  # Red for unknown
            else:
                # Distance-based color coding for known faces
                if distance <= 0.6:
                    color = (0, 255, 0)      # Green - excellent match
                elif distance <= 0.8:
                    color = (0, 255, 255)    # Yellow - good match  
                elif distance <= 1.0:
                    color = (0, 165, 255)    # Orange - fair match
                else:
                    color = (0, 100, 255)    # Red-orange - poor match
            
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            label = f"{name}"
            distance_label = f"Dist: {distance:.2f}"
            
            # draw main label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(frame, (left, top - 45), (left + max(label_size[0], 120), top), color, -1)
            
            # draw name
            cv2.putText(frame, label, (left + 2, top - 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # draw distance instead of probability
            cv2.putText(frame, distance_label, (left + 2, top - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame
    
    def run_live_test(self, threshold=0.85, camera_id=0):
        """Run live face recognition test"""
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("🎥 Starting live face recognition test...")
        print("Controls:")
        print("  'q' - Quit")
        print("  '+' - Increase threshold (less strict)")
        print("  '-' - Decrease threshold (more strict)")
        print("  'i' - Show database info")
        
        frame_count = 0
        fps_start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            # process frame
            results = self.process_frame(frame, threshold)
            
            # draw results
            frame = self.draw_results(frame, results)
            
            # calculate FPS
            frame_count += 1
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - fps_start_time)
                fps_start_time = time.time()
            else:
                fps = 0
            
            info_text = f"Threshold: {threshold:.2f} | Known: {len(self.classes)} people"
            if fps > 0:
                info_text += f" | FPS: {fps:.1f}"
            
            cv2.putText(frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # show detected faces count
            faces_text = f"Faces detected: {len(results)}"
            cv2.putText(frame, faces_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('FaceNet Live Recognition', frame)
            
            # handle controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('+') or key == ord('='):
                threshold += 0.1
                print(f"Threshold: {threshold:.2f}")
            elif key == ord('-'):
                threshold = max(0.1, threshold - 0.1)
                print(f"Threshold: {threshold:.2f}")
            elif key == ord('i'):
                self.show_database_info()
        
        cap.release()
        cv2.destroyAllWindows()
    
    def show_database_info(self):
        """Display database information"""
        print("\n📊 Database Information:")
        print(f"  Known people: {len(self.classes)}")
        print(f"  Total embeddings: {len(self.known_embeddings)}")
        for i, person in enumerate(self.classes):
            count = self.known_names.count(person)
            print(f"    {person}: {count} embeddings")

def main():
    if not os.path.exists("../models/face_database_all_v2.pkl"):
        print("Database not found. Creating from trained_cropped folder...")
        from create_database_from_cropped import FaceDatabaseCreator
        creator = FaceDatabaseCreator("../datasets/train_augmented")
        creator.create_database()
    
    recognizer = LiveFaceRecognitionWithDatabase()
    recognizer.run_live_test(threshold=0.85)

if __name__ == "__main__":
    import os
    main()
