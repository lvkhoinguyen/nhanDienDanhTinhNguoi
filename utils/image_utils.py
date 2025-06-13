import cv2
import numpy as np
from PIL import Image

def capture_face_image(video_capture):
    """Chụp ảnh từ webcam và trả về frame"""
    ret, frame = video_capture.read()
    if ret:
        return frame
    return None

def save_temp_image(image_array, path='static/temp_face.jpg'):
    """Lưu ảnh tạm thời"""
    image = Image.fromarray(image_array)
    image.save(path)
    return path

def load_image_for_preview(path):
    """Load ảnh để preview"""
    image = Image.open(path)
    return np.array(image)