import os


class Config:
    # Path configurations
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASETS_DIR = os.path.join(BASE_DIR, '../datasets')
    RAW_DIR = os.path.join(DATASETS_DIR, 'raw')
    TRAIN_DIR = os.path.join(DATASETS_DIR, 'train')
    MODEL_DIR = os.path.join(BASE_DIR, '../models')

    # Create directories if not exist
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Model parameters
    FACE_DETECTION_MODEL = 'hog'  # 'cnn' for GPU, 'hog' for CPU
    ENCODING_MODEL = 'large'  # 'small' for faster processing
    TOLERANCE = 0.6
    NUM_JITTERS = 1
    MIN_CONFIDENCE = 50.0  # Minimum confidence to recognize as known face

    # Registration settings
    FACE_SAMPLES = 50
    CAPTURE_INTERVAL = 0.5  # seconds between captures

    # UI settings
    WINDOW_TITLE = "Hệ thống nhận diện khuôn mặt"
    WINDOW_SIZE = "1200x800"
    DEFAULT_CAMERA_ID = 0