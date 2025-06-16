import tkinter as tk
from face_controller import FaceController
from ui import FaceRecognitionUI

def main():
    root = tk.Tk()
    controller = FaceController()
    app = FaceRecognitionUI(root, controller)
    root.mainloop()

if __name__ == "__main__":
    main()