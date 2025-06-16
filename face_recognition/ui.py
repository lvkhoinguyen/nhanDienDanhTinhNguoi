import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog
from PIL import Image, ImageTk
import threading
import os
import cv2
import time
from config import Config


class FaceRecognitionUI:
    def __init__(self, root, controller):
        self.root = root
        self.controller = controller
        self.is_camera_active = False
        self.current_photo = None  # To keep reference to photo
        self.setup_ui()
        self.result_window = None

    def setup_ui(self):
        """Thiết lập giao diện"""
        self.root.title(Config.WINDOW_TITLE)
        self.root.geometry(Config.WINDOW_SIZE)

        # Frame chính
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Frame hiển thị video
        video_frame = ttk.LabelFrame(main_frame, text="Camera")
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.video_label = ttk.Label(video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Frame điều khiển
        control_frame = ttk.LabelFrame(main_frame, text="Điều khiển")
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)

        # Nút chức năng
        ttk.Button(control_frame, text="Bật Camera", command=self.start_camera).pack(fill=tk.X, pady=5)
        ttk.Button(control_frame, text="Tắt Camera", command=self.stop_camera).pack(fill=tk.X, pady=5)
        ttk.Button(control_frame, text="Đăng ký khuôn mặt", command=self.register_face).pack(fill=tk.X, pady=5)
        ttk.Button(control_frame, text="Huấn luyện Model", command=self.train_model).pack(fill=tk.X, pady=5)

        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        ttk.Button(control_frame, text="Bắt đầu nhận diện", command=self.start_recognition).pack(fill=tk.X, pady=5)
        ttk.Button(control_frame, text="Dừng nhận diện", command=self.stop_recognition).pack(fill=tk.X, pady=5)
        ttk.Button(control_frame, text="Xử lý ảnh", command=self.process_image).pack(fill=tk.X, pady=5)

        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # Hiển thị độ chính xác
        self.show_accuracy_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="Hiển thị độ chính xác",
                        variable=self.show_accuracy_var).pack(fill=tk.X, pady=5)

        # Model detection
        self.model_var = tk.StringVar(value=Config.FACE_DETECTION_MODEL.upper())
        model_frame = ttk.LabelFrame(control_frame, text="Mô hình phát hiện")
        model_frame.pack(fill=tk.X, pady=5)

        ttk.Radiobutton(model_frame, text="CNN (Chính xác)", variable=self.model_var,
                        value="CNN", command=self.change_model).pack(anchor=tk.W)
        ttk.Radiobutton(model_frame, text="HOG (Nhanh)", variable=self.model_var,
                        value="HOG", command=self.change_model).pack(anchor=tk.W)

        # Hiển thị thông tin
        info_frame = ttk.LabelFrame(control_frame, text="Thông tin hệ thống")
        info_frame.pack(fill=tk.X, pady=5)

        self.avg_accuracy_label = ttk.Label(info_frame, text="Độ chính xác trung bình: -")
        self.avg_accuracy_label.pack(anchor=tk.W, padx=5, pady=2)

        self.faces_count_label = ttk.Label(info_frame, text="Số khuôn mặt: 0")
        self.faces_count_label.pack(anchor=tk.W, padx=5, pady=2)

        self.status_label = ttk.Label(info_frame, text="Trạng thái: Sẵn sàng")
        self.status_label.pack(anchor=tk.W, padx=5, pady=2)

        # Biến lưu trữ độ chính xác
        self.accuracy_values = []
        self.avg_accuracy = 0
        self.faces_count = 0

        # Bắt đầu cập nhật video
        self.update_video()

    def change_model(self):
        """Thay đổi mô hình phát hiện khuôn mặt"""
        model_type = self.model_var.get().lower()
        Config.FACE_DETECTION_MODEL = model_type
        self.status_label.config(text=f"Trạng thái: Đã chuyển sang mô hình {model_type.upper()}")

    def update_video(self):
        """Cập nhật hiển thị video"""
        if self.is_camera_active:
            frame = self.controller.get_frame()
            if frame is not None:
                # Nhận diện khuôn mặt nếu đang ở chế độ nhận diện
                if self.controller.recognizing:
                    results = self.controller.recognize_faces(frame)
                    self.faces_count = len(results)
                    self.faces_count_label.config(text=f"Số khuôn mặt: {self.faces_count}")

                    # Lấy danh sách độ chính xác
                    self.accuracy_values = [result['confidence'] for result in results]

                    # Tính độ chính xác trung bình
                    if self.accuracy_values:
                        self.avg_accuracy = sum(self.accuracy_values) / len(self.accuracy_values)
                        self.avg_accuracy_label.config(
                            text=f"Độ chính xác trung bình: {self.avg_accuracy:.2f}%"
                        )
                    else:
                        self.avg_accuracy_label.config(text="Độ chính xác trung bình: -")

                    # Vẽ kết quả nhận diện lên frame
                    for result in results:
                        top, right, bottom, left = result['location']
                        name = result['name']
                        confidence = result['confidence']

                        # Chọn màu dựa trên kết quả nhận diện
                        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

                        # Vẽ hộp xung quanh khuôn mặt
                        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

                        # Vẽ nền cho tên
                        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)

                        # Tạo văn bản hiển thị
                        display_text = name
                        if self.show_accuracy_var.get():
                            display_text = f"{name} ({confidence:.1f}%)"

                        # Hiển thị tên và độ chính xác
                        cv2.putText(frame, display_text, (left + 6, bottom - 6),
                                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)

                # Chuyển đổi frame để hiển thị trong Tkinter
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)

                # Tạo PhotoImage và giữ reference
                self.current_photo = ImageTk.PhotoImage(image=img)
                self.video_label.configure(image=self.current_photo)

        # Lặp lại sau 10ms
        self.root.after(10, self.update_video)

    def start_camera(self):
        """Bật camera"""
        if self.controller.start_camera():
            self.is_camera_active = True
            self.status_label.config(text="Trạng thái: Camera đang hoạt động")
        else:
            error_msg = "Không thể mở camera!\n\nNguyên nhân có thể:\n" \
                        "1. Camera chưa được kết nối\n" \
                        "2. Camera đang được ứng dụng khác sử dụng\n" \
                        "3. Driver camera có vấn đề\n" \
                        "4. Chưa cấp quyền truy cập camera\n\n" \
                        "Cách khắc phục:\n" \
                        "- Thử cắm lại camera\n" \
                        "- Đóng các ứng dụng khác\n" \
                        "- Kiểm tra quyền truy cập camera"
            messagebox.showerror("Lỗi Camera", error_msg)

    def stop_camera(self):
        """Tắt camera"""
        self.controller.stop_camera()
        self.is_camera_active = False
        self.status_label.config(text="Trạng thái: Camera đã tắt")
        self.video_label.configure(image='')
        self.current_photo = None  # Release photo reference

    def register_face(self):
        """Đăng ký khuôn mặt mới"""
        if not self.is_camera_active:
            messagebox.showwarning("Cảnh báo", "Vui lòng bật camera trước!")
            return

        name = simpledialog.askstring("Nhập tên", "Nhập tên người dùng mới:")
        if name:
            self.status_label.config(text=f"Trạng thái: Đang đăng ký {name}...")

            # Tạo cửa sổ tiến trình
            progress_window = tk.Toplevel(self.root)
            progress_window.title("Đang đăng ký...")
            progress_window.geometry("300x100")
            progress_window.resizable(False, False)

            progress = ttk.Progressbar(progress_window,
                                       mode='determinate',
                                       maximum=Config.FACE_SAMPLES)
            progress.pack(pady=10, padx=20, fill=tk.X)
            progress['value'] = 0

            label = ttk.Label(progress_window, text=f"Đang chụp ảnh cho {name}")
            label.pack()

            # Chạy trong thread riêng
            def register_thread():
                try:
                    # Cập nhật tiến trình
                    for i in range(Config.FACE_SAMPLES):
                        progress['value'] = i + 1
                        progress.update()
                        time.sleep(Config.CAPTURE_INTERVAL)

                    # Gọi hàm đăng ký
                    success = self.controller.register_face(name)
                    if success:
                        self.root.after(0, lambda: self.status_label.config(
                            text=f"Trạng thái: Đã đăng ký xong {name}"
                        ))
                        self.root.after(0, lambda: messagebox.showinfo("Thành công",
                                                                       f"Đã đăng ký xong {name} với {Config.FACE_SAMPLES} ảnh"))
                    else:
                        self.root.after(0, lambda: messagebox.showerror("Lỗi", "Đăng ký không thành công!"))
                except Exception as e:
                    self.root.after(0, lambda: messagebox.showerror("Lỗi", str(e)))
                finally:
                    self.root.after(0, progress_window.destroy)

            threading.Thread(target=register_thread, daemon=True).start()

    def train_model(self):
        """Huấn luyện lại model"""
        if not os.path.exists(Config.TRAIN_DIR) or not os.listdir(Config.TRAIN_DIR):
            messagebox.showwarning("Cảnh báo", "Không có dữ liệu để train! Hãy đăng ký khuôn mặt trước.")
            return

        # Tạo dialog tiến trình
        train_dialog = tk.Toplevel(self.root)
        train_dialog.title("Đang train model...")
        train_dialog.geometry("300x100")
        train_dialog.resizable(False, False)

        progress = ttk.Progressbar(train_dialog, mode='indeterminate')
        progress.pack(pady=20, padx=50, fill=tk.X)
        progress.start()

        label = ttk.Label(train_dialog, text="Đang huấn luyện model, vui lòng chờ...")
        label.pack()

        # Huấn luyện trong thread riêng
        def train_thread():
            try:
                num_faces = self.controller.train_model()
                self.root.after(0, lambda: self.status_label.config(
                    text=f"Trạng thái: Đã huấn luyện xong {num_faces} khuôn mặt"
                ))
                self.root.after(0, lambda: messagebox.showinfo("Thành công",
                                                               f"Đã huấn luyện xong model với {num_faces} khuôn mặt"))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Lỗi", str(e)))
            finally:
                self.root.after(0, train_dialog.destroy)

        threading.Thread(target=train_thread, daemon=True).start()

    def start_recognition(self):
        """Bắt đầu nhận diện"""
        if not self.is_camera_active:
            messagebox.showwarning("Cảnh báo", "Vui lòng bật camera trước!")
            return

        if not self.controller.model.known_face_encodings:
            messagebox.showwarning("Cảnh báo", "Chưa có model để nhận diện! Hãy huấn luyện model trước.")
            return

        self.controller.recognizing = True
        self.status_label.config(text="Trạng thái: Đang nhận diện...")

    def stop_recognition(self):
        """Dừng nhận diện"""
        self.controller.recognizing = False
        self.status_label.config(text="Trạng thái: Đã dừng nhận diện")
        self.faces_count_label.config(text="Số khuôn mặt: 0")
        self.avg_accuracy_label.config(text="Độ chính xác trung bình: -")

    def process_image(self):
        """Xử lý ảnh tĩnh từ file"""
        if not self.controller.model.known_face_encodings:
            messagebox.showwarning("Cảnh báo", "Chưa có model để nhận diện! Hãy huấn luyện model trước.")
            return

        # Tạo dialog tiến trình
        process_dialog = tk.Toplevel(self.root)
        process_dialog.title("Đang xử lý ảnh...")
        process_dialog.geometry("300x100")
        process_dialog.resizable(False, False)

        progress = ttk.Progressbar(process_dialog, mode='indeterminate')
        progress.pack(pady=20, padx=50, fill=tk.X)
        progress.start()

        label = ttk.Label(process_dialog, text="Đang nhận diện khuôn mặt trong ảnh...")
        label.pack()

        # Xử lý trong thread riêng
        def process_thread():
            try:
                # Xử lý ảnh
                result_image, results = self.controller.process_image_file()
                if result_image is None:
                    self.root.after(0, lambda: messagebox.showerror("Lỗi", "Không thể đọc ảnh!"))
                else:
                    # Hiển thị kết quả trong cửa sổ mới
                    self.root.after(0, lambda: self.show_image_result(result_image, results))

            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Lỗi", str(e)))
            finally:
                self.root.after(0, process_dialog.destroy)

        threading.Thread(target=process_thread, daemon=True).start()

    def show_image_result(self, image, results):
        """Hiển thị kết quả nhận diện trong cửa sổ mới"""
        # Đóng cửa sổ kết quả cũ nếu có
        if self.result_window:
            try:
                self.result_window.destroy()
            except:
                pass

        # Tạo cửa sổ mới
        self.result_window = tk.Toplevel(self.root)
        self.result_window.title("Kết quả nhận diện ảnh")
        self.result_window.geometry("900x700")

        # Frame chính
        main_frame = ttk.Frame(self.result_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Frame hiển thị ảnh
        img_frame = ttk.LabelFrame(main_frame, text="Ảnh kết quả")
        img_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Chuyển đổi ảnh để hiển thị
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(image_rgb)

        # Tính toán kích thước phù hợp
        img_width, img_height = img_pil.size
        max_width = 800
        max_height = 500

        if img_width > max_width or img_height > max_height:
            ratio = min(max_width / img_width, max_height / img_height)
            new_width = int(img_width * ratio)
            new_height = int(img_height * ratio)
            img_pil = img_pil.resize((new_width, new_height), Image.LANCZOS)

        img_tk = ImageTk.PhotoImage(image=img_pil)

        # Tạo label để hiển thị ảnh
        img_label = ttk.Label(img_frame, image=img_tk)
        img_label.image = img_tk  # Giữ reference
        img_label.pack(pady=10)

        # Frame kết quả chi tiết
        result_frame = ttk.LabelFrame(main_frame, text="Chi tiết nhận diện")
        result_frame.pack(fill=tk.BOTH, expand=True)

        # Tạo bảng kết quả
        columns = ("STT", "Tên", "Độ chính xác", "Vị trí")
        tree = ttk.Treeview(result_frame, columns=columns, show="headings", height=min(5, len(results)))

        # Đặt độ rộng cột
        tree.column("STT", width=50, anchor=tk.CENTER)
        tree.column("Tên", width=150)
        tree.column("Độ chính xác", width=100, anchor=tk.CENTER)
        tree.column("Vị trí", width=300)

        # Đặt tiêu đề
        for col in columns:
            tree.heading(col, text=col)

        # Thêm dữ liệu
        for i, result in enumerate(results, 1):
            top, right, bottom, left = result['location']
            tree.insert("", "end", values=(
                i,
                result['name'],
                f"{result['confidence']:.2f}%",
                f"Top: {top}, Right: {right}, Bottom: {bottom}, Left: {left}"
            ))

        # Thêm thanh cuộn
        scrollbar = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscroll=scrollbar.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Frame nút
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)

        # Nút lưu ảnh
        ttk.Button(button_frame, text="Lưu ảnh kết quả",
                   command=lambda: self.save_result_image(image)).pack(side=tk.LEFT, padx=5)

        # Nút đóng
        ttk.Button(button_frame, text="Đóng", command=self.result_window.destroy).pack(side=tk.RIGHT, padx=5)

    def save_result_image(self, image):
        """Lưu ảnh kết quả ra file"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*")]
        )
        if file_path:
            cv2.imwrite(file_path, image)
            messagebox.showinfo("Thành công", f"Đã lưu ảnh kết quả thành công:\n{file_path}")