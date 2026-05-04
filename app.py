import cv2
import easyocr
import torch
import customtkinter as ctk
from ultralytics import YOLO

class FinalResultApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("ANPR - Final Result Persistence")
        self.geometry("600x450")

        # تحميل الموديلات
        print("Loading Models...")
        self.yolo_model = YOLO('best.pt')
        self.easy_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        
        # واجهة المستخدم
        self.main_label = ctk.CTkLabel(self, text="نظام تحليل حركة المرور", font=("Arial", 22, "bold"))
        self.main_label.pack(pady=20)

        # المربع اللي هيظهر فيه النتيجة النهائية وتفضل ثابتة
        self.result_frame = ctk.CTkFrame(self, corner_radius=10)
        self.result_frame.pack(pady=10, padx=20, fill="x")

        self.res_title = ctk.CTkLabel(self.result_frame, text="آخر لوحة تم رصدها بدقة عالية:", font=("Arial", 14))
        self.res_title.pack(pady=5)

        self.final_plate_label = ctk.CTkLabel(self.result_frame, text="--- في انتظار البيانات ---", 
                                              font=("Arial", 28, "bold"), text_color="#28a745")
        self.final_plate_label.pack(pady=15)

        self.btn_start = ctk.CTkButton(self, text="تشغيل تحليل الفيديو", command=self.run_system,
                                       width=200, height=45, font=("Arial", 16, "bold"))
        self.btn_start.pack(pady=20)

        # متغيرات الذاكرة
        self.best_text = ""
        self.best_conf = 0.0

    def run_system(self):
        video_path = r'D:\Neural project\dataset\images\Automatic Number Plate Recognition (ANPR) _ Vehicle Number Plate Recognition (1).mp4'
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            frame_count += 1
            display_frame = cv2.resize(frame, (800, 450))

            # 1. Detection
            results = self.yolo_model(display_frame, conf=0.5, verbose=False)

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # 2. OCR كل 5 فريمات لتحسين السرعة
                    if frame_count % 5 == 0:
                        plate_crop = display_frame[y1:y2, x1:x2]
                        if plate_crop.size > 0:
                            ocr_res = self.easy_reader.readtext(plate_crop)
                            
                            if ocr_res:
                                current_text = ocr_res[0][1].upper()
                                current_conf = ocr_res[0][2]

                                # تحديث النتيجة لو كانت أدق
                                if current_conf > self.best_conf:
                                    self.best_text = current_text
                                    self.best_conf = current_conf
                                    
                                    # تحديث النص في واجهة البرنامج (الـ Window) فوراً
                                    self.final_plate_label.configure(text=f"{self.best_text}")

            cv2.imshow('Processing Video...', display_frame)
            
            # تحديث واجهة customtkinter عشان ما تهنجش أثناء الفيديو
            self.update()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        # بعد قفل الفيديو، النتيجة هتفضل مكتوبة في الـ final_plate_label

if __name__ == "__main__":
    app = FinalResultApp()
    app.mainloop()