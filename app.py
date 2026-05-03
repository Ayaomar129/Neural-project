import cv2
import easyocr
import torch
import os
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image
from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class LicensePlateApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("AI Plate Recognition System - Advanced Version")
        self.geometry("700x500")

        print("Loading YOLOv8...")
        self.yolo_model = YOLO('best.pt')

        print("Loading EasyOCR...")
        self.easy_reader = easyocr.Reader(['en'])

        print("Loading TrOCR (Transformers)... please wait")
        self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
        self.trocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')

        self.label = ctk.CTkLabel(self, text="AI License Plate Recognition System", font=("Arial", 24, "bold"))
        self.label.pack(pady=30)

        self.btn_select = ctk.CTkButton(self, text="Select Car Image", command=self.select_image, width=200, height=40)
        self.btn_select.pack(pady=10)

        self.model_choice = ctk.CTkComboBox(self, values=["EasyOCR", "TrOCR"], font=("Arial", 14), width=200)
        self.model_choice.set("EasyOCR")
        self.model_choice.pack(pady=15)

        self.btn_run = ctk.CTkButton(self, text="run the system", fg_color="#28a745", hover_color="#218838", 
                                    command=self.process_image, width=200, height=50, font=("Arial", 16, "bold"))
        self.btn_run.pack(pady=30)

        self.status_label = ctk.CTkLabel(self, text="ready to use", font=("Arial", 12))
        self.status_label.pack(side="bottom", pady=10)

        self.image_path = ""

    def select_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if self.image_path:
            self.status_label.configure(text=f"you selected: {os.path.basename(self.image_path)}", text_color="white")

    def process_image(self):
        if not self.image_path:
            self.status_label.configure(text="choose an image first", text_color="red")
            return

        self.status_label.configure(text="processing...", text_color="yellow")
        self.update()

        # تشغيل YOLO لاكتشاف اللوحة
        results = self.yolo_model(self.image_path)
        img = cv2.imread(self.image_path)
        selected_engine = self.model_choice.get()

        found_plate = False

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf_yolo = box.conf[0]

                if conf_yolo > 0.4:
                    found_plate = True
                    plate_crop = img[y1:y2, x1:x2]
                    
                    if selected_engine == "EasyOCR":
                        # قراءة النص بـ EasyOCR
                        ocr_data = self.easy_reader.readtext(plate_crop)
                        if ocr_data:
                            # ocr_data[0][1] هو النص، ocr_data[0][2] هو الـ Confidence
                            plate_text = ocr_data[0][1].upper()
                            conf_value = ocr_data[0][2]
                        else:
                            plate_text, conf_value = "Unknown", 0.0
                    else:
                        # قراءة النص بـ TrOCR (Transformers)
                        plate_pil = Image.fromarray(cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB))
                        pixel_values = self.processor(images=plate_pil, return_tensors="pt").pixel_values
                        
                        # توليد النص وحساب الـ Scores
                        outputs = self.trocr_model.generate(pixel_values, return_dict_in_generate=True, output_scores=True)
                        plate_text = self.processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0].upper()
                        
                        # حساب الـ Confidence لـ TrOCR
                        probs = torch.stack(outputs.scores, dim=1).softmax(-1)
                        probs = torch.gather(probs, 2, outputs.sequences[:, 1:, None])
                        conf_value = probs.mean().item()

                    # رسم النتائج على الصورة
                    color = (0, 255, 0) # أخضر
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                    label = f"{selected_engine}: {plate_text} ({conf_value:.2f})"
                    cv2.putText(img, label, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    print(f"[{selected_engine}] Detected: {plate_text} | Confidence: {conf_value:.4f}")

        if not found_plate:
            self.status_label.configure(text="we didn't find any license plates", text_color="orange")
        else:
            self.status_label.configure(text="processing completed successfully!", text_color="green")
            # عرض النتيجة
            cv2.imshow('Detection Result', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == "__main__":
    app = LicensePlateApp()
    app.mainloop()