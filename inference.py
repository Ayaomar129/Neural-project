import cv2
import easyocr
from ultralytics import YOLO
import os

model = YOLO('best.pt')

reader = easyocr.Reader(['en'])

def run_inference(source_path):
    # قراءة الصورة أو الفيديو
    results = model(source_path)

    for result in results:
        img = result.orig_img.copy()
        
        for box in result.boxes:
            # إحداثيات المربع (x1, y1, x2, y2)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0] # نسبة التأكد
            
            if conf > 0.4: # لو الموديل متأكد بنسبة أكتر من 40%
                # 1. قص صورة اللوحة
                plate_crop = img[y1:y2, x1:x2]
                
                # 2. قراءة النص من الجزء المقصوص
                # paragraph=True بتساعد في تجميع الحروف مع بعضها
                ocr_result = reader.readtext(plate_crop, detail=0) # بتطلع النصوص بس بدون إحداثيات أو نسب تأكد
                
                if ocr_result:
                    plate_text = " ".join(ocr_result).upper()
                    print(f"Detected Plate: {plate_text} (Confidence: {conf:.2f})")
                    
                    # 3. رسم المربع وكتابة النص على الصورة الأصلية للعرض
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(img, plate_text, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # عرض النتيجة
        cv2.imshow('Final Result', img)
        cv2.waitKey(0) # دوسي أي زرار في الكيبورد عشان يقفل الصورة

# --- تشغيل الكود ---
# حطي مسار أي صورة عندك للتجربة هنا
image_to_test = 'D:\\Neural project\\dataset\\images\\val\\video5_360.jpg'

if os.path.exists(image_to_test):
    run_inference(image_to_test)
else:
    print(f"خطأ: الصورة {image_to_test} مش موجودة، تأكدي من المسار!")

cv2.destroyAllWindows()