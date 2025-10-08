from ultralytics import YOLO

def main():
    # نحمّل نموذج YOLOv8n الصغير لتسريع التدريب
    model = YOLO("yolov8n.pt")

    # تدريب الموديل
    model.train(
        data="digits.yaml",   # ملف إعداد البيانات
        epochs=50,            # عدد epochs
        imgsz=640,            # حجم الصورة (يمكن تقليله لو أردت سرعة أكثر)
        batch=16,             # حجم الدفعة
        name="yolo_digits",   # اسم مجلد النتائج
        device=0,             # استخدم كرت الشاشة RTX 3050
        amp=False             # تعطيل AMP لتجنب التوقف
    )

if __name__ == "__main__":
    main()
