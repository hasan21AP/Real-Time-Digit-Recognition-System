from ultralytics import YOLO

def main():
    # تحميل نموذج YOLOv8n (خفيف ومناسب للبدء)
    model = YOLO("yolov8n.pt")

    # تدريب الموديل
    model.train(
        data="yolo/digits.yaml",     # ملف إعداد البيانات
        epochs=50,              # عدد الدورات التدريبية
        imgsz=128,              # حجم الصور الفعلي (نفس حجم بياناتك)
        batch=16,               # حجم الدفعة
        name="yolo_digits_128", # اسم مجلد النتائج
        device=0,               # استخدم GPU (RTX 3050)
        amp=False               # تعطيل mixed precision لتفادي مشاكل CUDA
    )

if __name__ == "__main__":
    main()
