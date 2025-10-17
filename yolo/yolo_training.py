from ultralytics import YOLO

def main():
    # تحميل نموذج YOLOv8n (خفيف ومناسب للبدء)
    model = YOLO("yolov8n.pt")

    # تدريب الموديل
    model.train(
    optimizer="AdamW",
    lr0=0.001,
    patience=15,
    data="yolo/digits.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    name="yolo_digits_tight_v1",
    device=0,
    mosaic=0.3,     
    rect=True,    
    hsv_h=0.02,          
    hsv_v=0.4,           
    degrees=5,           
    translate=0.08,      
    scale=0.3           
    

    )

if __name__ == "__main__":
    main()
