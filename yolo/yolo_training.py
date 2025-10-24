from ultralytics import YOLO

def main():
    # YOLO mode initialization
    model = YOLO("yolov8n.pt")

    # Train the model
    model.train(
        optimizer="AdamW",          # Better optimizer for convergence
        lr0=0.001,                  # Initial learning rate
        patience=20,                # Early stopping patience
        data="yolo/digits.yaml",    # Dataset configuration file
        epochs=500,                 # Number of training epochs
        imgsz=640,                  # Image size during training
        batch=16,                   # Batch size
        name="yolo_digits_detector_v4",  # Experiment name
        device=0,                   # Use GPU 0 (or "cpu" if no GPU)
        
        # Data augmentation (make model robust to lighting and distance)
        mosaic=0.3,                 # Medium image mixing (helps generalization)
        hsv_h=0.02,                 # Hue variation
        hsv_v=0.4,                  # Brightness variation
        degrees=5,                  # Slight rotation
        translate=0.05,             # Small translation
        scale=0.15,                  # Moderate scaling (handle near/far digits)
        shear=0.05,                  # Add perspective tilt
        fliplr=0.2,                 # Flip horizontally
        perspective=0.0005,          # Small perspective warp
        rect=False,                 # Randomize aspect ratio
        weight_decay=0.0005,        # Regularization to avoid overfitting

        # Evaluation & visualization
        project="runs/digits_yolo", # Output folder
        save=True,                  # Save checkpoints
        save_period=10,             # Save every 10 epochs
        verbose=True                # Print detailed logs          
    

    )

if __name__ == "__main__":
    main()
