import os
import torch
import subprocess
import sys

def main():
    # Set file paths
    data_yaml = "yolo/plates.yaml"
    weights = "yolov5n.pt"
    project_name = "runs/train"
    exp_name = "yolo5_plates_detector_v1"
    
    # Training parameters
    epochs = 200
    batch_size = 16
    img_size = 320
    device = 0 if torch.cuda.is_available() else 'cpu'
    
    # Use the same Python interpreter as the current environment
    python_exec = sys.executable
    
    # YOLOv5 training command
    cmd = [
        python_exec, os.path.join("yolov5", "train.py"),
        "--img", str(img_size),
        "--batch", str(batch_size),
        "--epochs", str(epochs),
        "--data", data_yaml,
        "--weights", weights,
        "--name", exp_name,
        "--project", project_name,
        "--device", str(device),
        "--exist-ok"
    ]
    
    print(f"🚀 Starting YOLOv5 training on device: {device}")
    subprocess.run(cmd)
    print("✅ Training complete!")

if __name__ == "__main__":
    main()
