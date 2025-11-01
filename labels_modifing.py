import os
import shutil

# === Configuration ===
SRC_DIR = "data/plates_data"  # Original folder containing images and labels

def rename_labels_to_match_images():
    """Rename labels to match image names"""
    images_dir = os.path.join(SRC_DIR, "images")
    labels_dir = os.path.join(SRC_DIR, "labels")
    
    # Check if source directories exist
    if not os.path.exists(images_dir):
        print(f"❌ Images directory not found: {images_dir}")
        return
    if not os.path.exists(labels_dir):
        print(f"❌ Labels directory not found: {labels_dir}")
        return
    
    # Create directory for new labels
    new_labels_dir = os.path.join(SRC_DIR, "data/plates_dataset_ready")
    os.makedirs(new_labels_dir, exist_ok=True)
    
    all_images = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    matched_count = 0
    unmatched_count = 0
    
    print("🔍 Starting file matching process...")
    
    for img_name in all_images:
        img_base = os.path.splitext(img_name)[0]  # 0_aug_0000
        
        # Find matching label
        matching_label = None
        for label_name in os.listdir(labels_dir):
            if label_name.endswith('.txt') and img_base in label_name:
                matching_label = label_name
                break
        
        if matching_label:
            # Copy label with new name matching the image
            label_src = os.path.join(labels_dir, matching_label)
            label_dst = os.path.join(new_labels_dir, img_base + ".txt")
            shutil.copy(label_src, label_dst)
            matched_count += 1
            if matched_count % 100 == 0:  # Print update every 100 files
                print(f"✅ Matched {matched_count} files...")
        else:
            print(f"⚠️ No label found for: {img_name}")
            unmatched_count += 1
    
    print(f"\n🎯 Final Results:")
    print(f"✅ Matched: {matched_count} files")
    print(f"⚠️ Unmatched: {unmatched_count} files")
    print(f"📁 New labels created in: {new_labels_dir}")
    
    # Show sample of matched files
    if matched_count > 0:
        print(f"\n📋 Sample of matched files:")
        sample_files = os.listdir(new_labels_dir)[:5]
        for file in sample_files:
            print(f"   📄 {file}")

if __name__ == "__main__":
    rename_labels_to_match_images()