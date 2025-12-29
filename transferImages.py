import os
import shutil

# Current directory as base
base_dir = "."
categories = [
    "Alternaria_Leaf_Spot",
    "Bacterial_Soft_Rot",
    "Black_Rot",
    "Downy_Mildew",
    "Healthy_Pechay"
]
splits = ["train", "valid", "test"]
destination = os.path.join(base_dir, "images")

# Create destination folder if it doesn't exist
os.makedirs(destination, exist_ok=True)

# Copy all images from train, valid, test folders
for split in splits:
    for category in categories:
        src_folder = os.path.join(base_dir, split, category)
        if not os.path.exists(src_folder):
            print(f"Warning: {src_folder} does not exist!")
            continue
        for filename in os.listdir(src_folder):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                src_path = os.path.join(src_folder, filename)
                dst_path = os.path.join(destination, f"{split}_{category}_{filename}")
                shutil.copy2(src_path, dst_path)

print(f"All images have been copied to {destination}")
