import os
import shutil
import random
from math import floor

# Paths
images_folder = "images"  # all images are here
splits = ["train", "valid", "test"]
split_ratios = {"train": 0.8, "valid": 0.1, "test": 0.1}

# Create split folders inside 'images'
for split in splits:
    os.makedirs(os.path.join(images_folder, split), exist_ok=True)

# Get all image files
all_images = [f for f in os.listdir(images_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
random.shuffle(all_images)

# Compute split counts
n_total = len(all_images)
n_train = floor(split_ratios["train"] * n_total)
n_valid = floor(split_ratios["valid"] * n_total)
n_test = n_total - n_train - n_valid

# Split the images
train_images = all_images[:n_train]
valid_images = all_images[n_train:n_train + n_valid]
test_images = all_images[n_train + n_valid:]

# Function to move images to target folder
def move_images(image_list, split_name):
    dest_folder = os.path.join(images_folder, split_name)
    for img in image_list:
        shutil.move(os.path.join(images_folder, img), os.path.join(dest_folder, img))

# Move images
move_images(train_images, "train")
move_images(valid_images, "valid")
move_images(test_images, "test")

print(f"Moved {len(train_images)} images to train, {len(valid_images)} to valid, {len(test_images)} to test")
