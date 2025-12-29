import cv2
import os
import numpy as np

input_folder = "images/valid"
mask_folder = "masks/valid"
os.makedirs(mask_folder, exist_ok=True)

for file in os.listdir(input_folder):
    if file.lower().endswith((".jpg", ".png", ".jpeg")):
        img = cv2.imread(os.path.join(input_folder, file))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Example threshold for brown/yellow spots (adjust as needed)
        lower = np.array([10, 50, 50])
        upper = np.array([30, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        cv2.imwrite(os.path.join(mask_folder, file), mask)
