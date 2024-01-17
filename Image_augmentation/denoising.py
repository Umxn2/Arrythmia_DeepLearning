import cv2
import os
import numpy as np  
kernel = np.ones((3,3), np.uint8)  
# Function to apply denoising to a given image
def denoise_image(image_path):
    img = cv2.imread(image_path)
    if img is not None:
           
        denoised_img = cv2.fastNlMeansDenoisingColored(img, None, 20, 20, 7, 21)
        img_erosion = cv2.erode(denoised_img, kernel, iterations=1) 
        
        cv2.imwrite(image_path, img_erosion)
        print(f"Denoising applied to {image_path}")
    else:
        print(f"Unable to read image from {image_path}")

# Folder containing PNG files
folder_path = r"C:\Users\Blitz\Downloads\pro_fold_test copy\pro_fold_test copy\5"

# Iterate through files in the folder
for filename in os.listdir(folder_path):
    # Check if the file is a PNG file
    if filename.lower().endswith(".png"):
        # Full path to the image file
        image_path = os.path.join(folder_path, filename)
        
        # Apply denoising to the image
        denoise_image(image_path)