import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from timeit import timeit

# Functions for dark and bright channel estimation
def estimatedarkchannel(im, sz):
    b, g, r = cv2.split(im)
    dc = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
    dark = cv2.erode(dc, kernel)
    return dark

def estimatebrightchannel(im, sz):
    b, g, r = cv2.split(im)
    bc = cv2.max(cv2.max(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
    bright = cv2.dilate(bc, kernel)
    return bright

# CLAHE for image enhancement
def enhance_image_clahe(im):
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)  # Convert to LAB color space
    l, a, b = cv2.split(lab)  # Split LAB channels
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))  # Create CLAHE object
    cl = clahe.apply(l)  # Apply CLAHE to L-channel
    limg = cv2.merge((cl, a, b))  # Merge enhanced L-channel back
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)  # Convert back to BGR
    return enhanced_img

# Load the ExDark dataset
dataset_path = "/Users/fatimah/Desktop/level 9/parallel computing/dark images/ExDark"
image_paths = []
folders = set()  # To keep track of unique folders

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".jpg"):
            image_paths.append(os.path.join(root, file))
            folders.add(root)  # Add the folder path

# Check if images are loaded
print(f"Number of images found: {len(image_paths)}")
print(f"Number of folders found: {len(folders)}")

# Sequential processing: Enhance images and display one from each class
def process_images():
    displayed_images = set()  # To track displayed images

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not load image: {img_path}")
            continue

        img_resized = cv2.resize(img, (512, 512))  # Resize to 512x512
        enhanced_img = enhance_image_clahe(img_resized)  # Apply CLAHE for enhancement
        
        # Dark and Bright channel estimations
        dark_channel = estimatedarkchannel(img_resized, 15)
        bright_channel = estimatebrightchannel(img_resized, 15)

        # Display only one image per folder
        folder_name = os.path.basename(os.path.dirname(img_path))
        if folder_name not in displayed_images:
            displayed_images.add(folder_name)

            # Display the original image, enhanced image, dark channel, and bright channel
            plt.figure(figsize=(15, 5))

            # Original image
            plt.subplot(1, 4, 1)
            plt.imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
            plt.title(f'Original Image from {folder_name}')
            plt.axis('off')

            # Enhanced image
            plt.subplot(1, 4, 2)
            plt.imshow(cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB))
            plt.title(f'Enhanced Image (CLAHE) from {folder_name}')
            plt.axis('off')

            # Dark channel
            plt.subplot(1, 4, 3)
            plt.imshow(dark_channel, cmap='gray')
            plt.title('Dark Channel')
            plt.axis('off')

            # Bright channel
            plt.subplot(1, 4, 4)
            plt.imshow(bright_channel, cmap='gray')
            plt.title('Bright Channel')
            plt.axis('off')

            plt.show()

# Function to benchmark the image processing
def benchmark_image_processing():
    process_images()

# Benchmark the sequential code
execution_time = timeit(benchmark_image_processing, number=1)  
print(f"Sequential execution time: {execution_time} seconds")