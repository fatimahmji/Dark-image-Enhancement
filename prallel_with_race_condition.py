import cv2
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from timeit import default_timer as timer

# Paths and variables
dataset_path = "/Users/fatimah/dark images/ExDark"
image_paths = []
displayed_images = set()  # Shared resource
folders = set()

# Functions for dark channel, bright channel, and CLAHE enhancement
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

def enhance_image_clahe(im):
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced_img

# Populate image_paths and folders
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".jpg"):
            image_paths.append(os.path.join(root, file))
            folders.add(root)

# Debugging: Check if images are loaded
print(f"Number of images found: {len(image_paths)}")
print(f"Number of folders found: {len(folders)}")

# Function to process images (with race condition)
def process_image_with_race_condition(img_path):
    global displayed_images

    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (512, 512))
    enhanced_img = enhance_image_clahe(img_resized)
    dark_channel = estimatedarkchannel(img_resized, 15)
    bright_channel = estimatebrightchannel(img_resized, 15)

    folder_name = os.path.basename(os.path.dirname(img_path))

    # Simulate race condition by accessing shared resource without locking
    if folder_name not in displayed_images:
        displayed_images.add(folder_name)  # Race condition here
        print(f"Processed and displayed: {folder_name}")

# Function to run image processing in parallel
def process_images_parallel():
    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(process_image_with_race_condition, image_paths)

# Benchmark function
def benchmark_image_processing():
    process_images_parallel()

# Benchmark the parallel code
start_time = timer()
benchmark_image_processing()
end_time = timer()

print(f"Parallel execution time with race conditions: {end_time - start_time:.2f} seconds")