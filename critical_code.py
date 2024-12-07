import cv2
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from timeit import default_timer as timer


lock = threading.Lock()
displayed_images = set() 

# Function to process images (with critical section for race condition)
def process_image_with_critical_section(img_path):
    global displayed_images
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (512, 512))
    enhanced_img = enhance_image_clahe(img_resized)
    dark_channel = estimatedarkchannel(img_resized, 15)
    bright_channel = estimatebrightchannel(img_resized, 15)

    folder_name = os.path.basename(os.path.dirname(img_path))

    # Lock for critical section to ensure thread-safe access
    with lock:
        if folder_name not in displayed_images:
            displayed_images.add(folder_name)  # Safe access to shared resource
            print(f"Processed and displayed: {folder_name}")

# Function to run image processing in parallel
def process_images_parallel():
    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(process_image_with_critical_section, image_paths)

# Benchmark function
def benchmark_image_processing():
    process_images_parallel()

# Benchmark the parallel code
start_time = timer()
benchmark_image_processing()
end_time = timer()

print(f"Parallel execution time with critical section: {end_time - start_time:.2f} seconds")
