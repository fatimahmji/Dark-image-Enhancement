import threading
import os
import cv2
from concurrent.futures import ThreadPoolExecutor
from timeit import default_timer as timer

thread_local = threading.local()

# Function to process images using reduction
def process_image_with_reduction(img_path):
    global thread_local
    if not hasattr(thread_local, 'processed_folders'):
        thread_local.processed_folders = set()

    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (512, 512))
    enhance_image_clahe(img_resized)
    estimatedarkchannel(img_resized, 15)
    estimatebrightchannel(img_resized, 15)

    folder_name = os.path.basename(os.path.dirname(img_path))
    thread_local.processed_folders.add(folder_name)

# Consolidate results from all threads
def collect_results():
    global thread_local
    all_processed = set()
    with ThreadPoolExecutor(max_workers=4) as executor:
        for _ in executor.map(process_image_with_reduction, image_paths):
            if hasattr(thread_local, 'processed_folders'):
                all_processed.update(thread_local.processed_folders)
    return all_processed

# Parallel processing with reduction
def process_images_parallel_reduction():
    processed_folders = collect_results()
    print(f"Processed folders: {processed_folders}")

# Benchmark function
def benchmark_reduction_processing():
    start_time = timer()
    process_images_parallel_reduction()
    end_time = timer()
    print(f"Parallel execution time with reduction: {end_time - start_time:.2f} seconds")

benchmark_reduction_processing()