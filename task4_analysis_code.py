import os
import cv2
from concurrent.futures import ThreadPoolExecutor
from timeit import default_timer as timer
import threading

# Global variables for Critical Section
lock = threading.Lock()
displayed_images = set()

# Critical Section Code
def process_image_with_critical_section(img_path):
    global displayed_images
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (512, 512))
    folder_name = os.path.basename(os.path.dirname(img_path))

    # Lock for critical section
    with lock:
        if folder_name not in displayed_images:
            displayed_images.add(folder_name)
            print(f"Critical: Processed and displayed: {folder_name}")

# Atomic Set class for Atomic Operations
class AtomicSet:
    """An atomic, thread-safe set"""
    def __init__(self):
        self._set = set()
        self._lock = threading.Lock()

    def add(self, item):
        """Atomically add an item to the set"""
        with self._lock:
            if item not in self._set:
                self._set.add(item)
                return True
            return False

atomic_displayed_images = AtomicSet()

# Atomic Code for Atomic Operations
def process_image_with_atomic(img_path):
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (512, 512))
    folder_name = os.path.basename(os.path.dirname(img_path))

    # Atomic-like behavior: adding to the set
    if atomic_displayed_images.add(folder_name):
        print(f"Atomic: Processed and displayed: {folder_name}")

# Reduction Code for Reduction Method
thread_local = threading.local()

def process_image_with_reduction(img_path):
    if not hasattr(thread_local, 'processed_folders'):
        thread_local.processed_folders = set()

    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (512, 512))
    folder_name = os.path.basename(os.path.dirname(img_path))
    thread_local.processed_folders.add(folder_name)

def collect_results(image_paths, num_threads):
    all_processed = set()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for _ in executor.map(process_image_with_reduction, image_paths):
            if hasattr(thread_local, 'processed_folders'):
                all_processed.update(thread_local.processed_folders)
    return all_processed

def process_images_parallel_reduction(image_paths, num_threads):
    processed_folders = collect_results(image_paths, num_threads)
    print(f"Reduction: Processed folders: {processed_folders}")

# Parallel processing functions for benchmarking
def process_images_parallel_critical(image_paths, num_threads):
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        executor.map(process_image_with_critical_section, image_paths)

def process_images_parallel_atomic(image_paths, num_threads):
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        executor.map(process_image_with_atomic, image_paths)

# Task 4: Analyze Performance
def analyze_execution(image_paths):
    thread_counts = [1, 2, 4, 8]
    solutions = {
        "Critical Section": process_images_parallel_critical,
        "Atomic Operations": process_images_parallel_atomic,
        "Reduction Method": process_images_parallel_reduction
    }

    for solution_name, processing_function in solutions.items():
        print(f"\nAnalyzing {solution_name}...")
        for num_threads in thread_counts:
            print(f"Testing with {num_threads} threads:")
            start_time = timer()
            processing_function(image_paths, num_threads)
            end_time = timer()
            print(f"Execution time: {end_time - start_time:.2f} seconds")

# Main Function
if __name__ == "__main__":
    dataset_path = "/Users/fatimah/dark images/ExDark"
    
    # Load image paths
    def load_image_paths(dataset_path):
        image_paths = []
        for root, _, files in os.walk(dataset_path):
            for file in files:
                if file.endswith(".jpg"):
                    image_paths.append(os.path.join(root, file))
        return image_paths

    image_paths = load_image_paths(dataset_path)

    # Run the analysis
    analyze_execution(image_paths)
