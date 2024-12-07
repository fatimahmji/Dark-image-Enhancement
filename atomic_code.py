import cv2
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from timeit import default_timer as timer
from collections import defaultdict


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
                return True  # Indicates item was newly added
            return False  # Item already in the set

# Atomic set instance for thread-safe operations
atomic_displayed_images = AtomicSet()

# Atomic Code
def process_image_with_atomic(img_path):
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (512, 512))
    folder_name = os.path.basename(os.path.dirname(img_path))

    # Atomic-like behavior: adding to the set
    if atomic_displayed_images.add(folder_name):
        print(f"Atomic: Processed and displayed: {folder_name}")

def process_images_parallel_atomic(image_paths, num_threads):
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        executor.map(process_image_with_atomic, image_paths)


# Parallel processing
def process_images_parallel_atomic():
    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(process_image_with_atomic, image_paths)

# Benchmark function
def benchmark_atomic_processing():
    start_time = timer()
    process_images_parallel_atomic()
    end_time = timer()
    print(f"Parallel execution time with atomic behavior: {end_time - start_time:.2f} seconds")

benchmark_atomic_processing()


