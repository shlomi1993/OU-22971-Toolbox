import time
import numpy as np
import ray


ray.init()


print("Q1: Create list of ObjectRefs, put list in store. What does ray.get() return?")
refs = [ray.put(100), ray.put(200), ray.put(300)]
nested_ref = ray.put(refs)
print(f"  ray.get(nested_ref): {ray.get(nested_ref)}")
print(f"  ray.get(ray.get(nested_ref)): {ray.get(ray.get(nested_ref))}")
print("Answer: Need double ray.get() to get values from nested refs", end="\n\n")


print("Q2: Pass ObjectRef as top-level and nested arg. When runnable? What types?")

@ray.remote
def slow_producer() -> str:
    time.sleep(1)
    return "VALUE"

@ray.remote
def consumer(top_level: str, nested_list: list) -> dict:
    return {
        'top_level_type': type(top_level).__name__,
        'nested_type': type(nested_list[0]).__name__
    }

ref = slow_producer.remote()
result = ray.get(consumer.remote(ref, [ref]))
print(f"  Top-level arg type: {result['top_level_type']}")
print(f"  Nested arg type: {result['nested_type']}")
print("Answer: Top-level refs are auto-dereferenced, nested refs are not", end="\n\n")


print("Q3: Repeat serialization timing with range of sizes. When does it matter?")

@ray.remote
def process(x: bytes) -> int:
    return len(x)

for size in [100_000, 1_000_000, 10_000_000]:
    blob = b'x' * size
    start = time.perf_counter()
    ray.get([process.remote(blob) for _ in range(10)])
    by_value = time.perf_counter() - start
    blob_ref = ray.put(blob)
    start = time.perf_counter()
    ray.get([process.remote(blob_ref) for _ in range(10)])
    by_ref = time.perf_counter() - start
    print(f"  {size // 1_000_000}MB: by-value={by_value:.3f}s, by-ref={by_ref:.3f}s, speedup={by_value/by_ref:.1f}x")
print("Answer: Large objects (>1MB) benefit significantly from ray.put()", end="\n\n")


print("Q4: Repeat timing with 10MB NumPy arrays. Is there a difference?")

@ray.remote
def process_array(arr: np.ndarray) -> int:
    return arr.shape[0]

big_array = np.random.rand(10_000_000 // 8)
start = time.perf_counter()
ray.get([process_array.remote(big_array) for _ in range(10)])
by_value = time.perf_counter() - start
array_ref = ray.put(big_array)
start = time.perf_counter()
ray.get([process_array.remote(array_ref) for _ in range(10)])
by_ref = time.perf_counter() - start
print(f"  By-value: {by_value:.3f}s")
print(f"  By-ref: {by_ref:.3f}s")
print(f"  Speedup: {by_value/by_ref:.1f}x")
print(f"  Read-only: {not ray.get(array_ref).flags.writeable}")
print("Answer: NumPy shows even bigger speedup, arrays are read-only", end="\n\n")


ray.shutdown()
print("Done!")
