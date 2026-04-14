import time
import ray


ray.init()


@ray.remote
def slow_square(x: int, delay: float = 1.0) -> int:
    time.sleep(delay)
    return x * x


print("Q1: Change xs from 4 to 8 or 16. What happens?")
for n in [4, 8, 16]:
    xs = list(range(1, n + 1))
    start = time.perf_counter()
    ray.get([slow_square.remote(x) for x in xs])
    print(f"  {n} tasks: {time.perf_counter() - start:.2f}s")
print("Answer: Ray time stays ~1s (parallel), speedup grows with task count", end="\n\n")


print("Q2: Change delay from 1.0 to 0.1. When does overhead matter?")
xs = [1, 2, 3, 4]
for delay in [1.0, 0.1, 0.01]:
    start = time.perf_counter()
    ray.get([slow_square.remote(x, delay) for x in xs])
    elapsed = time.perf_counter() - start
    overhead_pct = ((elapsed - delay) / delay) * 100
    print(f"  {delay}s: {elapsed:.3f}s (overhead: {overhead_pct:.0f}%)")
print("Answer: Overhead becomes significant below 0.1s", end="\n\n")


print("Q3: Sleep between submissions. How does that affect ray.get() wait?")
for sleep_time in [0.0, 0.5, 1.0]:
    refs = []
    start = time.perf_counter()
    for x in xs:
        refs.append(slow_square.remote(x, 1.0))
        time.sleep(sleep_time)
    submit_time = time.perf_counter() - start
    start = time.perf_counter()
    ray.get(refs)
    get_time = time.perf_counter() - start
    print(f"  Sleep {sleep_time}s: submit={submit_time:.2f}s, get={get_time:.2f}s")
print("Answer: Tasks run during submission, so ray.get() wait decreases", end="\n\n")


ray.shutdown()
print("Done!")
