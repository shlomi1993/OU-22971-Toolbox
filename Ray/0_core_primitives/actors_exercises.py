import time
import numpy as np
import ray


ray.init()


print("Q1: Create example where stale reads occur with Counter")

@ray.remote
class Counter:
    def __init__(self) -> None:
        self.value = 0

    def increment(self) -> int:
        self.value += 1
        return self.value

    def get_value(self) -> int:
        return self.value

@ray.remote
def stale_reader(counter_handle: ray.actor.ActorHandle) -> int:
    # Read state
    old_value = ray.get(counter_handle.get_value.remote())
    print(f"  Reader: read value={old_value}")
    # Delay before using it
    time.sleep(1)
    # Use stale value
    print(f"  Reader: using stale value={old_value} (may be outdated now)")
    return old_value


counter = Counter.remote()
reader_ref = stale_reader.remote(counter)
time.sleep(0.5)
ray.get(counter.increment.remote())  # Increment while reader is sleeping
print(f"  Counter incremented to: {ray.get(counter.get_value.remote())}")
stale_val = ray.get(reader_ref)
actual_val = ray.get(counter.get_value.remote())
print(f"  Stale read returned: {stale_val}, actual value: {actual_val}")
print("Answer: Reader used stale value while counter was updated", end="\n\n")


print("Q2: Modify Counter to introduce lost updates")

@ray.remote
class BrokenCounter:
    def __init__(self) -> None:
        self.value = 0

    def increment(self, x: int) -> int:
        # Bug: directly sets value instead of incrementing current state
        self.value = x + 1
        return self.value

    def get_value(self) -> int:
        return self.value

@ray.remote
def broken_worker(counter_handle: ray.actor.ActorHandle) -> int:
    old_val = ray.get(counter_handle.get_value.remote())
    time.sleep(0.1)
    new_val = ray.get(counter_handle.increment.remote(old_val))
    return new_val

broken_counter = BrokenCounter.remote()
refs = [broken_worker.remote(broken_counter) for _ in range(3)]
results = ray.get(refs)
final = ray.get(broken_counter.get_value.remote())
print(f"  Worker results: {results}")
print(f"  Final value: {final} (expected 3, but lost updates)")
print("Answer: increment(x) with stale x causes lost updates", end="\n\n")


print("Q3: Extend ToyModelServer with stats tracking")

@ray.remote
class ToyModelServer:
    def __init__(self) -> None:
        print("  Loading model...")
        time.sleep(0.5)
        self.bias = 0.5
        self.num_requests = 0
        self.last_input = None

    def predict(self, x: float) -> float:
        self.num_requests += 1
        self.last_input = x
        return x + self.bias

    def stats(self) -> dict[str, float | int | None]:
        return {
            'num_requests': self.num_requests,
            'last_input': self.last_input
        }

server = ToyModelServer.remote()
ray.get([server.predict.remote(x) for x in [1.0, 2.0, 3.0]])
stats = ray.get(server.stats.remote())
print(f"  Stats: {stats}")  # Output: `  Stats: {'num_requests': 3, 'last_input': 3.0}'")`
print("Answer: Actor tracks num_requests and last_input across calls", end="\n\n")


print("Q4: Four actors vs one - latency and throughput comparison")

@ray.remote
class ToyModelServer:
    def __init__(self) -> None:
        time.sleep(0.5)
        self.bias = 0.5

    def predict(self, x: float) -> float:
        time.sleep(0.1)  # Simulate inference time
        return x + self.bias

inputs = list(np.random.rand(20))

# One actor
print("  Testing with 1 actor:")
single_server = ToyModelServer.remote()
start = time.perf_counter()
results_single = ray.get([single_server.predict.remote(x) for x in inputs])
elapsed_single = time.perf_counter() - start
print(f"    Elapsed: {elapsed_single:.2f}s for {len(inputs)} requests")

# Four actors
print("  Testing with 4 actors:")
servers = [ToyModelServer.remote() for _ in range(4)]
start = time.perf_counter()
refs = []
for i, x in enumerate(inputs):
    server_idx = np.random.randint(4)
    refs.append(servers[server_idx].predict.remote(x))
results_multi = ray.get(refs)
elapsed_multi = time.perf_counter() - start
print(f"    Elapsed: {elapsed_multi:.2f}s for {len(inputs)} requests")
print(f"    Speedup: {elapsed_single / elapsed_multi:.2f}x")
print("Answer: Multiple actors process requests in parallel (higher throughput, lower latency)", end="\n\n")


ray.shutdown()
print("Done!")
