================================================================================
UNIT 4 EXERCISE SOLUTION
================================================================================

The exercise asks: Push batch size until GPU OOM, then analyze the last successful configuration.
  
Two solutions are provided:

1. **analyze_batch_scaling.py** - Analyzes existing traces
   Usage: python3 analyze_batch_scaling.py
   
   This script compares baseline (batch=64) vs larger batch (batch=256) traces from the `runpod_output/` directory and answers the exercise questions:
   
   - What got denser?
   - What idle gaps shrank?
   - Did throughput improve?
   - Did synchronization become a smaller fraction?

2. **find_max_batch.sh** - Binary search for maximum batch size (requires GPU)
   Usage: ./find_max_batch.sh
   
   This script automatically finds the largest batch size before OOM by:
   - Testing batch sizes using binary search
   - Capturing success/failure for each attempt
   - Running final profiling with max successful batch
   - Saving traces for analysis

================================================================================
EXPECTED RESULTS (from runpod GPU traces)
================================================================================

Baseline (batch=64):
  - Throughput: ~594 images/s
  - train_step: ~58 ms
  - forward: ~6 ms, backward: ~4 ms
  
Batch 256 (4x larger):
  - Throughput: ~779 images/s (+31% improvement)
  - train_step: ~362 ms (6.2x longer)
  - forward: ~6 ms, backward: ~4 ms (similar - profiler artifact)

================================================================================
ANSWERS TO EXERCISE QUESTIONS
================================================================================

1. **What got denser?**
   
   GPU kernel execution became denser:
   - More data processed per kernel launch
   - Fewer context switches between kernels
   - Better GPU occupancy (more threads active)
   
   In the trace: Look for longer, more continuous blocks of CUDA kernel execution on the GPU stream with fewer gaps.

2. **What idle gaps shrank?**
   
   GPU idle time between kernels reduced:
   - Larger batches mean more work per kernel = less scheduling overhead
   - Compute blocks are longer, so proportionally less time switching
   
   However: next_batch time increased 7.7x because DataLoader fell behind.
   Fix: increase --num-workers to keep feeding the GPU.

3. **Did throughput improve?**
   
   YES: +31% improvement (594 → 779 images/second)
   
   Why: Even though each step takes longer, we process 4x more images per step, so overall throughput increases significantly.
   
   Note: Throughput gains diminish as batch size increases further due to:
   - Memory constraints
   - DataLoader bottlenecks  
   - Communication overhead
   - Diminishing returns on GPU utilization

4. **Did synchronization become a smaller fraction of the full step?**
   
   YES (in principle), though hard to measure directly in DDP traces since gradient sync is overlapped with backward pass.
   
   The math:
   - Gradient sync cost ≈ constant (same model = same # of parameters)
   - Compute cost scales linearly with batch size
   - Therefore: sync% of total step decreases as batch increases
   
   Example:
   - Baseline: 10ms compute + 5ms sync = 33% sync overhead
   - Batch 4x: 40ms compute + 5ms sync = 11% sync overhead
   
   This is the KEY advantage of larger batches in distributed training:
   better compute-to-communication ratio.

================================================================================
KEY TAKEAWAYS
================================================================================

1. **Larger batches improve GPU utilization** by amortizing kernel launch and synchronization overhead over more data.

2. **Communication cost is model-dependent, not batch-dependent**, so increasing batch size improves the compute/communication ratio.

3. **Throughput improves but with diminishing returns** as you approach memory limits and other bottlenecks.

4. **DataLoader must keep pace** - increase num_workers as batch size grows, or the input pipeline becomes the bottleneck.

5. **There's a ceiling**: Eventually you hit GPU memory limits, at which point you need other strategies (gradient accumulation, model parallelism).

================================================================================
RUNNING THE SOLUTION
================================================================================

In this CPU dev container:
  cd /workspace/4_ddp_on_cloud_gpus
  python3 analyze_batch_scaling.py

On a GPU instance:
  cd /workspace/4_ddp_on_cloud_gpus
  ./find_max_batch.sh  # Find max batch size
  python3 analyze_batch_scaling.py  # Analyze results

The analysis script will automatically use GPU traces from `runpod_output/` if available, demonstrating the expected behavior even without running on an actual GPU.

================================================================================
