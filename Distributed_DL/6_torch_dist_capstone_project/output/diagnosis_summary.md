# Manual Batch-Size Sweep Diagnosis

## Summary Table

| run | local batch | global batch | images/s | comm % | stage1/stage0 | gather ms | loss ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| sweep_bs4 (best) | 4 | 8 | 5.76 | 61.12 | 0.064 | 2.19 | 0.39 |
| sweep_bs8 | 8 | 16 | 5.65 | 61.15 | 0.062 | 6.67 | 0.59 |
| sweep_bs16 | 16 | 32 | 5.46 | 60.08 | 0.063 | 10.96 | 0.93 |
| sweep_bs32 | 32 | 64 | 5.21 | 59.22 | 0.071 | 20.57 | 4.03 |

## Tuning Decision

The best observed configuration is `sweep_bs4` with local batch size 4, global batch size 8, and 5.76 images/s.

The decision is based primarily on global throughput. The secondary evidence is the trace-derived communication percentage, the stage imbalance ratio, and the odd-rank `gather_embeddings` and `loss_calculation` spans, which capture the contrastive-loss overhead.

## Why Larger Batches Stopped Helping

At least one larger batch was slower than the best run. For example, `sweep_bs32` reached 5.21 images/s with communication at 59.22% and an odd/even compute ratio of 0.071x. This suggests the larger batch increased communication or odd-rank loss-side work more than it improved the local-work-to-sync ratio.
