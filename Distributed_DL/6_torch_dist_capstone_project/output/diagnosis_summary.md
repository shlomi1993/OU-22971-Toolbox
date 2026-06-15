# Manual Batch-Size Sweep Diagnosis

## Summary Table

| run | local batch | global batch | images/s | comm % | stage1/stage0 | gather ms | loss ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| ctrl_bs4_layer2 | 4 | 8 | 2.96 | 66.05 | 0.117 | 2.09 | 0.39 |
| ctrl_bs8_layer2 (best) | 8 | 16 | 3.36 | 64.17 | 0.116 | 2.36 | 0.70 |

## Tuning Decision

The best observed configuration is `ctrl_bs8_layer2` with local batch size 8, global batch size 16, and 3.36 images/s.

The decision is based primarily on global throughput. The secondary evidence is the trace-derived communication percentage, the stage imbalance ratio, and the odd-rank `gather_embeddings` and `loss_calculation` spans, which capture the contrastive-loss overhead.

## Larger-Batch Behavior

Throughput did not decline after the best run in this sweep range. A wider sweep should continue until `images/s` flattens or degrades and then inspect whether communication percentage, waiting, or odd-rank loss-side work is responsible.
