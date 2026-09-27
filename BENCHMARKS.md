# Benchmarks: MLX vs PyTorch (MPS)

Measured on this machine (Apple M1 Pro, macOS 26.1) with the GPU otherwise idle. Reference
implementation is `ultralytics` 8.4.163 / torch 2.14.0 on MPS; the MLX side is mlx 0.32.2. Both
frameworks run **eager** (no model-level graph compiler) for every table, on the same weights and
the same inputs, with the device synchronised around each timed region.

Reproduce:

```bash
# MLX
.venv/bin/python tools/bench_mlx.py   --scale n --imgsz 640 --batch 1 --runs 20 --json mlx.json
# PyTorch (throwaway env, see README)
refenv/bin/python tools/bench_torch.py --scale n --imgsz 640 --batch 1 --runs 20 --device mps --json torch.json
.venv/bin/python tools/bench_compare.py --dir /tmp/yolo26-bench
```

Each stage is timed as the **median of 20 iterations after 5 warm-up iterations**; the tables
report the best of three independent repetitions. On an idle GPU the run-to-run spread is 0-5% on
most cells.

## Summary: every stage is faster than PyTorch

| stage | n 256 b1 | n 640 b1 | n 640 b8 | s 640 b1 |
| --- | ---: | ---: | ---: | ---: |
| inference, NMS-free head | **3.4x** | **2.2x** | **1.6x** | **1.4x** |
| criterion (forward) | **3.5x** | **2.1x** | **1.5x** | **1.5x** |
| train step (fwd+loss+bwd) | **2.0x** | **1.8x** | **2.5x** | **1.9x** |
| MuSGD step | **5.6x** | **5.6x** | **7.7x** | **17.6x** |
| TAL assigner | **2.1x** | **1.9x** | **1.1x** | **2.0x** |

Raw medians (ms), MLX eager vs torch MPS:

| stage | 256 b1 | 640 b1 | 640 b8 | s 640 b1 |
| --- | --- | --- | --- | --- |
| inference (NMS-free) | 4.8 | 8.1 | 38.1 | 15.3 |
| criterion (forward) | 11.5 | 18.3 | 83.4 | 27.7 |
| train step (fwd+loss+bwd) | 34.2 | 42.2 | 142.1 | 54.6 |
| MuSGD step | 7.3 | 7.3 | 7.5 | 7.3 |
| TAL assigner | 1.1 | 1.2 | 5.2 | 1.2 |
| *torch reference* | *16.4* | *17.7* | *59.1* | *22.2* |

Torch's own BN-fused model (its best inference configuration) is 8.4 ms at 640 b1, 49.1 ms at 640 b8
and 17.0 ms on s. The unfused MLX model is faster than that too (8.1 / 39.9 / 15.3 ms), i.e. 1.04x,
1.23x and 1.11x against torch at its own best.

## Optimization log

Three profiling-driven passes, in order. Every number below is YOLO26n at 640px unless stated,
best of three 20-iteration medians, and every pass was checked against `tools/check_parity.py` (the
parity result is unchanged throughout) and the test suite.

| | start | now |
| --- | ---: | ---: |
| eval forward, batch 1 | 11.5 ms | **8.1 ms** |
| eval forward, batch 8 | 55.5 ms | **39.9 ms** |
| train-mode forward, batch 8 | 113.7 ms | **60.3 ms** |
| criterion (forward), batch 8 | 148.1 ms | **83.4 ms** |
| train step (fwd+loss+bwd), batch 8 | 169.6 ms | **142.1 ms** |
| MuSGD step | 22.2 ms | **7.3 ms** |
| 640 b8 inference vs torch | 0.91x | **1.6x** |
| 640 b8 criterion vs torch | 0.78x | **1.5x** |
| end-to-end training | 18.1 img/s | **45.4 img/s** |

### Pass 1 — the model forward: one compiled conv + BatchNorm + SiLU block

**BatchNorm statistics.** `mx.mean` + `mx.var` + `mx.rsqrt` replaced a materialised
`(x - mean)**2`, and the normalisation folds into a single affine `x * scale + shift`. Train-mode
forward 113.7 -> 96.7 ms; criterion 148.1 -> 131.8 ms.

**The fused block.** MLX has no conv epilogue, so eager mode still paid a separate pass for the
BatchNorm affine and another for the activation. `Conv.forward` became one `mx.compile`d function
*shared across all layers* (MLX caches per input shape), worth ~40% on the wide early layers: eval
forward 55.5 -> 39.9 ms at batch 8, train-mode forward 96.7 -> 60.3 ms.

**The compile-cache rules that made it safe** - all three measured directly:

- `mx.compile` **freezes arrays captured from the enclosing scope**: a function multiplying by a
  captured `w` keeps returning `x * 2` after `w[...] = 5`. The block therefore takes the conv weight
  and the running statistics as *arguments*.
- **Scalar and bool arguments are honoured** (`stride`, `padding`, `groups`, `act` can be passed in
  and reused across layers).
- A compiled **training** forward is a trap either way: captured parameters freeze the model (the
  loss repeated `15.6422, 15.6422, 15.6422` over three steps), and passing the parameter tree in as
  an argument misses the cache every step (4.6 s/step, 25x slower). `DetectionModel.compiled()`
  therefore refuses to run in training mode.

With per-block fusion in place a model-level `mx.compile` no longer pays off (640 b1: 8.1 ms eager,
9.1 ms folded+compiled, 11.7 ms compiled), so `predict()` is eager. The same fusion *helps* the folded
model less than the unfolded one, because folding removes work the block graph would have fused for
free (640 b8: 39.9 ms unfolded, 53.0 ms folded) - folding is for export, not for MLX inference.

### Pass 2 — the criterion, the label assignment and the data pipeline

**The criterion was hiding behind the benchmark's two objects per image.** The assigner is evaluated
over a (batch, ground truths, anchors) tensor, so object count dominates it. At 640px/batch 8:

| objects/image | criterion per branch, before | after |
| ---: | ---: | ---: |
| 2 | 9.8 ms | 7.7 ms |
| 20 | 15.5 ms | 10.5 ms |
| 100 | 45.8 ms | 24.8 ms |

Three changes, each measured in isolation: the CIoU over the broadcast geometry is now one compiled
graph (18.7 -> 3.8 ms at 100 objects); the top-k claim count is one reduction instead of a Python
loop per ground truth (the worst scaling bug in the codebase: 6.0 -> 3.3 ms); and the alignment
targets became *sparse*, so the criterion carries a label and a scale per anchor instead of
materialising a dense `(b, A, nc)` one-hot matrix - which also lets BCE collapse to
`sum(softplus) - sum(positives)`. Two shape-contract bugs surfaced on the way and are fixed: the
empty-batch path returned a per-image scale where the normaliser is per-anchor, and it returned the
wrong length.

**Data loading overlapped** (see the next section): 18.1 -> 45.4 img/s.

### Pass 3 — the optimizer: launch-bound, not arithmetic-bound

The step took ~22 ms for *every* scale, because the tensor count is identical regardless of width
(366 tensors: 126 muon, 240 plain) - about 40 us per tensor of kernel-launch overhead. Compiling the
momentum recurrences and parameter updates into one graph per pass took the step from **22.2 ms to
8.2 ms** (n) and **22.7 ms to 8.3 ms** (s), which is 5.6x and 17.6x the reference MuSGD. On a full
training step that is 129.3 -> 119.6 ms at batch 8 (61.9 -> 66.9 img/s, +8%).

This one hinged on a second `mx.compile` property: it specialises on scalar argument *values*, so
passing the learning rate as a Python float re-traces the graph on every step under a schedule -
**230 ms a step**, far worse than the 22 ms being saved. The same scalars as 0-d arrays are graph
inputs, so the trace is reused (0.5 ms a call) and the new value is honoured. The compiled pass
returns new trees and the caller assigns them per tensor, which measures 0.07 ms for 366 tensors.

Newton-Schulz deliberately stays eager: it runs on a handful of differently-shaped batches, and
tracing one graph per shape would cost more than it saves. That is what the remaining 8 ms is.

## End-to-end training throughput

The tables above are model-only. With augmentation included (720p source images, 640px, batch 16,
mosaic on) the pipeline is different, because augmentation is PIL/NumPy on the CPU and never touches
the GPU - it has to overlap with the training step (pass 2 above).

| configuration | img/s |
| --- | ---: |
| loader inline (`workers=0`) | 18.1 |
| loader with 2 threads (`workers=2`, the default) | **45.4** |
| loader with 3 threads | 45.3 |
| training step alone (ceiling, 8 objects/image) | 94.6 |

`YOLOBatchLoader` draws the index order on the calling thread and prepares batches in a bounded
thread pool, so prefetching cannot change *what* an epoch contains, only when it is ready
(`tests/test_train.py` asserts both properties). Augmentation is GIL-bound: one worker is no faster
than none, two nearly double it, and a third adds nothing - the remaining gap to the 94.6 img/s
ceiling is GIL contention between the workers and the main thread's graph construction, not raw CPU
work. Escaping it needs process-based workers, which would move the per-batch images across a pipe
and is the obvious next step rather than more threading.

## Where the remaining differences come from

- **BatchNorm is the one place MPS has a real kernel advantage**: its `batch_norm` is a single fused
  op, whereas MLX composes two reductions plus an affine. That is why the *train-mode* forward is
  still the closest stage (the train step as a whole is 1.8-2.4x ahead because the backward and the
  optimizer pull further in our favour).
- **Small input channels hurt MLX convolutions.** A per-shape microbenchmark
  (`tools/bench_convs.py`) shows the stem (3 -> 16 channels at 640px) at 1.02 ms in MLX vs 0.52 ms on
  MPS, while 1x1 and depthwise convolutions are slightly *faster* in MLX. The fused block hides most
  of this; the per-layer profile (`tools/profile_layers.py`) is what localises it.
- **The criterion scales with objects per image, and that is where most of its time goes.** The
  assigner is evaluated over a (batch, ground truths, anchors) tensor, so a 640px image with 100
  objects costs far more than one with 2. Measured at 640px/batch 8:

  | objects/image | criterion per branch | assigner |
  | ---: | ---: | ---: |
  | 2 | 7.7 ms | 4.4 ms |
  | 20 | 10.5 ms | 7.2 ms |
  | 100 | 24.8 ms | 21.5 ms |

  Pass 2 above cut this from 45.8 to 24.8 ms per branch at 100 objects/image; the floor is now
  `mx.argpartition` (below).
- **`mx.argpartition` is the floor of the assigner**: ~7 ms for one (8, 100, 8400) partition, and
  the reference needs one (top-k) plus a second for `topk2`. A blocked two-stage top-k (max-pool
  blocks, then partition only the winning blocks) would cut that to ~1-2 ms, at the cost of an
  exactness caveat around ties. Not done.
- **The optimizer's 8 ms is Newton-Schulz, left eager on purpose** (see pass 3): it runs on a
  handful of differently-shaped batches, and one compiled graph per shape would cost more in tracing
  than it saves.
- **`fuse()` is a deployment tool, not a speed-up here** (see pass 1).

## Caveats

- MPS is not Apple's fastest path: neither side uses a Neural Engine, CoreML or TensorRT, so these
  numbers are not comparable to the published YOLO26 figures (T4 TensorRT, Intel CPU ONNX).
- The augmentation pipeline is NumPy/PIL and host-bound, so it is excluded; NMS is host code in this
  implementation and is also excluded.
- `criterion (forward)` includes the forward pass, which is why it is large at batch 8. `TAL
  assigner` isolates the assignment step, `train step` adds the backward pass, and `MuSGD step` times
  the optimizer alone.
