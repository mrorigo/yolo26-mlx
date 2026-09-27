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
and 17.0 ms on s. The unfused MLX model is faster than that too (8.1 / 39.9 / 16.1 ms), i.e. 1.04x,
1.23x and 1.06x against torch at its own best.

## What the optimization pass changed

Profiling drove three changes, each verified by `tools/check_parity.py` (unchanged) and the test
suite. Numbers are YOLO26n at 640px.

| | before | after |
| --- | ---: | ---: |
| eval forward, batch 1 | 11.5 ms | **8.1 ms** |
| eval forward, batch 8 | 55.5 ms | **39.9 ms** |
| train-mode forward, batch 8 | 96.7 ms | **60.3 ms** |
| train step (fwd+loss+bwd), batch 8 | 169.6 ms | **148.8 ms** |
| criterion (forward), batch 8 | 131.8 ms | **99.0 ms** |
| 640 b8 inference vs torch | 0.91x | **1.48x** |
| 640 b8 criterion vs torch | 0.78x | **1.24x** |

**1. BatchNorm statistics: one reduction, one affine.** `mx.mean` + `mx.var` + `mx.rsqrt` replaced a
materialised `(x - mean)**2`, and the normalisation folds into a single affine `x * scale + shift`.

**2. Fused conv + BatchNorm + SiLU block.** MLX has no conv epilogue, so eager mode paid a
separate pass for the BatchNorm affine and another for the activation. `Conv.forward` is now one
`mx.compile`d function per block, *shared across all layers* (MLX caches per input shape), which is
worth ~40% on the wide early layers.

**3. The compile-cache rules that made it safe.** Measured directly:

- `mx.compile` **freezes arrays captured from the enclosing scope** - a function that multiplies by
  a captured `w` keeps returning `x * 2` after `w[...] = 5`. So the block takes the conv weight and
  the running statistics as *arguments*.
- **Scalar and bool arguments are honoured** (`stride`, `padding`, `groups`, `act` can be passed in
  and reused across layers).
- A compiled **training** forward is a trap either way: captured parameters freeze the model (the
  loss repeated `15.6422, 15.6422, 15.6422` over three steps), and passing the parameter tree in as
  an argument misses the cache every step (4.6 s/step, 25x slower). `DetectionModel.compiled()`
  therefore refuses to run in training mode.

With the per-block fusion in place, a model-level `mx.compile` no longer pays off (640 b1: 8.1 ms
eager, 9.1 ms folded+compiled, 11.7 ms compiled), so it is off by default in `predict()`. The same
fusion *helps* the folded model less than the unfolded one, because folding removes work the block
graph would have fused for free (640 b8: 39.9 ms unfolded, 53.0 ms folded) - folding is for export,
not for MLX inference.

## End-to-end training throughput

The tables above are model-only. With augmentation included (720p source images, 640px, batch 16,
mosaic on), the pipeline is different: augmentation is PIL/NumPy on the CPU and does not touch the
GPU, so it has to overlap with the training step.

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

  What the optimization pass changed here: the CIoU over that broadcast geometry is now one
  compiled graph (18.7 -> 3.8 ms), the top-k claim count is one reduction instead of a Python loop
  per ground truth (the single worst scaling bug: 6.0 -> 3.3 ms at G=100), and the alignment targets
  are sparse - the criterion uses `(label, scale)` per anchor instead of materialising a dense
  `(b, A, nc)` one-hot matrix, which also lets BCE collapse to `sum(softplus) - sum(positives)`.
  Overall 45.8 -> 24.8 ms per branch at 100 objects/image.
- **The optimizer's remaining 8 ms is the Newton-Schulz iterations**, which stay eager: they run on
  a handful of differently-shaped batches, and compiling one graph per shape would cost more in
  tracing than it saves.
- **`mx.argpartition` is now the floor of the assigner**: ~7 ms for one (8, 100, 8400) partition, and
  the reference needs one (top-k) plus a second for `topk2`. A blocked two-stage top-k (max-pool
  blocks, then partition only the winning blocks) would cut that to ~1-2 ms, at the cost of an
  exactness caveat around ties. Not done.
- **`fuse()` is a deployment tool, not a speed-up here** (see above).

## Caveats

- MPS is not Apple's fastest path: neither side uses a Neural Engine, CoreML or TensorRT, so these
  numbers are not comparable to the published YOLO26 figures (T4 TensorRT, Intel CPU ONNX).
- The augmentation pipeline is NumPy/PIL and host-bound, so it is excluded; NMS is host code in this
  implementation and is also excluded.
- `criterion (forward)` includes the forward pass, which is why it is large at batch 8. `TAL
  assigner` isolates the assignment step, `train step` adds the backward pass, and `MuSGD step` times
  the optimizer alone.
