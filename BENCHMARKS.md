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
| inference, NMS-free head | **3.40x** | **2.18x** | **1.48x** | **1.38x** |
| inference, one-to-many head | **3.83x** | **2.24x** | **1.65x** | **1.40x** |
| criterion (forward) | **3.22x** | **1.94x** | **1.24x** | **1.37x** |
| train step (fwd+loss+bwd) | **2.00x** | **1.76x** | **2.42x** | **1.91x** |
| MuSGD step | **1.99x** | **1.90x** | **2.64x** | **6.24x** |
| TAL assigner | **2.12x** | **1.85x** | **1.11x** | **2.01x** |

Raw medians (ms), MLX eager vs torch MPS:

| stage | 256 b1 | 640 b1 | 640 b8 | s 640 b1 |
| --- | --- | --- | --- | --- |
| inference (e2e / o2m) | 4.8 / 4.6 | 8.1 / 7.9 | 39.9 / 39.2 | 16.1 / 15.7 |
| criterion (forward) | 12.5 | 19.9 | 99.0 | 30.4 |
| train step (fwd+loss+bwd) | 33.8 | 43.4 | 148.8 | 54.5 |
| MuSGD step | 21.4 | 21.5 | 21.8 | 20.7 |
| TAL assigner | 1.1 | 1.2 | 5.2 | 1.2 |
| *torch reference* | *16.4 / 17.7* | *17.7 / 17.6* | *59.1 / 64.5* | *22.2 / 22.0* |

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

## Where the remaining differences come from

- **BatchNorm is the one place MPS has a real kernel advantage**: its `batch_norm` is a single fused
  op, whereas MLX composes two reductions plus an affine. That is why the *train-mode* forward is
  still the closest stage (the train step as a whole is 1.8-2.4x ahead because the backward and the
  optimizer pull further in our favour).
- **Small input channels hurt MLX convolutions.** A per-shape microbenchmark
  (`tools/bench_convs.py`) shows the stem (3 -> 16 channels at 640px) at 1.02 ms in MLX vs 0.52 ms on
  MPS, while 1x1 and depthwise convolutions are slightly *faster* in MLX. The fused block hides most
  of this; the per-layer profile (`tools/profile_layers.py`) is what localises it.
- **The criterion is forward-bound at 640px/batch 8**: of the 99 ms, ~60 ms is the forward pass and
  ~34 ms the two branches' loss math. Our loss math is already faster than torch's; the forward
  gap is BatchNorm.
- **`fuse()` is a deployment tool, not a speed-up here** (see above).

## Caveats

- MPS is not Apple's fastest path: neither side uses a Neural Engine, CoreML or TensorRT, so these
  numbers are not comparable to the published YOLO26 figures (T4 TensorRT, Intel CPU ONNX).
- The augmentation pipeline is NumPy/PIL and host-bound, so it is excluded; NMS is host code in this
  implementation and is also excluded.
- `criterion (forward)` includes the forward pass, which is why it is large at batch 8. `TAL
  assigner` isolates the assignment step, `train step` adds the backward pass, and `MuSGD step` times
  the optimizer alone.
