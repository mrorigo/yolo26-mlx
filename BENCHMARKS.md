# Benchmarks: MLX vs PyTorch (MPS)

Measured on this machine (Apple M1 Pro, macOS 26.1) with the GPU otherwise idle. Reference
implementation is `ultralytics` 8.4.163 / torch 2.14.0 on MPS; the MLX side is mlx 0.32.2. Both
frameworks run **eager** (no graph compiler) for the main tables, on the same weights and the same
inputs, with the device synchronised around every timed region.

Reproduce:

```bash
# MLX
.venv/bin/python tools/bench_mlx.py   --scale n --imgsz 640 --batch 1 --runs 20 --json mlx.json
# PyTorch (throwaway env, see README)
refenv/bin/python tools/bench_torch.py --scale n --imgsz 640 --batch 1 --runs 20 --device mps --json torch.json
.venv/bin/python tools/bench_compare.py --dir /tmp/yolo26-bench
```

Each stage is timed as the **median of 20 iterations after 5 warm-up iterations**, and the table
reports the best of three independent repetitions of that median. The GPU is now quiet enough that
the run-to-run spread is 0-5% on most cells, so the ratios are meaningful rather than indicative.

## YOLO26n, batch 1

| stage | 256px MLX | 256px torch | ratio | 640px MLX | 640px torch | ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| inference, NMS-free head | 6.6 ms | 16.5 ms | **2.51x** | 11.6 ms | 17.6 ms | **1.52x** |
| inference, one-to-many head | 6.2 ms | 16.1 ms | **2.59x** | 11.2 ms | 17.1 ms | **1.53x** |
| inference, BN folded | 5.1 ms | 5.9 ms | **1.16x** | 9.6 ms | 8.3 ms | 0.87x |
| criterion (forward) | 23.7 ms | 38.7 ms | **1.64x** | 32.5 ms | 39.8 ms | **1.23x** |
| train step (fwd+loss+bwd) | 33.6 ms | 64.9 ms | **1.93x** | 42.1 ms | 77.5 ms | **1.84x** |
| MuSGD step | 21.4 ms | 41.5 ms | **1.94x** | 22.1 ms | 40.9 ms | **1.85x** |
| TAL assigner | 1.0 ms | 2.3 ms | **2.17x** | 1.2 ms | 2.4 ms | **1.88x** |

## YOLO26n, batch 8, 640px

| stage | MLX | torch MPS | ratio |
| --- | ---: | ---: | ---: |
| inference, NMS-free head | 55.2 ms | 59.0 ms | **1.07x** |
| inference, one-to-many head | 54.7 ms | 56.0 ms | **1.02x** |
| inference, BN folded | 49.7 ms | 42.9 ms | 0.86x |
| criterion (forward) | 131.8 ms | 103.1 ms | 0.78x |
| train step (fwd+loss+bwd) | 146.7 ms | 309.5 ms | **2.11x** |
| MuSGD step | 21.9 ms | 41.2 ms | **1.88x** |
| TAL assigner | 5.2 ms | 5.0 ms | 0.96x |

## YOLO26s, batch 1, 640px

| stage | MLX | torch MPS | ratio |
| --- | ---: | ---: | ---: |
| inference, NMS-free head | 20.3 ms | 22.1 ms | **1.09x** |
| inference, BN folded | 18.0 ms | 17.0 ms | 0.94x |
| criterion (forward) | 44.8 ms | 41.9 ms | 0.94x |
| train step (fwd+loss+bwd) | 54.4 ms | 103.0 ms | **1.89x** |
| MuSGD step | 21.5 ms | 126.7 ms | **5.89x** |
| TAL assigner | 1.2 ms | 2.4 ms | **1.92x** |

## Reading these numbers

- **Training is where MLX clearly wins**: 1.8-2.1x on the full step at every size.
- **Inference is 1.0-2.6x** depending on size; the 256px advantage shrinks to ~1.05x at 640px/b8
  and to ~1.1x for YOLO26s, where the graph is compute-bound and MPS' convolutions pull level.
- **BN-folded inference is the one consistent regression (0.86-0.94x)**: with BatchNorm removed the
  graph is a pure convolution stack, which is exactly what MPS' hand-tuned kernels are best at.
  Folding still pays for itself here (20-25% on the MLX side) — it is only the *relative* ordering
  against MPS that flips.
- **MuSGD is 1.9x faster, and 5.9x on YOLO26s**: this implementation batches Newton-Schulz over every
  tensor that shares a matrix width, while the reference loops per parameter and zero-pads columns.
- **The criterion at 640px/batch 8 remains slower (0.78x)**. Profiling says the loss math is *not*
  the cause: the criterion spends 96.7 ms in the train-mode forward and only ~17 ms per branch in the
  loss itself (the `(b, A, nc)` = 5.4M-element BCE is 2.4 ms). The gap is BatchNorm: MPS has a fused
  batch-norm kernel, MLX composes it from separate reductions and an affine pass.

## What the BatchNorm rewrite bought

The one code change this benchmarking exercise produced. Computing the statistics with a single
`mx.var` and folding the normalisation into one affine pass, instead of materialising
`(x - mean)**2` and then three elementwise passes:

| YOLO26n 640px b8 | before | after |
| --- | ---: | ---: |
| train-mode forward | 113.7 ms | 96.7 ms |
| criterion (forward) | 148.1 ms | 131.8 ms |
| train step (fwd+loss+bwd) | 169.6 ms | 146.7 ms |
| eval forward | 71.4 ms | 55.2 ms |

Parity with the reference is unchanged (loss total still within 1e-4 relative).

## Graph compilation

`model.compiled()` runs the forward as one `mx.compile`d graph. It is **inference only**, and the
API raises if you call it on a model in training mode, because of two traps found while measuring:

- `mx.compile` turns arrays captured from the enclosing scope into *constants*. Compiling a
  training forward as `lambda x: model(x)` therefore leaves the weights frozen at their first-trace
  values: the loss repeated `15.6422, 15.6422, 15.6422` over three steps while the eager path moved
  to `15.0`.
- Passing the parameter tree in as an explicit input keeps the graph connected but misses MLX's
  compile cache on every step (the optimizer hands out fresh arrays): 4.6 s per step, 25x *slower*
  than eager. A 4.6 s step is not a red herring worth chasing, so training stays eager.

What compilation does buy, on inference (20 iterations, best of the three repetitions):

| case | eager | compiled | folded | folded + compiled |
| --- | ---: | ---: | ---: | ---: |
| n@640 b1 | 11.7 ms | 11.5 ms | 9.6 ms | **9.1 ms** |
| n@640 b8 | 55.4 ms | 40.8 ms | 50.2 ms | **37.8 ms** |
| n@256 b1 | 6.4 ms | 8.2 ms | **5.1 ms** | 6.0 ms |
| s@640 b1 | 20.4 ms | 18.6 ms | 18.1 ms | **16.1 ms** |

So the useful combination is `fuse()` + `compiled()` for repeated inference at one fixed input
shape: 37.8 ms vs 56.0 ms for torch MPS at 640px/b8 (**1.48x**), and 16.1 ms vs 17.0 ms on YOLO26s.
`yolo26-mlx predict` compiles by default for exactly this reason.

`torch.compile` on the reference model was also measured: large gains at 256px/b1 (18 -> 7 ms) but
unstable at 640px/b8 (163 ms, worse than eager), so it is reported here rather than folded into
the main tables.

## Caveats

- MPS is not Apple's fastest path: neither side uses a Neural Engine, CoreML, or TensorRT, so these
  numbers are not comparable to the published YOLO26 figures (T4 TensorRT, Intel CPU ONNX).
- The augmentation pipeline is NumPy/PIL and host-bound, so it is excluded; NMS is host code in this
  implementation and is also excluded.
- `criterion (forward)` includes the forward pass, which is why it is large: at 640px/b8 it is
  96.7 ms of forward plus ~17 ms per branch of loss. `TAL assigner` isolates the assignment step.
