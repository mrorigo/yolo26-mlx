GOAL: a ground-up implementation of YOLO26 in Apple MLX framework. No pytorch, no shortcuts, 100% MLX

# YOLO26-MLX — working agreements

## Hard rule
The package under `src/yolo26_mlx/` must never import torch/torchvision. PyTorch is allowed in
exactly one place: `tools/export_reference.py` (a development tool that exports reference tensors
for parity checks) and in its own throwaway virtualenv. Nothing in `src/`, `tests/` or the CLI may
depend on it.

## Correctness bar
Before believing any change is right, check it against evidence, in this order:

1. `uv run pytest -q` — 46 tests, ~21s, must stay green.
2. `uv run python tools/check_parity.py --scale n --imgsz 256` — numerical parity with PyTorch
   (forward, both branch outputs, decoded outputs, NMS-free top-k, both loss terms). Needs
   reference data from `tools/export_reference.py`; the parity tests skip if it is missing.
3. Parameter counts must stay equal to the published values for all five scales
   (n 2,572,280 / s 10,009,784 / m 21,896,248 / l 26,299,704 / x 58,993,368).
4. `BENCHMARKS.md` must be refreshed if a change plausibly moves throughput: the GPU is shared, so
   measure best-of-three and check the spread before believing a number.

## Ground rules for the MLX code
- NHWC everywhere, `mx.array` values in float32, params addressed by dotted name.
- Parameters, buffers and derived caches are separate: trainable in `_params`, BN running stats in
  `_buffers`, computed anchors in `_cache` (underscore-prefixed). Only `_params` reach the optimizer
  and `state()`.
- MLX is lazy: never read `.item()`/`.tolist()` on a value inside a training step. Host syncs are
  allowed only in the assigner's shape logic and in metrics.
- `mx.split` takes split points and misbehaves on empty lists; use `split_sizes()` for
  torch-style "split by sizes".
- There is no `mx.where` with a single argument (no `nonzero`) and no `mx.one_hot`; use
  `argsort` on a boolean mask and `one_hot()` in `nn/ops.py`.
- Attention reshapes must move the channel axis *before* the spatial axis
  (`transpose(0, 3, 1, 2)`), not after — the wrong order is numerically wrong but still runs.
- BatchNorm must use Ultralytics' eps=1e-3 / momentum=0.03 (set by their `initialize_weights`),
  with the *unbiased* variance in the running stats, and it must stay inside the fused
  conv+BatchNorm+SiLU block: one `mx.var` reduction, one affine, all as graph *arguments*.
- `mx.compile` **freezes captured arrays but honours scalar and bool arguments**. Every compiled
  graph in this repo therefore takes weights, buffers and statistics as arguments
  (`_conv_block`, `_batch_norm_train`); a captured array would silently go stale on the first
  in-place update.
- Model-level `mx.compile` is off by default: the per-block graphs already fuse everything, so
  wrapping the whole model is slower (8.1 ms eager vs 11.7 ms compiled at 640px/b1). It is also
  inference only, and `DetectionModel.compiled()` enforces that.
- Anything added to `configs/*.yaml` must be reachable from `config.scale_config` and
  `tasks.DetectionModel._make`; keep the two in sync.

## Style
- No comments that restate the code. Docstrings explain *why* and any numerical subtleties.
- Reference the upstream source when porting (`ultralytics/nn/modules/block.py`,
  `ultralytics/utils/tal.py`, `ultralytics/optim/muon.py`, ...) so behaviour can be re-checked.
- Keep imports sorted and the public surface small; `__all__` lists what callers may use.
