# Provenance

Where each part of this repository comes from, and what licence governs it. Written so the
provenance of any file can be checked without archaeology.

## Inputs we used

| input | what we took from it | licence |
| --- | --- | --- |
| [YOLO26 paper](https://arxiv.org/abs/2606.03748) | the architecture and the training recipe: DFL-free ltrb regression, the dual one-to-many / one-to-one head, Progressive Loss, STAL label assignment, the MuSGD recipe | CC-BY-4.0 (arXiv default) |
| [Muon](https://arxiv.org/abs/2502.16982) | the Newton-Schulz orthogonalization and its coefficients | method is not copyrightable; the reference Muon implementations are MIT/Apache-2.0 |
| `ultralytics` 8.4.163 | **behaviour only**, as a development-time oracle (see below) | AGPL-3.0 |

## The reference implementation, and why it is not redistributed

`tools/export_reference.py` instantiates the upstream PyTorch model and dumps tensors from it
(weights, inputs, per-branch outputs, decoded boxes, label-assignment internals, criterion values).
`tools/check_parity.py` and `tests/test_parity.py` then diff our MLX implementation against those
tensors, which is how the forward pass, both heads, the NMS-free top-k and both loss terms are
verified.

That reference is **not vendored, not committed, and not a runtime dependency of the package**. It
lives in a separate virtualenv outside this repository (`~/.cache/yolo26-mlx/ref`, recreated with
`uv venv` + `uv pip install torch ultralytics safetensors`) and only the exported `.npy` /
`.safetensors` files are consumed. Nothing in `src/` or `tests/` imports torch; the AGPL code is
present on the machine, not in the distribution.

## Re-derived rather than copied

`configs/yolo26.yaml` and `configs/yolo26-p2.yaml` were originally byte-identical copies of the
upstream configuration files. They are now our own: the same architecture, written from scratch in
our own format documentation and commentary. Architecture parameters - channel widths, repeat
counts, scale multipliers, stride placement - are facts about the model, not protectable
expression, and they are independently corroborated here: the graphs in these files reproduce the
published parameter counts exactly, which `tests/test_model.py` asserts for all five scales
(2,572,280 / 10,009,784 / 21,896,248 / 26,299,704 / 58,993,368).

`MUSGD_SPEC.md` is likewise an independent behavioural specification of the optimizer, written so
the optimizer could be reimplemented from the description alone. It was validated by implementing
its pseudocode from scratch and checking against the reference's own output (exact on the plain-group
trajectory and both momentum buffers, within 2e-4 on the Muon-group trajectory). Where our
implementation deliberately differs from upstream it says so, e.g. the per-tensor update scale.

## Audit of the code

Every file in `src/`, `tools/` and `tests/` was compared against the installed `ultralytics` tree by
distinctive-line overlap (lines longer than 20 characters):

| our file | best upstream match | shared lines |
| --- | --- | ---: |
| `nn/block.py` | `nn/modules/block.py` | 21.5% |
| `loss/tal.py` | `utils/tal.py` | 8.8% |
| `nn/head.py` | `nn/modules/head.py` | 8.5% |
| `optim/musgd.py` | `optim/muon.py` | 5.3% |
| all other files | - | <= 2.9% |

Every shared line in the worst case is a function signature, a default argument, a one-line
attribute assignment, or `from __future__ import annotations` - the minimum expression needed to
express the same architecture. A separate scan of all comments and docstrings in this repository
against upstream found **zero** verbatim shared lines.

This is engineering evidence, not a legal opinion: a scan cannot settle "derivative work", which
is a fact- and jurisdiction-specific question. Get counsel to sign off before distributing.

## Note on history

The two configuration files were byte-identical copies of AGPL-licensed upstream files when the
repository was first published, and are still reachable in its git history. The working tree no
longer contains any upstream material.

## What is ours

The MLX implementation in `src/yolo26_mlx/`: the module system, the layers, the graph builder, the
label assignment, the criterion, the optimizer, the augmentation pipeline, the metrics, the
trainer, the CLI, the benchmark and profiling tools, and the tests - together with the benchmarks
and measurements in `BENCHMARKS.md`, which are our own measurements on our hardware.
