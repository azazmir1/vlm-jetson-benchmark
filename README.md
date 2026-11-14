# VLM Compression Benchmark on NVIDIA Jetson Orin Nano

A systematic study of which model-compression methods actually let vision-language models run on a memory-constrained edge device.

**Hardware:** NVIDIA Jetson Orin Nano 8GB (target) · NVIDIA A6000 48GB (baseline reference)
**Scope:** 9 VLM families, 19 models, 256M–8.3B parameters

## The problem

The Jetson Orin Nano has 8 GB of unified memory shared between CPU and GPU — roughly 6 GB usable after the OS. Most VLMs above ~2B parameters either fail to load at all, or load and then prove too slow or memory-hungry to be usable. This repository benchmarks compression methods against both failure modes: shrinking the model enough to fit, and reducing runtime memory enough to be practical once loaded.

## Methods benchmarked

| Category | Methods |
|---|---|
| Post-training quantization | PyTorch INT8, PyTorch INT4, AWQ, GPTQ, bitsandbytes, native INT4 |
| Pruning | magnitude pruning, SparseGPT, Wanda |
| Low-rank / factorization | SVD-LLM, PALU |
| Activation / attention | PACT, CASP, SLIM |
| Fine-tuning | QLoRA |
| Combined | AWP and multi-method pipelines |

## Models

SmolVLM (256M / 500M / 2.2B) · InternVL2.5 (1B / 2B / 4B / 8B) · Qwen2.5-VL (3B / 7B) · LFM2-VL (450M / 1.6B / 3B) · Moondream2 · FastVLM · LLaVA-OneVision · Florence-2 · Ovis2

## Layout

```
compression/    one runner per method (ptq/, pruning/, lowrank/, pact/, palu/, qlora/, ...)
evaluation/     accuracy and benchmark harnesses
profiling/      latency and memory instrumentation
results/        per-model, per-method output (293 files)
scripts/        orchestration and batch sweeps
configs/        experiment configuration
jetson/         device-specific setup and containers
setup_jetson.sh JetPack environment bootstrap
```

`EXPERIMENT_LOG.md` is the full run-by-run record: what was attempted, what loaded, what ran, and what failed.

## Usage

```bash
./setup_jetson.sh
pip install -r requirements.txt
python compression/ptq/run_awq.py --model <hf-model-id>
```
