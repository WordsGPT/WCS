# Gemma 3 27B Unquantized Run

Use this on the GPU machine from the repo root.

```bash
git clone https://github.com/WordsGPT/WCS.git
cd WCS
./goCarlos
```

The script runs `google/gemma-3-27b-pt` with `bfloat16`, not 4-bit or 8-bit
quantization. It uses Transformers `device_map=auto` with both GPUs capped by
default:

```bash
MAX_MEMORY=0=44GiB,1=44GiB,cpu=160GiB
```

That shards the unquantized model across GPU 0 and GPU 1, with CPU offload
available if needed.

If Hugging Face access is gated, authenticate first:

```bash
source .venv-gemma3-27b/bin/activate
huggingface-cli login
```

Useful smoke test before the full run:

```bash
LIMIT=5 ./goCarlos
```

Run base and instruct:

```bash
MODELS=gemma3-27b-base,gemma3-27b-it ./goCarlos
```

If the GPU has more or less VRAM, adjust the memory cap:

```bash
MAX_MEMORY=0=70GiB,1=70GiB,cpu=160GiB ./goCarlos
```

Outputs:

```text
results/gemma3_27b_unquantized/t1/
results/gemma3_27b_unquantized/t0p7/
results/gemma3_27b_unquantized/t1p5/
logs/gemma3_27b_unquantized/
```

The same command is resumable. If it stops, rerun it and completed outputs will
be skipped.
