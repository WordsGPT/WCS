# Gemma 3 27B Unquantized Run

Use this on the GPU machine from the repo root.

```bash
git clone https://github.com/WordsGPT/WCS.git
cd WCS
./goCarlos
```

The script runs `google/gemma-3-27b-pt` with `bfloat16`, not 4-bit or 8-bit
quantization. It creates a local virtual environment, installs dependencies,
detects the available NVIDIA GPUs with `nvidia-smi`, and uses Transformers
`device_map=auto` to shard the model across them.

On two 48 GB GPUs, the detected memory setting will be approximately:

```bash
MAX_MEMORY=0=44GiB,1=44GiB,cpu=160GiB
```

CPU offload remains available if needed. You can still override any setting
inline, but the default path should be just `./goCarlos`.

If Hugging Face access is gated and no token is already present, `./goCarlos`
starts the Hugging Face login flow automatically. Paste a token with Gemma
access when prompted.

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

Skip dependency installation if the environment is already prepared:

```bash
INSTALL_DEPS=0 ./goCarlos
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

The default retry count is zero so real setup/model errors print immediately.
Set `RETRIES=1 ./goCarlos` if you want one automatic retry after transient
failures.
