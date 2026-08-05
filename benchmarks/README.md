# Benchmarks

Scripts to reproduce the benchmark results from the project README.

## Audio File

Download the test audio used for all benchmarks:

[sam_altman_lex_podcast_367.flac](https://huggingface.co/datasets/reach-vb/random-audios/blob/main/sam_altman_lex_podcast_367.flac) (~2 h 9 min, English)

Place the file in this `benchmarks/` directory (or pass the path via `--audio`).

> This FLAC has no duration metadata in its header, so the scripts skip the
> "Audio duration" and "Real-Time Factor" output lines for it.

## Setup

Create a standard Python virtual environment (not conda) and activate it:

```sh
python -m venv .
.\Scripts\activate
```

These benchmarks measure GPU VRAM usage and require a CUDA-enabled version of PyTorch.
PyPI only ships CPU builds of torch by default, so install the CUDA 12.8 build explicitly:

```sh
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
```

Install the NVIDIA CUDA libraries (pip-installable, no system CUDA toolkit required):

```sh
pip install nvidia-cuda-runtime-cu12==12.8.90 nvidia-cublas-cu12==12.8.4.1 nvidia-ml-py
```

Then install the benchmark dependencies for whichever script you want to run:

```sh
# For whisper-s2t-reborn benchmark
pip install whisper-s2t-reborn av

# For openai-whisper benchmark (huggingface_hub fetches the Distil-Whisper checkpoints)
pip install openai-whisper av huggingface_hub
```

> The scripts include a `set_cuda_paths()` helper that automatically configures the pip-installed NVIDIA libraries on Windows, so no system-wide CUDA installation is needed.
>
> `nvidia-ml-py` is imported as `pynvml` in Python. It provides GPU VRAM monitoring via NVML.
>
> cuDNN does not need to be installed: CTranslate2 4.6.3+ uses a pure-CUDA `Conv1d` and ships without cuDNN, and torch's CUDA wheels bundle their own copy for the VAD models.

## Running

The published README table uses `distil-large-v3` for both backends.

### whisper-s2t-reborn

```sh
python bench_whispers2t.py --audio sam_altman_lex_podcast_367.flac --model distil-large-v3 --batch-size 1
```

To sweep batch sizes (as in the README table):

```sh
python bench_whispers2t.py --audio sam_altman_lex_podcast_367.flac --model distil-large-v3 --batch-size 2
python bench_whispers2t.py --audio sam_altman_lex_podcast_367.flac --model distil-large-v3 --batch-size 4
python bench_whispers2t.py --audio sam_altman_lex_podcast_367.flac --model distil-large-v3 --batch-size 8
python bench_whispers2t.py --audio sam_altman_lex_podcast_367.flac --model distil-large-v3 --batch-size 16
python bench_whispers2t.py --audio sam_altman_lex_podcast_367.flac --model distil-large-v3 --batch-size 32
python bench_whispers2t.py --audio sam_altman_lex_podcast_367.flac --model distil-large-v3 --batch-size 64
```

### openai-whisper

```sh
python bench_whisper_vanilla.py --audio sam_altman_lex_podcast_367.flac --model-size distil-large-v3
```

openai-whisper has no built-in Distil-Whisper support, so the script resolves distil model
names to the OpenAI-format checkpoints published in the distil-whisper Hugging Face repos
and passes the downloaded checkpoint to `whisper.load_model()`.

## Command-Line Arguments

### bench_whispers2t.py

| Argument | Default | Description |
|---|---|---|
| `--audio` | `sam_altman_lex_podcast_367.flac` | Path to input audio file |
| `--model` | `large-v3` | Model identifier (e.g. `large-v3`, `large-v3-turbo`, `distil-large-v3`) |
| `--compute-type` | `float16` | Compute type (`float16`, `float32`, `bfloat16`) |
| `--beam-size` | `1` | Beam size for decoding |
| `--batch-size` | `1` | GPU batch size for transcription |
| `--language` | `en` | Language code |
| `--device` | `auto` | Device for inference (`auto`, `cuda`, `cpu`) |
| `--word-timestamps` | off | Enable word-level timestamps |
| `--tokenizer` | none | HuggingFace tokenizer path for tokens/sec calculation |
| `--json-output` | none | Path to write JSON results |

### bench_whisper_vanilla.py

| Argument | Default | Description |
|---|---|---|
| `--audio` | `sam_altman_lex_podcast_367.flac` | Path to input audio file |
| `--model-size` | `large-v3` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large`, `turbo`) or a Distil-Whisper model (`distil-small.en`, `distil-medium.en`, `distil-large-v3`, `distil-large-v3.5`) |
| `--compute-type` | `float16` | Compute type (`float16` or `float32`) |
| `--beam-size` | `1` | Beam size for decoding |
| `--language` | `en` | Language code |
| `--device` | `auto` | Device for inference (`auto`, `cuda`, `cpu`) |
| `--word-timestamps` | off | Enable word-level timestamps |
| `--json-output` | none | Path to write JSON results |

## Output

Both scripts print a summary table to the console:

```
============================================================
Backend:                whisper-s2t-reborn
Model:                  distil-large-v3
Audio:                  sam_altman_lex_podcast_367.flac
------------------------------------------------------------
Baseline VRAM:          1628.08 MB
VRAM after model load:  3878.79 MB
Peak VRAM (delta):      2643.96 MB
Inference VRAM (delta): 393.25 MB
Transcription time:     65.53 s
------------------------------------------------------------
Transcription: We have been a misunderstood and badly mocked org for a long time...
============================================================
```

Use `--json-output results.json` to save structured results for further analysis.
