# Benchmarks

Scripts to reproduce the benchmark results from the project README.

## Audio File

Download the test audio used for all benchmarks:

[sam_altman_lex_podcast_367.flac](https://huggingface.co/datasets/reach-vb/random-audios/blob/main/sam_altman_lex_podcast_367.flac) (~2 hours, English)

Place the file in this `benchmarks/` directory (or pass the path via `--audio`).

## Dependencies

These benchmarks measure GPU VRAM usage and require a CUDA-enabled version of PyTorch.
PyPI only ships CPU builds of torch by default, so install the CUDA 12.8 build explicitly:

```sh
pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
```

Then install the NVIDIA CUDA libraries (pip-installable, no system CUDA toolkit required) and benchmark dependencies:

```sh
# NVIDIA CUDA libraries
pip install nvidia-cuda-runtime-cu12==12.8.90 nvidia-cublas-cu12==12.8.4.1 nvidia-cudnn-cu12==9.10.2.21 nvidia-ml-py==13.580.82

# For whisper-s2t-reborn benchmark
pip install whisper-s2t-reborn av

# For openai-whisper benchmark
pip install openai-whisper av

```

> The scripts include a `set_cuda_paths()` helper that automatically configures the pip-installed NVIDIA libraries on Windows, so no system-wide CUDA installation is needed.
>
> `nvidia-ml-py` is imported as `pynvml` in Python. It provides GPU VRAM monitoring via NVML.

## Running

### whisper-s2t-reborn

```sh
python bench_whispers2t.py --audio sam_altman_lex_podcast_367.flac
```

To sweep batch sizes (as in the README table):

```sh
python bench_whispers2t.py --audio sam_altman_lex_podcast_367.flac --batch-size 1
python bench_whispers2t.py --audio sam_altman_lex_podcast_367.flac --batch-size 2
python bench_whispers2t.py --audio sam_altman_lex_podcast_367.flac --batch-size 4
python bench_whispers2t.py --audio sam_altman_lex_podcast_367.flac --batch-size 8
python bench_whispers2t.py --audio sam_altman_lex_podcast_367.flac --batch-size 16
python bench_whispers2t.py --audio sam_altman_lex_podcast_367.flac --batch-size 32
```

### openai-whisper

```sh
python bench_whisper_vanilla.py --audio sam_altman_lex_podcast_367.flac
```

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
| `--model-size` | `large-v3` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large`, `turbo`) |
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
Model:                  large-v3
Audio:                  sam_altman_lex_podcast_367.flac
------------------------------------------------------------
Baseline VRAM:          512.00 MB
VRAM after model load:  2048.00 MB
Peak VRAM (delta):      4096.00 MB
Inference VRAM (delta): 2048.00 MB
Transcription time:     57.10 s
Audio duration:         3601.00 s
Real-Time Factor:       63.07x
------------------------------------------------------------
Transcription: Let's bring in Phil Mackie who is there at the palace...
============================================================
```

Use `--json-output results.json` to save structured results for further analysis.
