<h1 align="center"> WhisperS2T-Reborn ⚡ </h1>
<p align="center"><b>An Optimized Speech-to-Text Pipeline for the Whisper Model Using CTranslate2</b></p>

WhisperS2T-Reborn is a modernized fork of [WhisperS2T](https://github.com/shashikg/WhisperS2T), an optimized lightning-fast **Speech-to-Text** (ASR) pipeline. It is tailored for the Whisper model using the CTranslate2 backend to provide faster transcription. It includes several heuristics to enhance transcription accuracy.

[**Whisper**](https://github.com/openai/whisper) is a general-purpose speech recognition model developed by OpenAI. It is trained on a large dataset of diverse audio and is also a multitasking model that can perform multilingual speech recognition, speech translation, and language identification.

## Installation

```sh
pip install -U whisper-s2t-reborn
```

## Quick Start

#### Transcribe a single file

```py
import whisper_s2t

model = whisper_s2t.load_model(model_identifier="large-v3")

files = ['audio1.wav']
lang_codes = ['en']
tasks = ['transcribe']
initial_prompts = [None]

out = model.transcribe_with_vad(files,
                                lang_codes=lang_codes,
                                tasks=tasks,
                                initial_prompts=initial_prompts,
                                batch_size=32)

print(out[0][0]) # Print first utterance for first file
"""
[Console Output]

{'text': "Let's bring in Phil Mackie who is there at the palace...",
 'avg_logprob': -0.25426941679184695,
 'no_speech_prob': 8.147954940795898e-05,
 'start_time': 0.0,
 'end_time': 24.8}
"""
```

#### Batch across multiple files

Passing multiple files allows segments from different files to be batched together, making better use of the GPU:

```py
import whisper_s2t

model = whisper_s2t.load_model(model_identifier="large-v3")

files = ['audio1.wav', 'audio2.wav', 'audio3.wav']
lang_codes = ['en', 'en', 'en']
tasks = ['transcribe', 'transcribe', 'transcribe']
initial_prompts = [None, None, None]

out = model.transcribe_with_vad(files,
                                lang_codes=lang_codes,
                                tasks=tasks,
                                initial_prompts=initial_prompts,
                                batch_size=32)

# out[0] = results for audio1.wav, out[1] = results for audio2.wav, etc.
for file_idx, transcript in enumerate(out):
    print(f"File {files[file_idx]}: {len(transcript)} segments")
```

#### Word-level alignment

To enable word-level timestamps, load the model with:

```py
model = whisper_s2t.load_model("large-v3", asr_options={'word_timestamps': True})
```

## Supported Models

| Model | Identifier |
|:---|:---|
| Tiny | `tiny` / `tiny.en` |
| Base | `base` / `base.en` |
| Small | `small` / `small.en` |
| Medium | `medium` / `medium.en` |
| Large V3 | `large-v3` |
| Large V3 Turbo | `large-v3-turbo` |
| Distil Small | `distil-small.en` |
| Distil Medium | `distil-medium.en` |
| Distil Large V3 | `distil-large-v3` |
| Distil Large V3.5 | `distil-large-v3.5` |

All models are available in `float16`, `float32`, and `bfloat16` compute types via [CTranslate2-4you](https://huggingface.co/ctranslate2-4you) on Hugging Face.

## Benchmarks

**Model:** Whisper `large-v3` · FP16 · CUDA · RTX 4090
**Audio:** [`sam_altman_lex_podcast_367.flac`](https://huggingface.co/datasets/reach-vb/random-audios/blob/main/sam_altman_lex_podcast_367.flac)

Comparing [`openai-whisper`](https://pypi.org/project/openai-whisper/) (no batch support) against [`whisper-s2t-reborn`](https://pypi.org/project/whisper-s2t-reborn/).

| Backend | Batch Size | Time (s) | Speedup | Inference VRAM (MB) |
|:---|:---:|---:|:---:|---:|
| openai-whisper | 1 | 508.5 | 1.0× | 362 |
| whisper-s2t-reborn | 1 | 372.4 | 1.4× | 560 |
| whisper-s2t-reborn | 2 | 239.6 | 2.1× | 840 |
| whisper-s2t-reborn | 4 | 145.5 | 3.5× | 1,387 |
| whisper-s2t-reborn | 8 | 95.5 | 5.3× | 2,427 |
| whisper-s2t-reborn | 16 | 69.4 | 7.3× | 4,608 |
| whisper-s2t-reborn | 32 | 57.1 | 8.9× | 8,964 |
| whisper-s2t-reborn | 64 | 49.8 | 10.2× | 17,665.75 |
> The increased VRAM usage even at batch size 1 is largely due to the VAD model.  Openai's implementation doesn't use voice activity detection.
> The ```benchmarks``` folder has the actual scripts used.

<details><summary>VISUAL OF BENCHMARK RESULTS</summary>

<img width="925" height="980" alt="image" src="https://github.com/user-attachments/assets/4d8afafe-f218-4cf8-8c21-510540c22980" />

</details>

## Acknowledgements
- [**Original WhisperS2T**](https://github.com/shashikg/WhisperS2T): Thanks to shashig for the original WhisperS2T project that this fork is based on.
- [**OpenAI Whisper Team**](https://github.com/openai/whisper): Thanks to the OpenAI Whisper Team for open-sourcing the Whisper model.
- [**CTranslate2 Team**](https://github.com/OpenNMT/CTranslate2/): Thanks to the CTranslate2 Team for providing a faster inference engine for Transformers architecture.
- [**NVIDIA NeMo Team**](https://github.com/NVIDIA/NeMo): Thanks to the NVIDIA NeMo Team for their contribution of the open-source VAD model used in this pipeline.


## License

This project is licensed under MIT License - see the [LICENSE](LICENSE) file for details.

