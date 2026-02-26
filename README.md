<h1 align="center"> WhisperS2T-Reborn ⚡ </h1>
<p align="center"><b>An Optimized Speech-to-Text Pipeline for the Whisper Model Using CTranslate2</b></p>
<br>

WhisperS2T-Reborn is a modernized fork of [WhisperS2T](https://github.com/shashikg/WhisperS2T), an optimized lightning-fast **Speech-to-Text** (ASR) pipeline. It is tailored for the Whisper model using the CTranslate2 backend to provide faster transcription. It includes several heuristics to enhance transcription accuracy.

[**Whisper**](https://github.com/openai/whisper) is a general-purpose speech recognition model developed by OpenAI. It is trained on a large dataset of diverse audio and is also a multitasking model that can perform multilingual speech recognition, speech translation, and language identification.

## Getting Started

### Installation

You must install FFMPEG on your system first.

Install or update to the latest released version of WhisperS2T-Reborn:

```sh
pip install -U whisper-s2t-reborn
```

### Quick Start

```py
import whisper_s2t

model = whisper_s2t.load_model(model_identifier="large-v3")

files = ['data/KINCAID46/audio/1.wav']
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

{'text': "Let's bring in Phil Mackie who is there at the palace. We're looking at Teresa and Philip May. Philip, can you see how he's being transferred from the helicopters? It looks like, as you said, the beast. It's got its headlights on because the sun is beginning to set now, certainly sinking behind some clouds. It's about a quarter of a mile away down the Grand Drive",
 'avg_logprob': -0.25426941679184695,
 'no_speech_prob': 8.147954940795898e-05,
 'start_time': 0.0,
 'end_time': 24.8}
"""
```

To enable word-level alignment, load the model with:

```py
model = whisper_s2t.load_model("large-v3", asr_options={'word_timestamps': True})
```

## Acknowledgements
- [**Original WhisperS2T**](https://github.com/shashikg/WhisperS2T): Thanks to shashig for the original WhisperS2T project that this fork is based on.
- [**OpenAI Whisper Team**](https://github.com/openai/whisper): Thanks to the OpenAI Whisper Team for open-sourcing the Whisper model.
- [**CTranslate2 Team**](https://github.com/OpenNMT/CTranslate2/): Thanks to the CTranslate2 Team for providing a faster inference engine for Transformers architecture.
- [**NVIDIA NeMo Team**](https://github.com/NVIDIA/NeMo): Thanks to the NVIDIA NeMo Team for their contribution of the open-source VAD model used in this pipeline.


## License

This project is licensed under MIT License - see the [LICENSE](LICENSE) file for details.

