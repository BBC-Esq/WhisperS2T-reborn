<h1 align="center">WhisperS2T-Reborn ⚡</h1>
<p align="center"><b>A Streamlined Speech-to-Text Pipeline for Whisper Models using CTranslate2</b></p>

<hr><br>

WhisperS2T-Reborn is a streamlined fork of the original [WhisperS2T](https://github.com/shashikg/WhisperS2T) project, focused exclusively on the CTranslate2 backend for fast and efficient speech transcription.

## What's Different from the Original?

This fork simplifies the original WhisperS2T by:

- **Single Backend Focus**: Removed TensorRT-LLM, HuggingFace, and OpenAI backends—CTranslate2 only
- **Curated Model Selection**: Uses optimized CTranslate2 whisper models from [ctranslate2-4you](https://huggingface.co/ctranslate2-4you) on HuggingFace
- **Cleaner Codebase**: Streamlined architecture with reduced dependencies
- **Simplified Setup**: Easier installation without complex backend configurations

## Features

- 🚀 **Fast Inference**: CTranslate2 backend provides excellent speed/accuracy tradeoff
- 🎙️ **Built-in VAD**: Integrated Voice Activity Detection using NeMo's Marblenet models
- 🎧 **Flexible Audio Input**: Handles both small and large audio files efficiently
- 🌐 **Multi-language Support**: Transcription and translation for 99+ languages
- ⏱️ **Word-level Timestamps**: Optional word alignment for precise timing
- 📝 **Multiple Export Formats**: Export to TXT, JSON, TSV, SRT, and VTT

## Supported Models

| Model | English-only | Multilingual |
|-------|--------------|--------------|
| tiny | ✅ tiny.en | ✅ tiny |
| base | ✅ base.en | ✅ base |
| small | ✅ small.en | ✅ small |
| medium | ✅ medium.en | ✅ medium |
| large-v3 | — | ✅ large-v3 |
| distil-small.en | ✅ | — |
| distil-medium.en | ✅ | — |
| distil-large-v3 | — | ✅ |

## Installation

### Prerequisites

Install FFmpeg for audio processing:

**Ubuntu/Debian:**
```sh
apt-get install -y libsndfile1 ffmpeg
```

**macOS:**
```sh
brew install ffmpeg
```

**Conda (any platform):**
```sh
conda install conda-forge::ffmpeg
```

### Install WhisperS2T-Reborn
```sh
pip install whisper-s2t-reborn
```

Or install from source:
```sh
pip install git+https://github.com/YOUR_USERNAME/whisper_s2t_reborn.git
```

### CUDA Setup (GPU Users)

If using pip-installed CUDA libraries, add them to your library path:
```sh
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:`python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))'`
```

## Quick Start

### Basic Transcription
```python
import whisper_s2t

# Load model (downloads automatically on first use)
model = whisper_s2t.load_model(model_identifier="large-v3")

# Transcribe with VAD
files = ['audio/sample.wav']
out = model.transcribe_with_vad(files,
                                lang_codes=['en'],
                                tasks=['transcribe'],
                                initial_prompts=[None],
                                batch_size=32)

print(out[0][0])
# {'text': 'Your transcribed text here...',
#  'avg_logprob': -0.25,
#  'no_speech_prob': 0.0001,
#  'start_time': 0.0,
#  'end_time': 24.8}
```

### With Word Timestamps
```python
model = whisper_s2t.load_model("large-v3", asr_options={'word_timestamps': True})

out = model.transcribe_with_vad(files,
                                lang_codes=['en'],
                                tasks=['transcribe'],
                                initial_prompts=[None],
                                batch_size=32)
```

### Export Transcripts
```python
from whisper_s2t import write_outputs

# Export to various formats
write_outputs(out, format='srt', save_dir='./output/')
write_outputs(out, format='vtt', save_dir='./output/')
write_outputs(out, format='json', save_dir='./output/')
```

### Translation
```python
# Translate non-English audio to English
out = model.transcribe_with_vad(files,
                                lang_codes=['fr'],  # Source language
                                tasks=['translate'],  # Translate to English
                                initial_prompts=[None],
                                batch_size=32)
```

## Configuration Options

### Model Loading Options
```python
model = whisper_s2t.load_model(
    model_identifier="large-v3",  # Model name or path
    device="cuda",                 # "cuda" or "cpu"
    compute_type="float16",        # "float16", "float32", or "bfloat16"
    asr_options={
        'beam_size': 5,
        'word_timestamps': False,
        'repetition_penalty': 1.01,
    }
)
```

### Transcription Options
```python
out = model.transcribe_with_vad(
    files,
    lang_codes=['en'],           # Language codes for each file
    tasks=['transcribe'],        # 'transcribe' or 'translate'
    initial_prompts=[None],      # Optional prompts for each file
    batch_size=32                # Batch size for inference
)
```

## Acknowledgements

- [**Original WhisperS2T**](https://github.com/shashikg/WhisperS2T): This project is a fork of WhisperS2T by Shashi Kant Gupta
- [**OpenAI Whisper**](https://github.com/openai/whisper): The foundational Whisper model
- [**CTranslate2**](https://github.com/OpenNMT/CTranslate2/): Fast inference engine for Transformer models
- [**NVIDIA NeMo**](https://github.com/NVIDIA/NeMo): VAD models used in this pipeline
