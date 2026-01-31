<h1 align="center">

<img width="1536" height="385" alt="splash" src="https://github.com/user-attachments/assets/a27db29b-8ff0-4436-a726-18f26f523dbd" />

</h1>
<p align="center"><b>A Streamlined Speech-to-Text Pipeline for Whisper Models using CTranslate2</b></p>

<hr><br>

## Installation

### Prerequisites

**FFmpeg** is required for audio processing

**GPU Support:** For GPU-accelerated inference using an Nvidia GPU, you need a compatible CUDA Toolkit and cuDNN installed. Refer to the [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) and [cuDNN](https://developer.nvidia.com/cudnn) documentation for installation instructions.

### Install WhisperS2T-Reborn
```sh
pip install whisper-s2t-reborn
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

## Community Projects
[Batch Audio File Transcriber](https://github.com/BBC-Esq/WhisperS2T-transcriber)
