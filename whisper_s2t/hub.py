import re
import requests

import huggingface_hub
from typing import List, Optional

from . import CACHE_DIR


_SUPPORTED_COMPUTE_TYPES = {"float32", "float16", "bfloat16"}


def _normalize_compute_type(compute_type: str) -> str:
    if compute_type in _SUPPORTED_COMPUTE_TYPES:
        return compute_type
    return "float32"


_MODELS = {
    "tiny": {
        "float32": "ctranslate2-4you/whisper-tiny-ct2-float32",
        "float16": "ctranslate2-4you/whisper-tiny-ct2-float16",
        "bfloat16": "ctranslate2-4you/whisper-tiny-ct2-bfloat16",
    },
    "tiny.en": {
        "float32": "ctranslate2-4you/whisper-tiny.en-ct2-float32",
        "float16": "ctranslate2-4you/whisper-tiny.en-ct2-float16",
        "bfloat16": "ctranslate2-4you/whisper-tiny.en-ct2-bfloat16",
    },
    "base": {
        "float32": "ctranslate2-4you/whisper-base-ct2-float32",
        "float16": "ctranslate2-4you/whisper-base-ct2-float16",
        "bfloat16": "ctranslate2-4you/whisper-base-ct2-bfloat16",
    },
    "base.en": {
        "float32": "ctranslate2-4you/whisper-base.en-ct2-float32",
        "float16": "ctranslate2-4you/whisper-base.en-ct2-float16",
        "bfloat16": "ctranslate2-4you/whisper-base.en-ct2-bfloat16",
    },
    "small": {
        "float32": "ctranslate2-4you/whisper-small-ct2-float32",
        "float16": "ctranslate2-4you/whisper-small-ct2-float16",
        "bfloat16": "ctranslate2-4you/whisper-small-ct2-bfloat16",
    },
    "small.en": {
        "float32": "ctranslate2-4you/whisper-small.en-ct2-float32",
        "float16": "ctranslate2-4you/whisper-small.en-ct2-float16",
        "bfloat16": "ctranslate2-4you/whisper-small.en-ct2-bfloat16",
    },
    "medium": {
        "float32": "ctranslate2-4you/whisper-medium-ct2-float32",
        "float16": "ctranslate2-4you/whisper-medium-ct2-float16",
        "bfloat16": "ctranslate2-4you/whisper-medium-ct2-bfloat16",
    },
    "medium.en": {
        "float32": "ctranslate2-4you/whisper-medium.en-ct2-float32",
        "float16": "ctranslate2-4you/whisper-medium.en-ct2-float16",
        "bfloat16": "ctranslate2-4you/whisper-medium.en-ct2-bfloat16",
    },
    "large-v3": {
        "float32": "ctranslate2-4you/whisper-large-v3-ct2-float32",
        "float16": "ctranslate2-4you/whisper-large-v3-ct2-float16",
        "bfloat16": "ctranslate2-4you/whisper-large-v3-ct2-bfloat16",
    },
    "large-v3-turbo": {
        "float32": "ctranslate2-4you/whisper-large-v3-turbo-ct2-float32",
        "float16": "ctranslate2-4you/whisper-large-v3-turbo-ct2-float16",
        "bfloat16": "ctranslate2-4you/whisper-large-v3-turbo-ct2-bfloat16",
    },
    "distil-small.en": {
        "float32": "ctranslate2-4you/distil-whisper-small.en-ct2-float32",
        "float16": "ctranslate2-4you/distil-whisper-small.en-ct2-float16",
        "bfloat16": "ctranslate2-4you/distil-whisper-small.en-ct2-bfloat16",
    },
    "distil-medium.en": {
        "float32": "ctranslate2-4you/distil-whisper-medium.en-ct2-float32",
        "float16": "ctranslate2-4you/distil-whisper-medium.en-ct2-float16",
        "bfloat16": "ctranslate2-4you/distil-whisper-medium.en-ct2-bfloat16",
    },
    "distil-large-v3": {
        "float32": "ctranslate2-4you/distil-whisper-large-v3-ct2-float32",
        "float16": "ctranslate2-4you/distil-whisper-large-v3-ct2-float16",
        "bfloat16": "ctranslate2-4you/distil-whisper-large-v3-ct2-bfloat16",
    },
    "distil-large-v3.5": {
        "float32": "ctranslate2-4you/whisper-distil-large-v3.5-ct2-float32",
        "float16": "ctranslate2-4you/whisper-distil-large-v3.5-ct2-float16",
        "bfloat16": "ctranslate2-4you/whisper-distil-large-v3.5-ct2-bfloat16",
    },
}

_MODELS["large"] = _MODELS["large-v3"]

_128_MEL_MODELS = {"large-v3", "large", "large-v3-turbo", "distil-large-v3", "distil-large-v3.5"}


def available_models() -> List[str]:
    return list(_MODELS.keys())


def _n_mels(size_or_id: str) -> Optional[int]:
    if size_or_id not in _MODELS:
        return None
    return 128 if size_or_id in _128_MEL_MODELS else 80


def default_word_aligner(size: str) -> Optional[str]:
    if size not in _MODELS:
        raise ValueError(
            "Invalid model size '%s', expected one of: %s"
            % (size, ", ".join(_MODELS.keys()))
        )
    if size in _128_MEL_MODELS:
        return None
    return "tiny.en" if size.endswith(".en") else "tiny"


def download_model(
    size_or_id: str,
    compute_type: str = "float16",
    output_dir: Optional[str] = None,
    local_files_only: bool = False,
    cache_dir: Optional[str] = None,
    *,
    force_download: bool = False,
):
    if re.match(r".*/.*", size_or_id):
        repo_id = size_or_id
    else:
        model_variants = _MODELS.get(size_or_id)
        if model_variants is None:
            raise ValueError(
                "Invalid model size '%s', expected one of: %s"
                % (size_or_id, ", ".join(_MODELS.keys()))
            )

        repo_id = model_variants[_normalize_compute_type(compute_type)]

    allow_patterns = [
        "config.json",
        "model.bin",
        "tokenizer.json",
        "vocabulary.*",
    ]

    kwargs = {
        "local_files_only": local_files_only,
        "allow_patterns": allow_patterns,
        "force_download": force_download,
    }

    if output_dir is not None:
        kwargs["local_dir"] = output_dir

    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
    else:
        kwargs["cache_dir"] = f"{CACHE_DIR}/models"

    if local_files_only:
        return huggingface_hub.snapshot_download(repo_id, **kwargs)

    try:
        return huggingface_hub.snapshot_download(repo_id, **kwargs)
    except (
        huggingface_hub.utils.HfHubHTTPError,
        requests.exceptions.ConnectionError,
    ) as exception:
        print(
            "An error occured while synchronizing the model %s from the Hugging Face Hub:\n%s"
            % (repo_id, exception)
        )
        print("Trying to load the model directly from the local cache, if it exists.")

        kwargs["local_files_only"] = True
        return huggingface_hub.snapshot_download(repo_id, **kwargs)