import os
import re
import requests

import huggingface_hub
from typing import List, Optional

from ... import CACHE_DIR


os.makedirs(f"{CACHE_DIR}/models", exist_ok=True)


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
}

_MODELS["large"] = _MODELS["large-v3"]


def available_models() -> List[str]:
    return list(_MODELS.keys())


def download_model(
    size_or_id: str,
    compute_type: str = "float16",
    output_dir: Optional[str] = None,
    local_files_only: bool = False,
    cache_dir: Optional[str] = None,
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

        repo_id = model_variants[compute_type]

    allow_patterns = [
        "config.json",
        "model.bin",
        "tokenizer.json",
        "vocabulary.*",
    ]

    kwargs = {
        "local_files_only": local_files_only,
        "allow_patterns": allow_patterns,
    }

    if output_dir is not None:
        kwargs["local_dir"] = output_dir
        kwargs["local_dir_use_symlinks"] = False

    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
    else:
        kwargs["cache_dir"] = f"{CACHE_DIR}/models"

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
