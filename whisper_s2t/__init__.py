import os
from importlib.metadata import version, PackageNotFoundError
from platformdirs import user_cache_dir
from .utils import write_outputs

try:
    __version__ = version("whisper-s2t-reborn")
except PackageNotFoundError:
    __version__ = "1.8.0"

__all__ = ["load_model", "download_model", "default_word_aligner", "write_outputs", "CACHE_DIR", "BASE_PATH",
           "__version__"]

BASE_PATH = os.path.dirname(__file__)

CACHE_DIR = user_cache_dir("whisper_s2t")


def load_model(model_identifier="large-v3", **model_kwargs):
    # n_mels is no longer inferred from the model name here. The CTranslate2
    # backend derives it from the loaded model itself (model.n_mels), which is
    # authoritative and works for every identifier, alias, and custom HF repo.
    from .backends.ctranslate2.model import WhisperModelCT2
    return WhisperModelCT2(model_identifier, **model_kwargs)


def download_model(size_or_id, compute_type="float16", output_dir=None, local_files_only=False, cache_dir=None,
                   *, force_download=False):
    from .hub import download_model as _download_model
    return _download_model(size_or_id, compute_type, output_dir, local_files_only, cache_dir,
                           force_download=force_download)


def default_word_aligner(size):
    from .hub import default_word_aligner as _default_word_aligner
    return _default_word_aligner(size)