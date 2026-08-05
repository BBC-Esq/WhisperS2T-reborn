"""
Benchmark script for OpenAI Whisper (vanilla/original implementation).
Note: This backend does not support batching.

Setup:
  python -m venv .
  .\\Scripts\\activate
  pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
  pip install openai-whisper av huggingface_hub
  pip install nvidia-cuda-runtime-cu12==12.8.90 nvidia-cublas-cu12==12.8.4.1 nvidia-cudnn-cu12==9.10.2.21 nvidia-ml-py==13.580.82

Audio used for README benchmarks:
  https://huggingface.co/datasets/reach-vb/random-audios/blob/main/sam_altman_lex_podcast_367.flac

Usage:
  python bench_whisper_vanilla.py --audio path/to/audio.wav
  python bench_whisper_vanilla.py --audio path/to/audio.wav --model-size large-v3
  python bench_whisper_vanilla.py --audio path/to/audio.wav --model-size distil-large-v3

Distil-Whisper models are loaded from the OpenAI-format checkpoints published in
the distil-whisper Hugging Face repos (same approach as docling PR #3741).
"""
import os
import sys
import platform
from pathlib import Path

import argparse
import json
import time
import threading
import wave
from multiprocessing import Process, Pipe


def set_cuda_paths():
    if platform.system() != "Windows":
        return
    venv_base = Path(sys.executable).parent.parent
    nvidia_base = venv_base / "Lib" / "site-packages" / "nvidia"
    if not nvidia_base.exists():
        return
    paths_to_add = [
        nvidia_base / "cuda_runtime" / "bin",
        nvidia_base / "cuda_runtime" / "lib" / "x64",
        nvidia_base / "cuda_runtime" / "include",
        nvidia_base / "cublas" / "bin",
        nvidia_base / "cudnn" / "bin",
        nvidia_base / "cuda_nvrtc" / "bin",
        nvidia_base / "cuda_nvcc" / "bin",
    ]
    current_value = os.environ.get("PATH", "")
    new_value = os.pathsep.join(
        [str(p) for p in paths_to_add] + ([current_value] if current_value else [])
    )
    os.environ["PATH"] = new_value
    triton_cuda_path = nvidia_base / "cuda_runtime"
    current_cuda_path = os.environ.get("CUDA_PATH", "")
    new_cuda_path = os.pathsep.join(
        [str(triton_cuda_path)]
        + ([current_cuda_path] if current_cuda_path else [])
    )
    os.environ["CUDA_PATH"] = new_cuda_path
    if hasattr(os, "add_dll_directory"):
        for path in paths_to_add:
            if path.exists():
                try:
                    os.add_dll_directory(str(path))
                except OSError:
                    pass


set_cuda_paths()

import torch
import pynvml
import av
import whisper

DISTIL_OPENAI_CHECKPOINTS = {
    "distil-small.en": ("distil-whisper/distil-small.en", "original-model.bin"),
    "distil-medium.en": ("distil-whisper/distil-medium.en", "original-model.bin"),
    "distil-large-v3": ("distil-whisper/distil-large-v3-openai", "model.bin"),
    "distil-large-v3.5": ("distil-whisper/distil-large-v3.5-openai", "model.bin"),
}

# ── Hardcoded defaults (edit here for quick runs) ────────────────────────────
DEFAULTS = {
    "audio": "sam_altman_lex_podcast_367.flac",
    "model_size": "large-v3",
    "beam_size": 1,
    "compute_type": "float16",
    "language": "en",
}

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark OpenAI Whisper (vanilla) on an audio file.")
    parser.add_argument("--audio", default=DEFAULTS["audio"], help="Path to input audio file.")
    parser.add_argument("--model-size", default=DEFAULTS["model_size"],
                        help="Whisper model size (tiny, base, small, medium, large, turbo) "
                             "or a Distil-Whisper model (distil-small.en, distil-medium.en, "
                             "distil-large-v3, distil-large-v3.5).")
    parser.add_argument("--beam-size", type=int, default=DEFAULTS["beam_size"], help="Beam size for decoding.")
    parser.add_argument("--compute-type", default=DEFAULTS["compute_type"],
                        choices=["float16", "float32"],
                        help="Compute type (float16 or float32). Vanilla whisper only supports these two.")
    parser.add_argument("--language", default=DEFAULTS["language"], help="Language code.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                        help="Device for inference (auto, cuda, cpu).")
    parser.add_argument("--word-timestamps", action="store_true", default=False,
                        help="Enable word-level timestamps (cross-attention + DTW alignment).")
    parser.add_argument("--json-output", default=None, help="Path to write JSON results.")
    return parser.parse_args()

def get_audio_duration(audio_file):
    try:
        with av.open(audio_file) as container:
            stream = container.streams.audio[0]
            return float(stream.duration * stream.time_base)
    except Exception:
        try:
            with wave.open(audio_file, 'rb') as wf:
                return wf.getnframes() / float(wf.getframerate())
        except Exception:
            return None

def print_results(backend, model_name, audio_path, baseline_vram, after_load_vram, peak_vram_delta,
                  inference_vram_delta, transcription_time, audio_duration, transcription=""):
    print("\n" + "=" * 60)
    print(f"{'Backend:':<24}{backend}")
    print(f"{'Model:':<24}{model_name}")
    print(f"{'Audio:':<24}{Path(audio_path).name}")
    print("-" * 60)
    print(f"{'Baseline VRAM:':<24}{baseline_vram:.2f} MB")
    print(f"{'VRAM after model load:':<24}{after_load_vram:.2f} MB")
    print(f"{'Peak VRAM (delta):':<24}{peak_vram_delta:.2f} MB")
    print(f"{'Inference VRAM (delta):':<24}{inference_vram_delta:.2f} MB")
    print(f"{'Transcription time:':<24}{transcription_time:.2f} s")
    if audio_duration:
        print(f"{'Audio duration:':<24}{audio_duration:.2f} s")
        rtf = audio_duration / transcription_time if transcription_time > 0 else 0
        print(f"{'Real-Time Factor:':<24}{rtf:.2f}x")
    print("-" * 60)
    if transcription:
        preview = transcription[:200] + ("..." if len(transcription) > 200 else "")
        print(f"Transcription: {preview}")
    print("=" * 60)

def _benchmark_worker(conn, model_size, compute_type, beam_size, language, audio_file, word_timestamps, device):
    # Resolve device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and compute_type == "float16":
        print("WARNING: float16 not supported on CPU, falling back to float32")
        compute_type = "float32"

    use_gpu = (device == "cuda")

    if use_gpu:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        baseline_vram = pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024**2
    else:
        baseline_vram = 0.0

    if model_size in DISTIL_OPENAI_CHECKPOINTS:
        from huggingface_hub import hf_hub_download
        repo_id, filename = DISTIL_OPENAI_CHECKPOINTS[model_size]
        load_target = hf_hub_download(repo_id=repo_id, filename=filename)
    else:
        load_target = model_size

    model = whisper.load_model(load_target, device=device)

    if use_gpu:
        after_load_vram = pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024**2
    else:
        after_load_vram = 0.0

    max_vram = after_load_vram
    stop_event = threading.Event()

    def poll_vram():
        nonlocal max_vram
        while not stop_event.is_set():
            usage = pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024**2
            if usage > max_vram:
                max_vram = usage
            time.sleep(0.1)

    if use_gpu:
        poll_thread = threading.Thread(target=poll_vram, daemon=True)
        poll_thread.start()

    start = time.perf_counter()
    try:
        fp16 = (compute_type == "float16") and use_gpu
        result = model.transcribe(audio_file, language=language, beam_size=beam_size, without_timestamps=True, fp16=fp16, word_timestamps=word_timestamps)
        transcription = result["text"]
        elapsed = time.perf_counter() - start
    finally:
        if use_gpu:
            stop_event.set()
            poll_thread.join()

    if use_gpu:
        pynvml.nvmlShutdown()

    conn.send({
        "baseline_vram": baseline_vram,
        "after_load_vram": after_load_vram,
        "peak_vram_delta": max_vram - baseline_vram,
        "inference_vram_delta": max_vram - after_load_vram,
        "transcription_time": elapsed,
        "transcription": transcription,
        "device": device,
    })
    conn.close()

def run_benchmark(args):
    audio_duration = get_audio_duration(args.audio)
    if audio_duration:
        print(f"Audio duration: {audio_duration:.2f} s ({audio_duration / 60:.1f} min)")

    parent_conn, child_conn = Pipe()
    process = Process(
        target=_benchmark_worker,
        args=(child_conn, args.model_size, args.compute_type, args.beam_size,
              args.language, args.audio, args.word_timestamps, args.device),
    )
    process.start()
    child_conn.close()
    result = None
    try:
        while result is None:
            if parent_conn.poll(1.0):
                result = parent_conn.recv()
            elif not process.is_alive():
                if parent_conn.poll(0):
                    result = parent_conn.recv()
                break
    except EOFError:
        pass
    process.join()
    if result is None:
        sys.exit(f"Benchmark worker died before returning a result (exit code {process.exitcode}).")

    print_results(
        backend="openai-whisper",
        model_name=args.model_size,
        audio_path=args.audio,
        baseline_vram=result["baseline_vram"],
        after_load_vram=result["after_load_vram"],
        peak_vram_delta=result["peak_vram_delta"],
        inference_vram_delta=result["inference_vram_delta"],
        transcription_time=result["transcription_time"],
        audio_duration=audio_duration,
        transcription=result["transcription"],
    )

    if args.json_output:
        rtf = audio_duration / result["transcription_time"] if (audio_duration and result["transcription_time"] > 0) else None
        json_data = {
            "backend": "openai-whisper",
            "model": args.model_size,
            "audio": args.audio,
            "audio_duration_s": audio_duration,
            "batch_size": None,
            "beam_size": args.beam_size,
            "compute_type": args.compute_type,
            "language": args.language,
            "device": result.get("device", args.device),
            "word_timestamps": args.word_timestamps,
            "baseline_vram_mb": result["baseline_vram"],
            "after_load_vram_mb": result["after_load_vram"],
            "peak_vram_delta_mb": result["peak_vram_delta"],
            "inference_vram_delta_mb": result["inference_vram_delta"],
            "transcription_time_s": result["transcription_time"],
            "real_time_factor": rtf,
            "transcription": result["transcription"],
        }
        Path(args.json_output).write_text(json.dumps(json_data, indent=2), encoding="utf-8")

if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args)
