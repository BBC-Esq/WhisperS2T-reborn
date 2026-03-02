"""
Benchmark script for OpenAI Whisper (vanilla/original implementation).
Note: This backend does not support batching.

Required:  pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu124
           pip install openai-whisper av nvidia-ml-py

Audio used for README benchmarks:
  https://huggingface.co/datasets/reach-vb/random-audios/blob/main/sam_altman_lex_podcast_367.flac

Usage:
  python bench_whisper_vanilla.py --audio path/to/audio.wav
  python bench_whisper_vanilla.py --audio path/to/audio.wav --model-size large-v3
"""
import os
from pathlib import Path

import argparse
import json
import time
import threading
import wave
from multiprocessing import Process, Pipe
import torch
import pynvml
import av
import whisper

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
                        help="Whisper model size (tiny, base, small, medium, large, turbo).")
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

    model = whisper.load_model(model_size, device=device)

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
        poll_thread = threading.Thread(target=poll_vram)
        poll_thread.start()

    start = time.perf_counter()
    fp16 = (compute_type == "float16") and use_gpu
    result = model.transcribe(audio_file, language=language, beam_size=beam_size, without_timestamps=True, fp16=fp16, word_timestamps=word_timestamps)
    transcription = result["text"]
    elapsed = time.perf_counter() - start

    if use_gpu:
        stop_event.set()
        poll_thread.join()
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
    result = parent_conn.recv()
    process.join()

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
