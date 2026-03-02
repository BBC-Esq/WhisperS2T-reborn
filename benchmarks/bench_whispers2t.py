"""
Benchmark script for whisper-s2t-reborn (CTranslate2-based Whisper inference with built-in VAD).

Required:  pip install whisper-s2t-reborn av nvidia-ml-py

Audio used for README benchmarks:
  https://huggingface.co/datasets/reach-vb/random-audios/blob/main/sam_altman_lex_podcast_367.flac

Usage:
  python bench_whispers2t.py --audio path/to/audio.flac
  python bench_whispers2t.py --audio path/to/audio.flac --batch-size 32
  python bench_whispers2t.py --audio path/to/audio.flac --model large-v3-turbo
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
import whisper_s2t
from whisper_s2t.backends.ctranslate2.model import FAST_ASR_OPTIONS

# ── Hardcoded defaults (edit here for quick runs) ────────────────────────────
DEFAULTS = {
    "audio": "sam_altman_lex_podcast_367.flac",
    "model": "large-v3",
    "compute_type": "float16",
    "beam_size": 1,
    "batch_size": 1,
    "language": "en",
    "tokenizer": None,
}

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark whisper-s2t-reborn on an audio file.")
    parser.add_argument("--audio", default=DEFAULTS["audio"], help="Path to input audio file.")
    parser.add_argument("--model", default=DEFAULTS["model"], help="Model identifier or path.")
    parser.add_argument("--compute-type", default=DEFAULTS["compute_type"], help="Compute type (float16, float32, int8, etc.).")
    parser.add_argument("--beam-size", type=int, default=DEFAULTS["beam_size"], help="Beam size for decoding.")
    parser.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"], help="Batch size for VAD transcription.")
    parser.add_argument("--language", default=DEFAULTS["language"], help="Language code.")
    parser.add_argument("--tokenizer", default=DEFAULTS["tokenizer"], help="Tokenizer path for tokens/sec calculation (optional).")
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

def convert_to_wav(audio_file):
    """Convert any audio file to mono 16 kHz WAV for Whisper. Returns path to converted file."""
    output_file = Path(audio_file).stem + "_converted.wav"
    output_path = Path(__file__).parent / output_file

    container = av.open(audio_file)
    stream = next(s for s in container.streams if s.type == 'audio')

    resampler = av.AudioResampler(format='s16', layout='mono', rate=16000)

    output_container = av.open(str(output_path), mode='w')
    output_stream = output_container.add_stream('pcm_s16le', rate=16000)
    output_stream.layout = 'mono'

    for frame in container.decode(audio=0):
        frame.pts = None
        resampled_frames = resampler.resample(frame)
        if resampled_frames is not None:
            for resampled_frame in resampled_frames:
                for packet in output_stream.encode(resampled_frame):
                    output_container.mux(packet)

    for packet in output_stream.encode(None):
        output_container.mux(packet)

    output_container.close()
    container.close()
    return str(output_path)

def print_results(backend, model_name, audio_path, baseline_vram, after_load_vram, peak_vram_delta,
                  inference_vram_delta, transcription_time, audio_duration, transcription="",
                  tokens_per_second=None, total_tokens=None):
    print("\n" + "=" * 60)
    print(f"{'Backend:':<24}{backend}")
    print(f"{'Model:':<24}{Path(model_name).name}")
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
    if total_tokens is not None:
        print(f"{'Total tokens:':<24}{total_tokens}")
    if tokens_per_second is not None:
        print(f"{'Tokens per second:':<24}{tokens_per_second:.2f}")
    print("-" * 60)
    if transcription:
        preview = transcription[:200] + ("..." if len(transcription) > 200 else "")
        print(f"Transcription: {preview}")
    print("=" * 60)

def _benchmark_worker(conn, model_identifier, compute_type, beam_size, batch_size,
                      language, tokenizer_path, audio_file, word_timestamps, device):
    # Resolve device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and compute_type in ("float16", "bfloat16"):
        print(f"WARNING: {compute_type} not supported on CPU, falling back to float32")
        compute_type = "float32"

    use_gpu = (device == "cuda")

    if use_gpu:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        baseline_vram = pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024**2
    else:
        baseline_vram = 0.0

    model_kwargs = {
        'compute_type': compute_type,
        'device': device,
        'asr_options': {**FAST_ASR_OPTIONS, 'beam_size': beam_size, 'word_timestamps': word_timestamps},
        'model_identifier': model_identifier,
    }
    if 'large-v3' in model_identifier:
        model_kwargs['n_mels'] = 128

    model = whisper_s2t.load_model(**model_kwargs)

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

    # Auto-convert non-WAV files (included in pipeline timing)
    converted = None
    ext = Path(audio_file).suffix.lower()
    if ext != ".wav":
        converted = convert_to_wav(audio_file)
        audio_to_use = converted
    else:
        audio_to_use = audio_file

    with torch.inference_mode():
        out = model.transcribe_with_vad(
            [audio_to_use],
            lang_codes=[language],
            tasks=['transcribe'],
            initial_prompts=[None],
            batch_size=batch_size,
        )
        transcription = " ".join(seg['text'] for seg in out[0]).strip()
    elapsed = time.perf_counter() - start

    if use_gpu:
        stop_event.set()
        poll_thread.join()

    # Token counting (optional)
    tokens_per_second = None
    total_tokens = None
    if tokenizer_path:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        total_tokens = len(tokenizer.encode(transcription))
        tokens_per_second = total_tokens / elapsed

    # Clean up converted file
    if converted:
        try:
            os.remove(converted)
        except OSError:
            pass

    if use_gpu:
        pynvml.nvmlShutdown()

    conn.send({
        "baseline_vram": baseline_vram,
        "after_load_vram": after_load_vram,
        "peak_vram_delta": max_vram - baseline_vram,
        "inference_vram_delta": max_vram - after_load_vram,
        "transcription_time": elapsed,
        "transcription": transcription,
        "tokens_per_second": tokens_per_second,
        "total_tokens": total_tokens,
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
        args=(child_conn, args.model, args.compute_type, args.beam_size,
              args.batch_size, args.language, args.tokenizer, args.audio,
              args.word_timestamps, args.device),
    )
    process.start()
    result = parent_conn.recv()
    process.join()

    print_results(
        backend="whisper-s2t-reborn",
        model_name=args.model,
        audio_path=args.audio,
        baseline_vram=result["baseline_vram"],
        after_load_vram=result["after_load_vram"],
        peak_vram_delta=result["peak_vram_delta"],
        inference_vram_delta=result["inference_vram_delta"],
        transcription_time=result["transcription_time"],
        audio_duration=audio_duration,
        transcription=result["transcription"],
        tokens_per_second=result["tokens_per_second"],
        total_tokens=result["total_tokens"],
    )

    if args.json_output:
        rtf = audio_duration / result["transcription_time"] if (audio_duration and result["transcription_time"] > 0) else None
        json_data = {
            "backend": "whisper-s2t-reborn",
            "model": args.model,
            "audio": args.audio,
            "audio_duration_s": audio_duration,
            "batch_size": args.batch_size,
            "beam_size": args.beam_size,
            "compute_type": args.compute_type,
            "language": args.language,
            "word_timestamps": args.word_timestamps,
            "baseline_vram_mb": result["baseline_vram"],
            "after_load_vram_mb": result["after_load_vram"],
            "peak_vram_delta_mb": result["peak_vram_delta"],
            "inference_vram_delta_mb": result["inference_vram_delta"],
            "transcription_time_s": result["transcription_time"],
            "real_time_factor": rtf,
            "tokens_per_second": result["tokens_per_second"],
            "total_tokens": result["total_tokens"],
            "device": result.get("device", args.device),
            "transcription": result["transcription"],
        }
        Path(args.json_output).write_text(json.dumps(json_data, indent=2), encoding="utf-8")

if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args)
