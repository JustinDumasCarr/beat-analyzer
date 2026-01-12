#!/usr/bin/env python3
"""
Audio Analyzer - Beat, tempo, and structure analysis for Suno tracks.
Uses librosa + madmom for robust music analysis.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import librosa
    from madmom.features.beats import DBNBeatTrackingProcessor, RNNBeatProcessor
    from madmom.features.downbeats import DBNDownBeatTrackingProcessor, RNNDownBeatProcessor
except ImportError as e:
    print(f"Error: Missing dependency - {e}")
    print("Run: pip install librosa madmom")
    sys.exit(1)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def detect_sections_ssm(y: np.ndarray, sr: int, n_sections: int = 6) -> list:
    """
    Detect song sections using self-similarity matrix and spectral clustering.

    Args:
        y: Audio time series
        sr: Sample rate
        n_sections: Approximate number of sections to detect

    Returns:
        List of section dictionaries with start, end, label
    """
    # Extract chroma features (captures harmonic content)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512)

    # Stack time-delay embeddings for richer representation
    chroma_stack = librosa.feature.stack_memory(chroma, n_steps=3, delay=3)

    # Build recurrence matrix (self-similarity)
    rec = librosa.segment.recurrence_matrix(
        chroma_stack,
        mode='affinity',
        sym=True,
        sparse=False,
        bandwidth=1.0
    )

    # Enhance diagonal paths
    rec = librosa.segment.path_enhance(rec, 33)

    # Detect segment boundaries using spectral clustering
    try:
        bounds = librosa.segment.agglomerative(chroma, n_sections)
        bound_times = librosa.frames_to_time(bounds, sr=sr, hop_length=512)
    except Exception:
        # Fallback: evenly spaced sections
        duration = len(y) / sr
        bound_times = np.linspace(0, duration, n_sections + 1)

    # Create section list
    sections = []
    labels = ['intro', 'verse', 'chorus', 'verse', 'chorus', 'outro', 'bridge', 'break']

    for i in range(len(bound_times) - 1):
        sections.append({
            'start': round(float(bound_times[i]), 3),
            'end': round(float(bound_times[i + 1]), 3),
            'label': labels[i % len(labels)]  # Placeholder labels
        })

    return sections


def analyze_audio(audio_path: str, output_path: Optional[str] = None) -> dict:
    """
    Analyze an audio file for BPM, beats, downbeats, and sections.

    Args:
        audio_path: Path to the audio file
        output_path: Optional path to save JSON output

    Returns:
        Dictionary with analysis results
    """
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    print(f"Analyzing: {path.name}")
    print("-" * 50)

    # Load audio with librosa
    print("Loading audio...")
    y, sr = librosa.load(str(path), sr=44100, mono=True)
    duration = len(y) / sr
    print(f"Duration: {duration:.1f}s")

    # Beat detection with madmom (neural network based)
    print("Detecting beats (madmom RNN)...")
    beat_proc = RNNBeatProcessor()
    beat_act = beat_proc(str(path))
    dbn_proc = DBNBeatTrackingProcessor(fps=100)
    beats = dbn_proc(beat_act)

    # Downbeat detection with madmom
    print("Detecting downbeats...")
    try:
        downbeat_proc = RNNDownBeatProcessor()
        downbeat_act = downbeat_proc(str(path))
        dbn_downbeat_proc = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)
        downbeats_raw = dbn_downbeat_proc(downbeat_act)
        # Extract just the downbeat positions (beat_number == 1)
        downbeats = [float(row[0]) for row in downbeats_raw if row[1] == 1]
    except Exception as e:
        print(f"  Downbeat detection failed: {e}")
        # Fallback: assume 4/4, every 4th beat
        downbeats = beats[::4].tolist() if len(beats) > 0 else []

    # Calculate BPM from beat intervals
    if len(beats) > 1:
        beat_intervals = np.diff(beats)
        median_interval = np.median(beat_intervals)
        bpm = 60.0 / median_interval
    else:
        # Fallback to librosa tempo estimation
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(tempo[0]) if hasattr(tempo, '__len__') else float(tempo)

    # Section detection using self-similarity
    print("Detecting sections...")
    sections = detect_sections_ssm(y, sr, n_sections=6)

    # Build analysis result
    analysis = {
        "file": path.name,
        "path": str(path.absolute()),
        "bpm": round(bpm, 2),
        "beats": [round(float(b), 3) for b in beats],
        "downbeats": [round(float(d), 3) for d in downbeats],
        "segments": sections,
        "beat_count": len(beats),
        "downbeat_count": len(downbeats),
        "duration": round(duration, 2),
        "bars": len(downbeats),
        "time_signature": "4/4",  # Assumed
    }

    # Print summary
    print(f"\nResults:")
    print(f"  BPM: {analysis['bpm']}")
    print(f"  Beats: {analysis['beat_count']}")
    print(f"  Bars (downbeats): {analysis['bars']}")
    print(f"\nSections ({len(sections)}):")
    for seg in sections:
        seg_duration = seg["end"] - seg["start"]
        print(f"  [{seg['start']:6.2f}s - {seg['end']:6.2f}s] {seg['label']:8} ({seg_duration:.1f}s)")

    # Save to JSON if output path provided
    if output_path:
        out = Path(output_path)
        with open(out, "w") as f:
            json.dump(analysis, f, indent=2)
        print(f"\nSaved to: {out}")

    return analysis


def main():
    parser = argparse.ArgumentParser(
        description="Analyze audio for BPM, beats, and structure"
    )
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument(
        "-o", "--output",
        help="Output JSON file path (optional)",
        default=None
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON to stdout"
    )

    args = parser.parse_args()

    try:
        result = analyze_audio(args.audio, args.output)

        if args.json:
            print(json.dumps(result, indent=2))

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Analysis failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
