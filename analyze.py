#!/usr/bin/env python3
"""
Audio Analyzer - Beat, tempo, phrase, and structure analysis for DJ/mixing applications.
Uses madmom for beats, allin1 for sections, with phrase detection from downbeats.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np

try:
    import librosa
    from madmom.features.beats import DBNBeatTrackingProcessor, RNNBeatProcessor
    from madmom.features.downbeats import DBNDownBeatTrackingProcessor, RNNDownBeatProcessor
except ImportError as e:
    print(f"Error: Missing dependency - {e}")
    print("Run: pip install librosa madmom")
    sys.exit(1)

# Optional allin1 for ML-based section detection with real labels
try:
    import allin1
    ALLIN1_AVAILABLE = True
except ImportError:
    ALLIN1_AVAILABLE = False

# Optional MSAF as fallback for section detection
try:
    import msaf
    MSAF_AVAILABLE = True
except ImportError:
    MSAF_AVAILABLE = False

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def detect_phrases(downbeats: List[float], beats_per_bar: int = 4,
                   bars_per_phrase: int = 4) -> List[Dict]:
    """
    Detect musical phrases from downbeat positions.

    In most electronic/pop music:
    - 1 bar = 4 beats (4/4 time)
    - 1 phrase = 4 bars (16 beats) or 8 bars (32 beats)
    - Sections often align to phrase boundaries

    Args:
        downbeats: List of downbeat timestamps (bar starts)
        beats_per_bar: Beats per bar (default 4 for 4/4 time)
        bars_per_phrase: Bars per phrase (default 4, can be 8)

    Returns:
        List of phrase dictionaries with start, end, bar_count, phrase_number
    """
    if len(downbeats) < 2:
        return []

    phrases = []
    phrase_num = 1

    # Group downbeats into phrases
    i = 0
    while i < len(downbeats):
        phrase_start = downbeats[i]

        # Calculate end of phrase (bars_per_phrase bars later)
        end_idx = min(i + bars_per_phrase, len(downbeats) - 1)

        # If we have enough bars for a full phrase
        if end_idx > i:
            phrase_end = downbeats[end_idx]
            bar_count = end_idx - i
        else:
            # Last partial phrase - estimate end from bar duration
            if i > 0:
                avg_bar_duration = (downbeats[i] - downbeats[0]) / i
                remaining_bars = len(downbeats) - i
                phrase_end = phrase_start + avg_bar_duration * remaining_bars
                bar_count = remaining_bars
            else:
                break

        phrases.append({
            'start': round(phrase_start, 3),
            'end': round(phrase_end, 3),
            'bar_count': bar_count,
            'phrase_number': phrase_num,
            'is_complete': bar_count == bars_per_phrase
        })

        phrase_num += 1
        i += bars_per_phrase

    return phrases


def detect_energy_sections(y: np.ndarray, sr: int, phrases: List[Dict]) -> List[Dict]:
    """
    Analyze energy levels to identify buildups and drops within phrases.

    Args:
        y: Audio time series
        sr: Sample rate
        phrases: List of phrase dictionaries

    Returns:
        Updated phrases with energy classification (buildup, drop, stable)
    """
    if len(phrases) == 0:
        return phrases

    # Calculate RMS energy over time
    hop_length = 512
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    # Calculate spectral centroid for brightness (higher = more energy/excitement)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]

    # Normalize
    rms_norm = (rms - rms.min()) / (rms.max() - rms.min() + 1e-6)
    sc_norm = (spectral_centroid - spectral_centroid.min()) / (spectral_centroid.max() - spectral_centroid.min() + 1e-6)

    # Combined energy metric
    energy = 0.7 * rms_norm + 0.3 * sc_norm

    for phrase in phrases:
        # Get energy values within this phrase
        mask = (times >= phrase['start']) & (times < phrase['end'])
        phrase_energy = energy[mask]

        if len(phrase_energy) < 2:
            phrase['energy_type'] = 'stable'
            phrase['energy_level'] = 0.5
            continue

        # Calculate energy statistics
        avg_energy = np.mean(phrase_energy)
        energy_start = np.mean(phrase_energy[:len(phrase_energy)//3])
        energy_end = np.mean(phrase_energy[-len(phrase_energy)//3:])
        energy_change = energy_end - energy_start

        phrase['energy_level'] = round(float(avg_energy), 3)

        # Classify based on energy trajectory
        if energy_change > 0.15:
            phrase['energy_type'] = 'buildup'
        elif energy_change < -0.15:
            phrase['energy_type'] = 'drop'
        elif avg_energy > 0.7:
            phrase['energy_type'] = 'high'
        elif avg_energy < 0.3:
            phrase['energy_type'] = 'low'
        else:
            phrase['energy_type'] = 'stable'

    return phrases


# Krumhansl-Schmuckler key profiles (perceptual weightings for pitch classes)
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# Note names (chromagram order: C, C#, D, D#, E, F, F#, G, G#, A, A#, B)
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Camelot wheel mapping for DJ harmonic mixing
CAMELOT_MAJOR = {
    'C': '8B', 'G': '9B', 'D': '10B', 'A': '11B', 'E': '12B', 'B': '1B',
    'F#': '2B', 'C#': '3B', 'G#': '4B', 'D#': '5B', 'A#': '6B', 'F': '7B'
}
CAMELOT_MINOR = {
    'A': '8A', 'E': '9A', 'B': '10A', 'F#': '11A', 'C#': '12A', 'G#': '1A',
    'D#': '2A', 'A#': '3A', 'F': '4A', 'C': '5A', 'G': '6A', 'D': '7A'
}


def detect_key(y: np.ndarray, sr: int) -> Dict:
    """
    Detect musical key using Krumhansl-Schmuckler algorithm.

    Analyzes the chromagram (pitch class distribution) and correlates it
    with major/minor key profiles to find the best matching key.

    Args:
        y: Audio time series
        sr: Sample rate

    Returns:
        Dictionary with key, scale, camelot code, and confidence score
    """
    # Extract chromagram and average across time
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    chroma_mean = chroma_mean / (np.linalg.norm(chroma_mean) + 1e-6)  # Normalize

    best_corr = -1
    best_key = 'C'
    best_scale = 'major'

    # Test all 24 keys (12 major + 12 minor)
    for i in range(12):
        # Rotate profiles to test each key
        major_rotated = np.roll(MAJOR_PROFILE, i)
        minor_rotated = np.roll(MINOR_PROFILE, i)

        # Normalize profiles
        major_norm = major_rotated / np.linalg.norm(major_rotated)
        minor_norm = minor_rotated / np.linalg.norm(minor_rotated)

        # Correlate with chroma
        major_corr = np.corrcoef(chroma_mean, major_norm)[0, 1]
        minor_corr = np.corrcoef(chroma_mean, minor_norm)[0, 1]

        if major_corr > best_corr:
            best_corr = major_corr
            best_key = NOTE_NAMES[i]
            best_scale = 'major'

        if minor_corr > best_corr:
            best_corr = minor_corr
            best_key = NOTE_NAMES[i]
            best_scale = 'minor'

    # Get Camelot code for DJ mixing
    camelot = CAMELOT_MAJOR.get(best_key) if best_scale == 'major' else CAMELOT_MINOR.get(best_key)

    return {
        'key': best_key,
        'scale': best_scale,
        'camelot': camelot,
        'confidence': round(float(best_corr), 3)
    }


def detect_sections_allin1(audio_path: str) -> Tuple[List[Dict], float, List[float], List[float]]:
    """
    Detect song sections using allin1 (All-In-One Music Structure Analyzer).
    Returns sections with real semantic labels (intro, verse, chorus, etc.)

    Args:
        audio_path: Path to the audio file

    Returns:
        Tuple of (sections, bpm, beats, downbeats)
    """
    if not ALLIN1_AVAILABLE:
        raise ImportError("allin1 not available")

    result = allin1.analyze(audio_path)

    sections = []
    for seg in result.segments:
        sections.append({
            'start': round(float(seg.start), 3),
            'end': round(float(seg.end), 3),
            'label': seg.label
        })

    beats = [round(float(b), 3) for b in result.beats]
    downbeats = [round(float(d), 3) for d in result.downbeats]
    bpm = float(result.bpm)

    return sections, bpm, beats, downbeats


def detect_sections_msaf(audio_path: str) -> List[Dict]:
    """
    Detect song sections using MSAF (Music Structure Analysis Framework).
    Uses spectral clustering for boundaries and labels.

    Args:
        audio_path: Path to the audio file

    Returns:
        List of section dictionaries with start, end, label
    """
    if not MSAF_AVAILABLE:
        raise ImportError("MSAF not available")

    boundaries, labels = msaf.process(
        audio_path,
        boundaries_id="scluster",
        labels_id="scluster",
        feature="cqt",
    )

    label_names = ['intro', 'verse', 'chorus', 'bridge', 'outro', 'break', 'inst', 'solo']

    sections = []
    for i in range(len(boundaries) - 1):
        label_idx = labels[i] if i < len(labels) else 0
        sections.append({
            'start': round(float(boundaries[i]), 3),
            'end': round(float(boundaries[i + 1]), 3),
            'label': label_names[label_idx % len(label_names)]
        })

    return sections


def detect_sections_ssm(y: np.ndarray, sr: int, n_sections: int = 6) -> List[Dict]:
    """
    Detect song sections using self-similarity matrix and spectral clustering.
    Fallback method when allin1/MSAF aren't available.

    Args:
        y: Audio time series
        sr: Sample rate
        n_sections: Approximate number of sections to detect

    Returns:
        List of section dictionaries with start, end, label
    """
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512)
    chroma_stack = librosa.feature.stack_memory(chroma, n_steps=3, delay=3)

    rec = librosa.segment.recurrence_matrix(
        chroma_stack, mode='affinity', sym=True, sparse=False, bandwidth=1.0
    )
    rec = librosa.segment.path_enhance(rec, 33)

    try:
        bounds = librosa.segment.agglomerative(chroma, n_sections)
        bound_times = librosa.frames_to_time(bounds, sr=sr, hop_length=512)
    except Exception:
        duration = len(y) / sr
        bound_times = np.linspace(0, duration, n_sections + 1)

    sections = []
    labels = ['intro', 'verse', 'chorus', 'verse', 'chorus', 'outro', 'bridge', 'break']

    for i in range(len(bound_times) - 1):
        sections.append({
            'start': round(float(bound_times[i]), 3),
            'end': round(float(bound_times[i + 1]), 3),
            'label': labels[i % len(labels)]
        })

    return sections


def analyze_audio(audio_path: str, output_path: Optional[str] = None,
                  use_allin1: bool = True) -> dict:
    """
    Analyze an audio file for BPM, beats, downbeats, phrases, and sections.

    Args:
        audio_path: Path to the audio file
        output_path: Optional path to save JSON output
        use_allin1: Whether to try allin1 first (slower but better labels)

    Returns:
        Dictionary with analysis results
    """
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    print(f"Analyzing: {path.name}")
    print("-" * 50)

    # Load audio with librosa (needed for energy analysis and fallbacks)
    print("Loading audio...")
    y, sr = librosa.load(str(path), sr=44100, mono=True)
    duration = len(y) / sr
    print(f"Duration: {duration:.1f}s")

    # Detect musical key
    print("Detecting key...")
    key_data = detect_key(y, sr)

    sections = None
    beats = None
    downbeats = None
    bpm = None
    section_method = None

    # Try allin1 first (provides beats, downbeats, sections, and BPM)
    if use_allin1 and ALLIN1_AVAILABLE:
        print("Analyzing with allin1 (ML model)...")
        try:
            sections, bpm, beats, downbeats = detect_sections_allin1(str(path))
            section_method = "allin1"
            print(f"  allin1: BPM={bpm}, {len(beats)} beats, {len(downbeats)} downbeats, {len(sections)} sections")
        except Exception as e:
            print(f"  allin1 failed: {e}")

    # Fallback to madmom for beats/downbeats if allin1 didn't work
    if beats is None:
        print("Detecting beats (madmom RNN)...")
        beat_proc = RNNBeatProcessor()
        beat_act = beat_proc(str(path))
        dbn_proc = DBNBeatTrackingProcessor(fps=100)
        beats_raw = dbn_proc(beat_act)
        beats = [round(float(b), 3) for b in beats_raw]

    if downbeats is None:
        print("Detecting downbeats (madmom)...")
        try:
            downbeat_proc = RNNDownBeatProcessor()
            downbeat_act = downbeat_proc(str(path))
            dbn_downbeat_proc = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)
            downbeats_raw = dbn_downbeat_proc(downbeat_act)
            downbeats = [round(float(row[0]), 3) for row in downbeats_raw if row[1] == 1]
        except Exception as e:
            print(f"  Downbeat detection failed: {e}")
            downbeats = beats[::4] if len(beats) > 0 else []

    # Calculate BPM if not from allin1
    if bpm is None:
        if len(beats) > 1:
            beat_intervals = np.diff(beats)
            median_interval = np.median(beat_intervals)
            bpm = 60.0 / median_interval
        else:
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            bpm = float(tempo[0]) if hasattr(tempo, '__len__') else float(tempo)

    # Section detection fallbacks
    if sections is None:
        print("Detecting sections...")
        if MSAF_AVAILABLE:
            try:
                sections = detect_sections_msaf(str(path))
                section_method = "msaf"
                print("  Using MSAF (scluster)")
            except Exception as e:
                print(f"  MSAF failed: {e}")

        if sections is None:
            sections = detect_sections_ssm(y, sr, n_sections=6)
            section_method = "ssm"
            print("  Using SSM (librosa)")

    # Detect phrases from downbeats
    print("Detecting phrases...")
    phrases = detect_phrases(downbeats, beats_per_bar=4, bars_per_phrase=4)

    # Add energy analysis to phrases
    phrases = detect_energy_sections(y, sr, phrases)

    # Also detect 8-bar phrases for larger structure
    phrases_8bar = detect_phrases(downbeats, beats_per_bar=4, bars_per_phrase=8)
    phrases_8bar = detect_energy_sections(y, sr, phrases_8bar)

    # Build analysis result
    analysis = {
        "file": path.name,
        "path": str(path.absolute()),
        "duration": round(duration, 2),
        "bpm": round(bpm, 2),
        "time_signature": "4/4",

        # Key data for harmonic mixing
        "key": key_data['key'],
        "key_scale": key_data['scale'],
        "key_camelot": key_data['camelot'],
        "key_confidence": key_data['confidence'],

        # Beat data
        "beats": beats,
        "beat_count": len(beats),
        "downbeats": downbeats,
        "bar_count": len(downbeats),

        # Phrase data (4-bar phrases)
        "phrases": phrases,
        "phrase_count": len(phrases),

        # 8-bar phrases for larger structure
        "phrases_8bar": phrases_8bar,

        # Section data with real labels
        "sections": sections,
        "section_count": len(sections),
        "section_method": section_method,
    }

    # Print summary
    print(f"\n{'='*50}")
    print(f"RESULTS: {path.name}")
    print(f"{'='*50}")
    print(f"  BPM: {analysis['bpm']}")
    print(f"  Key: {analysis['key']} {analysis['key_scale']} ({analysis['key_camelot']})")
    print(f"  Duration: {analysis['duration']}s")
    print(f"  Beats: {analysis['beat_count']}")
    print(f"  Bars: {analysis['bar_count']}")
    print(f"  Phrases (4-bar): {analysis['phrase_count']}")

    print(f"\nSECTIONS ({len(sections)}) [{section_method}]:")
    for seg in sections:
        seg_duration = seg["end"] - seg["start"]
        print(f"  [{seg['start']:6.2f}s - {seg['end']:6.2f}s] {seg['label']:8} ({seg_duration:.1f}s)")

    print(f"\nPHRASES (4-bar):")
    for phrase in phrases[:10]:  # Show first 10
        energy_info = f"[{phrase.get('energy_type', '?'):8}]" if 'energy_type' in phrase else ""
        complete = "✓" if phrase.get('is_complete', False) else "○"
        print(f"  {complete} Phrase {phrase['phrase_number']:2}: [{phrase['start']:6.2f}s - {phrase['end']:6.2f}s] {phrase['bar_count']} bars {energy_info}")
    if len(phrases) > 10:
        print(f"  ... and {len(phrases) - 10} more phrases")

    # Save to JSON if output path provided
    if output_path:
        out = Path(output_path)
        with open(out, "w") as f:
            json.dump(analysis, f, indent=2)
        print(f"\nSaved to: {out}")

    return analysis


def main():
    parser = argparse.ArgumentParser(
        description="Analyze audio for BPM, beats, phrases, and structure"
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
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip allin1 (faster but no semantic section labels)"
    )

    args = parser.parse_args()

    try:
        result = analyze_audio(args.audio, args.output, use_allin1=not args.fast)

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
