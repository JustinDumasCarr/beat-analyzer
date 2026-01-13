#!/usr/bin/env python3
"""
Audio Analyzer - Beat, tempo, phrase, and structure analysis for DJ/mixing applications.
Uses madmom for beats, allin1 for sections, with phrase detection from downbeats.
"""

import argparse
import json
import logging
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Callable

import numpy as np

# Module-level logger
logger = logging.getLogger(__name__)

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

# Optional demucs for stem separation
try:
    import demucs.api
    DEMUCS_AVAILABLE = True
except ImportError:
    DEMUCS_AVAILABLE = False

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Progress stages for callback reporting
PROGRESS_STAGES = {
    'loading': 10,
    'key_detection': 20,
    'beat_detection': 40,
    'section_detection': 60,
    'phrase_detection': 70,
    'intensity': 80,
    'loudness': 85,
    'waveform': 87,
    'vocals': 89,
    'stems': 95,
    'mix_points': 90,
    'complete': 100
}


def _report_progress(callback: Optional[Callable[[int, str], None]], stage: str):
    """Report progress to callback if provided."""
    if callback:
        pct = PROGRESS_STAGES.get(stage, 0)
        callback(pct, stage)

# Supported audio extensions
SUPPORTED_EXTENSIONS = {'.mp3', '.wav', '.flac', '.m4a', '.ogg', '.aac', '.wma'}


def setup_logging(log_file: Optional[str] = None, verbose: bool = False):
    """
    Configure logging with optional file output.

    Args:
        log_file: Path to log file (optional)
        verbose: Enable debug level logging
    """
    level = logging.DEBUG if verbose else logging.INFO
    format_str = '%(asctime)s [%(levelname)s] %(message)s'
    datefmt = '%Y-%m-%d %H:%M:%S'

    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format=format_str,
        datefmt=datefmt,
        handlers=handlers,
        force=True  # Override any existing config
    )


def validate_audio_file(audio_path: str, min_duration: float = 30.0) -> Dict:
    """
    Validate audio file before analysis.

    Args:
        audio_path: Path to audio file
        min_duration: Minimum duration in seconds

    Returns:
        {"valid": True} or {"valid": False, "reason": "..."}
    """
    path = Path(audio_path)

    # Check file exists
    if not path.exists():
        return {"valid": False, "reason": f"File not found: {audio_path}"}

    # Check extension
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return {"valid": False, "reason": f"Unsupported format: {path.suffix}"}

    # Try to load and check duration
    try:
        # Use librosa to get duration without loading full audio
        duration = librosa.get_duration(path=str(path))
        if duration < min_duration:
            return {"valid": False, "reason": f"Too short: {duration:.1f}s < {min_duration}s minimum"}
    except Exception as e:
        return {"valid": False, "reason": f"Cannot decode: {e}"}

    return {"valid": True, "duration": duration}


def save_checkpoint(checkpoint_path: str, processed: List[str],
                    results: List[Dict], errors: List[Dict], directory: str):
    """
    Save batch progress to checkpoint file.

    Args:
        checkpoint_path: Path to checkpoint file
        processed: List of processed file paths
        results: List of successful results
        errors: List of error records
        directory: Original batch directory
    """
    checkpoint = {
        "timestamp": datetime.now().isoformat(),
        "directory": directory,
        "processed_files": processed,
        "results": results,
        "errors": errors
    }
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    logger.debug(f"Checkpoint saved: {len(processed)} files processed")


def load_checkpoint(checkpoint_path: str) -> Dict:
    """
    Load checkpoint and return checkpoint data.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Checkpoint dictionary with processed_files, results, errors
    """
    with open(checkpoint_path) as f:
        checkpoint = json.load(f)
    logger.info(f"Loaded checkpoint: {len(checkpoint.get('processed_files', []))} files already processed")
    return checkpoint


def analyze_with_retry(audio_path: str, max_retries: int = 3,
                       backoff: float = 1.0, **kwargs) -> Dict:
    """
    Analyze with exponential backoff retry on failure.

    Args:
        audio_path: Path to audio file
        max_retries: Maximum retry attempts (0 = no retry)
        backoff: Initial backoff in seconds (doubles each retry)
        **kwargs: Arguments to pass to analyze_audio

    Returns:
        Analysis result dictionary

    Raises:
        Last exception if all retries fail
    """
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return analyze_audio(audio_path, **kwargs)
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                wait = backoff * (2 ** attempt)
                logger.warning(f"Attempt {attempt + 1}/{max_retries + 1} failed for {Path(audio_path).name}: {e}")
                logger.info(f"Retrying in {wait:.1f}s...")
                time.sleep(wait)
            else:
                logger.error(f"All {max_retries + 1} attempts failed for {Path(audio_path).name}")

    raise last_error


def analyze_with_timeout(audio_path: str, timeout: int = 300, **kwargs) -> Dict:
    """
    Analyze with timeout protection.

    Args:
        audio_path: Path to audio file
        timeout: Maximum seconds per file (default 5 minutes)
        **kwargs: Arguments to pass to analyze_audio

    Returns:
        Analysis result dictionary

    Raises:
        TimeoutError if analysis exceeds timeout
    """
    result_queue = Queue()

    def worker():
        try:
            result = analyze_audio(audio_path, **kwargs)
            result_queue.put({"success": True, "result": result})
        except Exception as e:
            result_queue.put({"success": False, "error": str(e)})

    p = Process(target=worker)
    p.start()
    p.join(timeout=timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        logger.error(f"Analysis timed out after {timeout}s: {Path(audio_path).name}")
        raise TimeoutError(f"Analysis timed out after {timeout}s")

    if result_queue.empty():
        raise RuntimeError("Worker process ended without result")

    outcome = result_queue.get()
    if outcome["success"]:
        return outcome["result"]
    raise Exception(outcome["error"])


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


def detect_mix_points(sections: List[Dict], phrases: List[Dict],
                      downbeats: List[float], duration: float) -> Dict:
    """
    Find optimal mix in/out points based on sections and phrases.

    Args:
        sections: List of section dictionaries with start, end, label
        phrases: List of phrase dictionaries with start, end, energy_type
        downbeats: List of downbeat timestamps
        duration: Total track duration

    Returns:
        Dictionary with mix_in, mix_out, drop_start, buildup_start timestamps
    """
    mix_points = {
        'mix_in': None,
        'mix_out': None,
        'drop_start': None,
        'buildup_start': None
    }

    # Find intro end / mix_in point (first downbeat after intro)
    intro_end = 0.0
    for section in sections:
        if section['label'].lower() in ['intro']:
            intro_end = section['end']
            break

    # Find first downbeat at or after intro end
    for db in downbeats:
        if db >= intro_end:
            mix_points['mix_in'] = round(db, 3)
            break

    # If no intro found, use first downbeat or first phrase start
    if mix_points['mix_in'] is None:
        if downbeats:
            mix_points['mix_in'] = round(downbeats[0], 3)
        elif phrases:
            mix_points['mix_in'] = phrases[0]['start']

    # Find outro start / mix_out point (last phrase start before outro)
    outro_start = duration
    for section in sections:
        if section['label'].lower() in ['outro']:
            outro_start = section['start']
            break

    # Find last phrase that starts before outro
    for phrase in reversed(phrases):
        if phrase['start'] < outro_start - 1.0:  # Give 1s buffer
            mix_points['mix_out'] = phrase['start']
            break

    # If no outro found, use last phrase start (before final phrase)
    if mix_points['mix_out'] is None and len(phrases) >= 2:
        mix_points['mix_out'] = phrases[-2]['start']

    # Find drop_start (first high-energy section: drop, chorus, or high-energy phrase)
    drop_labels = ['drop', 'chorus', 'hook']
    for section in sections:
        if section['label'].lower() in drop_labels:
            mix_points['drop_start'] = round(section['start'], 3)
            break

    # Fallback: find first 'high' or 'drop' energy phrase
    if mix_points['drop_start'] is None:
        for phrase in phrases:
            if phrase.get('energy_type') in ['high', 'drop']:
                mix_points['drop_start'] = phrase['start']
                break

    # Find buildup_start (phrase before drop with 'buildup' energy type)
    if mix_points['drop_start']:
        for i, phrase in enumerate(phrases):
            if phrase['start'] >= mix_points['drop_start']:
                # Look at previous phrase(s) for buildup
                for j in range(max(0, i-2), i):
                    if phrases[j].get('energy_type') == 'buildup':
                        mix_points['buildup_start'] = phrases[j]['start']
                        break
                break

    # Fallback: find any buildup phrase
    if mix_points['buildup_start'] is None:
        for phrase in phrases:
            if phrase.get('energy_type') == 'buildup':
                mix_points['buildup_start'] = phrase['start']
                break

    return mix_points


def calculate_intensity(y: np.ndarray, sr: int, bpm: float) -> Dict:
    """
    Calculate overall song intensity on 0-100 scale.

    Combines multiple factors:
    - RMS energy (loudness)
    - Tempo (higher BPM = higher intensity)
    - Spectral centroid (brightness/excitement)
    - Dynamic range (variance in energy)

    Args:
        y: Audio time series
        sr: Sample rate
        bpm: Beats per minute

    Returns:
        Dictionary with intensity_score (0-100) and intensity_class
    """
    hop_length = 512

    # Calculate RMS energy
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms_mean = np.mean(rms)
    rms_max = np.max(rms)

    # Calculate spectral centroid (brightness)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    sc_mean = np.mean(spectral_centroid)

    # Normalize each component to 0-1 range
    # RMS: typical range 0.01-0.3 for music
    rms_score = np.clip(rms_mean / 0.2, 0, 1)

    # Spectral centroid: typical range 1000-5000 Hz for music
    sc_score = np.clip((sc_mean - 1000) / 4000, 0, 1)

    # BPM: map 60-180 BPM to 0-1
    bpm_score = np.clip((bpm - 60) / 120, 0, 1)

    # Dynamic range: higher variance = more dynamic = more intensity
    rms_std = np.std(rms)
    dynamic_score = np.clip(rms_std / 0.1, 0, 1)

    # Combine scores with weights
    # RMS (loudness) is most important, then tempo, then brightness
    intensity = (
        0.35 * rms_score +
        0.25 * bpm_score +
        0.25 * sc_score +
        0.15 * dynamic_score
    )

    # Scale to 0-100
    intensity_score = int(round(intensity * 100))
    intensity_score = max(0, min(100, intensity_score))

    # Classify into categories
    if intensity_score <= 30:
        intensity_class = 'warmup'
    elif intensity_score <= 60:
        intensity_class = 'moderate'
    elif intensity_score <= 85:
        intensity_class = 'peak'
    else:
        intensity_class = 'extreme'

    return {
        'intensity_score': intensity_score,
        'intensity_class': intensity_class
    }


def generate_tags(bpm: float, key_data: Dict, intensity_data: Dict,
                  phrases: List[Dict], sections: List[Dict]) -> Dict:
    """
    Generate searchable tags from analysis data.

    Args:
        bpm: Beats per minute
        key_data: Dictionary with key, scale, camelot
        intensity_data: Dictionary with intensity_score, intensity_class
        phrases: List of phrase dictionaries with energy_type
        sections: List of section dictionaries with label

    Returns:
        Dictionary of tags for filtering/searching
    """
    # BPM range classification
    if bpm < 100:
        bpm_range = 'slow'
    elif bpm <= 130:
        bpm_range = 'medium'
    else:
        bpm_range = 'fast'

    # Energy classification from intensity
    score = intensity_data['intensity_score']
    if score < 30:
        energy = 'low'
    elif score < 55:
        energy = 'medium'
    elif score < 80:
        energy = 'high'
    else:
        energy = 'extreme'

    # Extract key family (Camelot number for harmonic grouping)
    camelot = key_data.get('camelot', '')
    key_family = ''.join(c for c in camelot if c.isdigit()) if camelot else ''

    # Check for buildup and drop
    energy_types = [p.get('energy_type', '') for p in phrases]
    section_labels = [s.get('label', '').lower() for s in sections]

    has_buildup = 'buildup' in energy_types
    has_drop = 'drop' in section_labels or 'drop' in energy_types or 'chorus' in section_labels

    return {
        'bpm_range': bpm_range,
        'energy': energy,
        'key_family': key_family,
        'has_buildup': has_buildup,
        'has_drop': has_drop
    }


def analyze_loudness(y: np.ndarray, sr: int) -> Dict:
    """
    Measure loudness for normalization and mixing.

    Uses pyloudnorm for EBU R128 LUFS measurement.

    Args:
        y: Audio time series
        sr: Sample rate

    Returns:
        Dictionary with LUFS, peak dBFS, and gain suggestion
    """
    try:
        import pyloudnorm as pyln
    except ImportError:
        return {
            'lufs': None,
            'peak_dbfs': None,
            'gain_suggestion': None,
            'error': 'pyloudnorm not installed'
        }

    # Create meter with sample rate
    meter = pyln.Meter(sr)

    # Measure integrated loudness (LUFS)
    try:
        lufs = meter.integrated_loudness(y)
    except Exception:
        # Handle very quiet or silent audio
        lufs = -70.0

    # Calculate peak level in dBFS
    peak_linear = np.max(np.abs(y))
    if peak_linear > 0:
        peak_dbfs = 20 * np.log10(peak_linear)
    else:
        peak_dbfs = -70.0

    # Suggest gain adjustment to reach -14 LUFS (streaming standard)
    target_lufs = -14.0
    if lufs > -70:
        gain_suggestion = target_lufs - lufs
    else:
        gain_suggestion = 0.0

    return {
        'lufs': round(float(lufs), 2),
        'peak_dbfs': round(float(peak_dbfs), 2),
        'gain_suggestion': round(float(gain_suggestion), 2)
    }


def create_trimmed_versions(audio_path: str, analysis: dict, output_dir: str) -> List[Dict]:
    """
    Create trimmed versions of audio file based on analysis.

    Creates three versions:
    - {name}_mix: Starts at mix_in point (after intro)
    - {name}_drop: Starts at buildup/drop point
    - {name}_body: Only the body (mix_in to mix_out, no intro/outro)

    Args:
        audio_path: Path to original audio file
        analysis: Analysis dictionary with mix_points
        output_dir: Directory to save trimmed files

    Returns:
        List of dictionaries describing created versions
    """
    try:
        from pydub import AudioSegment
    except ImportError:
        logger.warning("pydub not installed, skipping trim")
        return []

    path = Path(audio_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Get mix points
    mix_points = analysis.get('mix_points', {})
    mix_in = mix_points.get('mix_in')
    mix_out = mix_points.get('mix_out')
    drop_start = mix_points.get('drop_start')
    buildup_start = mix_points.get('buildup_start')
    duration = analysis.get('duration', 0)

    # Load audio
    logger.debug("Loading audio for trimming...")
    try:
        audio = AudioSegment.from_file(str(path))
    except Exception as e:
        logger.error(f"Error loading audio for trim: {e}")
        return []

    # Determine output format from input extension
    input_ext = path.suffix.lower().lstrip('.')
    # Use same format as input, or wav as fallback
    output_format = input_ext if input_ext in ['mp3', 'wav', 'flac', 'ogg', 'm4a'] else 'wav'

    versions = []
    stem = path.stem

    # Version 1: _mix - starts at mix_in (after intro)
    if mix_in is not None and mix_in > 0.5:  # Only create if there's an intro to cut
        start_ms = int(mix_in * 1000)
        trimmed = audio[start_ms:]
        out_file = out_dir / f"{stem}_mix.{output_format}"
        try:
            trimmed.export(str(out_file), format=output_format)
            versions.append({
                'type': 'mix',
                'start': round(mix_in, 3),
                'end': round(duration, 3),
                'file': str(out_file.name),
                'path': str(out_file.absolute())
            })
            logger.info(f"  Created: {out_file.name} (from {mix_in:.1f}s)")
        except Exception as e:
            logger.error(f"  Error creating _mix version: {e}")

    # Version 2: _drop - starts at buildup or drop
    drop_point = buildup_start or drop_start
    if drop_point is not None and drop_point > 1.0:
        start_ms = int(drop_point * 1000)
        trimmed = audio[start_ms:]
        out_file = out_dir / f"{stem}_drop.{output_format}"
        try:
            trimmed.export(str(out_file), format=output_format)
            versions.append({
                'type': 'drop',
                'start': round(drop_point, 3),
                'end': round(duration, 3),
                'file': str(out_file.name),
                'path': str(out_file.absolute())
            })
            logger.info(f"  Created: {out_file.name} (from {drop_point:.1f}s)")
        except Exception as e:
            logger.error(f"  Error creating _drop version: {e}")

    # Version 3: _body - mix_in to mix_out (no intro/outro)
    if mix_in is not None and mix_out is not None and mix_out > mix_in:
        start_ms = int(mix_in * 1000)
        end_ms = int(mix_out * 1000)
        trimmed = audio[start_ms:end_ms]
        out_file = out_dir / f"{stem}_body.{output_format}"
        try:
            trimmed.export(str(out_file), format=output_format)
            versions.append({
                'type': 'body',
                'start': round(mix_in, 3),
                'end': round(mix_out, 3),
                'file': str(out_file.name),
                'path': str(out_file.absolute())
            })
            logger.info(f"  Created: {out_file.name} ({mix_in:.1f}s - {mix_out:.1f}s)")
        except Exception as e:
            logger.error(f"  Error creating _body version: {e}")

    return versions


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


def extract_waveform_data(y: np.ndarray, sr: int, num_points: int = 1000) -> Dict:
    """
    Extract waveform data for visualization.

    Args:
        y: Audio time series
        sr: Sample rate
        num_points: Number of points for visualization (default 1000)

    Returns:
        Dictionary with peaks, minmax pairs, sample_rate, duration
    """
    duration = len(y) / sr
    samples_per_point = max(1, len(y) // num_points)

    peaks = []
    minmax = []

    for i in range(num_points):
        start = i * samples_per_point
        end = min(start + samples_per_point, len(y))
        segment = y[start:end]

        if len(segment) > 0:
            # Peak (absolute max for simple waveform)
            peak = float(np.max(np.abs(segment)))
            peaks.append(round(peak, 4))

            # Min/max for detailed rendering
            seg_min = float(np.min(segment))
            seg_max = float(np.max(segment))
            minmax.append([round(seg_min, 4), round(seg_max, 4)])
        else:
            peaks.append(0.0)
            minmax.append([0.0, 0.0])

    return {
        'peaks': peaks,
        'minmax': minmax,
        'sample_rate': sr,
        'duration': round(duration, 3),
        'points': num_points
    }


def detect_vocal_presence(y: np.ndarray, sr: int, threshold: float = 0.5) -> List[Dict]:
    """
    Detect presence of vocals using spectral analysis.

    Uses harmonic-percussive separation and pitch tracking to identify
    vocal sections without requiring full stem separation.

    Args:
        y: Audio time series
        sr: Sample rate
        threshold: Detection threshold (0-1)

    Returns:
        List of {start, end, confidence} for vocal sections
    """
    hop_length = 512

    try:
        # Harmonic-percussive separation
        y_harmonic, _ = librosa.effects.hpss(y)

        # Spectral flatness (lower = more harmonic/tonal = more likely vocals)
        flatness = librosa.feature.spectral_flatness(y=y_harmonic, hop_length=hop_length)[0]

        # Fundamental frequency tracking (vocals typically 80-400 Hz)
        f0, voiced_flag, voiced_prob = librosa.pyin(
            y_harmonic, fmin=80, fmax=400, sr=sr, hop_length=hop_length
        )

        # Normalize flatness (invert so higher = more likely vocal)
        flatness_norm = 1 - np.clip(flatness / 0.1, 0, 1)

        # Clean up voiced_prob
        voiced_norm = np.nan_to_num(voiced_prob, nan=0.0)

        # Ensure arrays are same length
        min_len = min(len(flatness_norm), len(voiced_norm))
        flatness_norm = flatness_norm[:min_len]
        voiced_norm = voiced_norm[:min_len]

        # Weighted combination
        vocal_score = 0.4 * flatness_norm + 0.6 * voiced_norm

        # Smooth with moving average (1 second window)
        kernel_size = max(1, int(sr / hop_length))
        vocal_score_smooth = np.convolve(
            vocal_score,
            np.ones(kernel_size) / kernel_size,
            mode='same'
        )

        # Convert to time
        times = librosa.frames_to_time(
            np.arange(len(vocal_score_smooth)),
            sr=sr,
            hop_length=hop_length
        )

        # Find regions above threshold
        vocal_regions = []
        in_vocal = False
        region_start = 0
        region_scores = []

        for i, (t, score) in enumerate(zip(times, vocal_score_smooth)):
            if score >= threshold and not in_vocal:
                in_vocal = True
                region_start = t
                region_scores = [score]
            elif score >= threshold and in_vocal:
                region_scores.append(score)
            elif score < threshold and in_vocal:
                in_vocal = False
                if t - region_start >= 0.5:  # Minimum 0.5s vocal region
                    vocal_regions.append({
                        'start': round(float(region_start), 3),
                        'end': round(float(t), 3),
                        'confidence': round(float(np.mean(region_scores)), 3)
                    })

        # Handle region that extends to end
        if in_vocal and len(times) > 0 and times[-1] - region_start >= 0.5:
            vocal_regions.append({
                'start': round(float(region_start), 3),
                'end': round(float(times[-1]), 3),
                'confidence': round(float(np.mean(region_scores)), 3)
            })

        return vocal_regions

    except Exception as e:
        logger.warning(f"Vocal detection failed: {e}")
        return []


def separate_stems(audio_path: str, output_dir: str, model: str = "htdemucs") -> Optional[Dict]:
    """
    Separate audio into stems using Demucs.

    Args:
        audio_path: Path to audio file
        output_dir: Directory to save stems
        model: Demucs model to use (default: htdemucs)

    Returns:
        Dictionary with stem paths: {vocals, drums, bass, other}
        or None if demucs not available
    """
    if not DEMUCS_AVAILABLE:
        logger.warning("Demucs not available, skipping stem separation")
        return None

    try:
        separator = demucs.api.Separator(model=model)
        origin, separated = separator.separate_audio_file(audio_path)

        # Save stems
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        stem_name = Path(audio_path).stem
        stems = {}

        for stem, source in separated.items():
            stem_file = out_path / f"{stem_name}_{stem}.wav"
            demucs.api.save_audio(
                source,
                str(stem_file),
                samplerate=separator.samplerate
            )
            stems[stem] = str(stem_file.absolute())

        return stems

    except Exception as e:
        logger.error(f"Stem separation failed: {e}")
        return None


def analyze_audio(audio_path: str, output_path: Optional[str] = None,
                  use_allin1: bool = True, trim: bool = False,
                  trim_output_dir: Optional[str] = None,
                  extract_waveform: bool = False,
                  detect_vocals: bool = False,
                  separate_stems_flag: bool = False,
                  stems_output_dir: Optional[str] = None,
                  progress_callback: Optional[Callable[[int, str], None]] = None) -> dict:
    """
    Analyze an audio file for BPM, beats, downbeats, phrases, and sections.

    Args:
        audio_path: Path to the audio file
        output_path: Optional path to save JSON output
        use_allin1: Whether to try allin1 first (slower but better labels)
        trim: Whether to create trimmed audio versions
        trim_output_dir: Directory for trimmed files (default: same as input)
        extract_waveform: Extract waveform data for visualization
        detect_vocals: Detect vocal presence in audio
        separate_stems_flag: Separate into stems (vocals, drums, bass, other)
        stems_output_dir: Directory for stem files
        progress_callback: Callback function(pct, stage) for progress reporting

    Returns:
        Dictionary with analysis results
    """
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    logger.info(f"Analyzing: {path.name}")
    logger.debug("-" * 50)

    # Report loading progress
    _report_progress(progress_callback, 'loading')

    # Load audio with librosa (needed for energy analysis and fallbacks)
    logger.debug("Loading audio...")
    y, sr = librosa.load(str(path), sr=44100, mono=True)
    duration = len(y) / sr
    logger.debug(f"Duration: {duration:.1f}s")

    # Detect musical key
    _report_progress(progress_callback, 'key_detection')
    logger.debug("Detecting key...")
    key_data = detect_key(y, sr)

    sections = None
    beats = None
    downbeats = None
    bpm = None
    section_method = None

    # Try allin1 first (provides beats, downbeats, sections, and BPM)
    _report_progress(progress_callback, 'beat_detection')
    if use_allin1 and ALLIN1_AVAILABLE:
        logger.debug("Analyzing with allin1 (ML model)...")
        try:
            sections, bpm, beats, downbeats = detect_sections_allin1(str(path))
            section_method = "allin1"
            logger.debug(f"  allin1: BPM={bpm}, {len(beats)} beats, {len(downbeats)} downbeats, {len(sections)} sections")
        except Exception as e:
            logger.warning(f"  allin1 failed: {e}")

    # Fallback to madmom for beats/downbeats if allin1 didn't work
    if beats is None:
        logger.debug("Detecting beats (madmom RNN)...")
        beat_proc = RNNBeatProcessor()
        beat_act = beat_proc(str(path))
        dbn_proc = DBNBeatTrackingProcessor(fps=100)
        beats_raw = dbn_proc(beat_act)
        beats = [round(float(b), 3) for b in beats_raw]

    if downbeats is None:
        logger.debug("Detecting downbeats (madmom)...")
        try:
            downbeat_proc = RNNDownBeatProcessor()
            downbeat_act = downbeat_proc(str(path))
            dbn_downbeat_proc = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)
            downbeats_raw = dbn_downbeat_proc(downbeat_act)
            downbeats = [round(float(row[0]), 3) for row in downbeats_raw if row[1] == 1]
        except Exception as e:
            logger.warning(f"  Downbeat detection failed: {e}")
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
    _report_progress(progress_callback, 'section_detection')
    if sections is None:
        logger.debug("Detecting sections...")
        if MSAF_AVAILABLE:
            try:
                sections = detect_sections_msaf(str(path))
                section_method = "msaf"
                logger.debug("  Using MSAF (scluster)")
            except Exception as e:
                logger.warning(f"  MSAF failed: {e}")

        if sections is None:
            sections = detect_sections_ssm(y, sr, n_sections=6)
            section_method = "ssm"
            logger.debug("  Using SSM (librosa)")

    # Detect phrases from downbeats
    _report_progress(progress_callback, 'phrase_detection')
    logger.debug("Detecting phrases...")
    phrases = detect_phrases(downbeats, beats_per_bar=4, bars_per_phrase=4)

    # Add energy analysis to phrases
    phrases = detect_energy_sections(y, sr, phrases)

    # Also detect 8-bar phrases for larger structure
    phrases_8bar = detect_phrases(downbeats, beats_per_bar=4, bars_per_phrase=8)
    phrases_8bar = detect_energy_sections(y, sr, phrases_8bar)

    # Calculate intensity score
    _report_progress(progress_callback, 'intensity')
    logger.debug("Calculating intensity...")
    intensity_data = calculate_intensity(y, sr, bpm)

    # Analyze loudness (LUFS)
    _report_progress(progress_callback, 'loudness')
    logger.debug("Analyzing loudness...")
    loudness_data = analyze_loudness(y, sr)

    # Extract waveform data if requested
    waveform_data = None
    if extract_waveform:
        _report_progress(progress_callback, 'waveform')
        logger.debug("Extracting waveform data...")
        waveform_data = extract_waveform_data(y, sr, num_points=1000)

    # Detect vocals if requested
    vocal_presence = None
    if detect_vocals:
        _report_progress(progress_callback, 'vocals')
        logger.debug("Detecting vocal presence...")
        vocal_presence = detect_vocal_presence(y, sr)

    # Separate stems if requested
    stems_result = None
    if separate_stems_flag:
        _report_progress(progress_callback, 'stems')
        logger.debug("Separating stems (this may take a while)...")
        out_dir = stems_output_dir or str(path.parent / "stems")
        stems_result = separate_stems(str(path), out_dir)

    # Detect mix points
    _report_progress(progress_callback, 'mix_points')
    logger.debug("Finding mix points...")
    mix_points = detect_mix_points(sections, phrases, downbeats, duration)

    # Generate tags
    tags = generate_tags(bpm, key_data, intensity_data, phrases, sections)

    # Generate trim points for audio export
    trim_points = {
        'intro_end': mix_points.get('mix_in'),
        'body_start': mix_points.get('mix_in'),
        'body_end': mix_points.get('mix_out'),
        'drop_start': mix_points.get('buildup_start') or mix_points.get('drop_start'),
        'outro_start': mix_points.get('mix_out')
    }

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

        # Intensity analysis
        "intensity_score": intensity_data['intensity_score'],
        "intensity_class": intensity_data['intensity_class'],

        # Loudness analysis
        "loudness_lufs": loudness_data.get('lufs'),
        "loudness_peak": loudness_data.get('peak_dbfs'),
        "loudness_gain_suggestion": loudness_data.get('gain_suggestion'),

        # Mix points for DJ transitions
        "mix_points": mix_points,

        # Searchable tags
        "tags": tags,

        # Trim points for audio export
        "trim_points": trim_points,

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

    # Add optional waveform data
    if waveform_data:
        analysis["waveform"] = waveform_data

    # Add optional vocal presence data
    if vocal_presence is not None:
        analysis["vocal_presence"] = vocal_presence

    # Add optional stems data
    if stems_result:
        analysis["stems"] = stems_result

    # Report complete
    _report_progress(progress_callback, 'complete')

    # Print summary
    print(f"\n{'='*50}")
    print(f"RESULTS: {path.name}")
    print(f"{'='*50}")
    print(f"  BPM: {analysis['bpm']}")
    print(f"  Key: {analysis['key']} {analysis['key_scale']} ({analysis['key_camelot']})")
    print(f"  Duration: {analysis['duration']}s")
    print(f"  Intensity: {analysis['intensity_score']}/100 ({analysis['intensity_class']})")
    print(f"  Loudness: {analysis['loudness_lufs']} LUFS (gain: {analysis['loudness_gain_suggestion']:+.1f}dB)")
    print(f"  Beats: {analysis['beat_count']}")
    print(f"  Bars: {analysis['bar_count']}")
    print(f"  Phrases (4-bar): {analysis['phrase_count']}")

    print(f"\nMIX POINTS:")
    print(f"  Mix In:   {mix_points['mix_in']:>7.2f}s" if mix_points['mix_in'] else "  Mix In:   N/A")
    print(f"  Mix Out:  {mix_points['mix_out']:>7.2f}s" if mix_points['mix_out'] else "  Mix Out:  N/A")
    print(f"  Drop:     {mix_points['drop_start']:>7.2f}s" if mix_points['drop_start'] else "  Drop:     N/A")
    print(f"  Buildup:  {mix_points['buildup_start']:>7.2f}s" if mix_points['buildup_start'] else "  Buildup:  N/A")

    print(f"\nTAGS:")
    print(f"  BPM: {tags['bpm_range']}, Energy: {tags['energy']}, Key Family: {tags['key_family']}")
    print(f"  Has Buildup: {tags['has_buildup']}, Has Drop: {tags['has_drop']}")

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

    # Create trimmed versions if requested
    if trim:
        print(f"\nCREATING TRIMMED VERSIONS:")
        out_dir = trim_output_dir or str(path.parent)
        versions = create_trimmed_versions(str(path), analysis, out_dir)
        analysis['versions'] = versions
        if versions:
            print(f"  Created {len(versions)} trimmed version(s)")
        else:
            print("  No trimmed versions created (mix points may be too early)")

    # Save to JSON if output path provided
    if output_path:
        out = Path(output_path)
        with open(out, "w") as f:
            json.dump(analysis, f, indent=2)
        print(f"\nSaved to: {out}")

    return analysis


def _analyze_single_file(args: tuple) -> dict:
    """Worker function for parallel batch processing."""
    file_path, use_allin1 = args
    try:
        result = analyze_audio(str(file_path), output_path=None, use_allin1=use_allin1)
        return {"success": True, "result": result}
    except Exception as e:
        return {
            "success": False,
            "error": {
                "file": Path(file_path).name,
                "path": str(file_path),
                "error": str(e)
            }
        }


def batch_analyze(directory: str, extensions: List[str], recursive: bool = False,
                  use_allin1: bool = True, workers: int = 1,
                  min_duration: float = 30.0, checkpoint_path: Optional[str] = None,
                  resume_path: Optional[str] = None, retry: int = 0,
                  timeout: Optional[int] = None) -> dict:
    """
    Analyze all audio files in a directory.

    Args:
        directory: Path to directory containing audio files
        extensions: List of file extensions to include
        recursive: Whether to search subdirectories
        use_allin1: Whether to use allin1 for analysis
        workers: Number of parallel workers (1 = sequential)
        min_duration: Minimum file duration in seconds (default 30)
        checkpoint_path: Path to save checkpoint file (optional)
        resume_path: Path to resume from checkpoint (optional)
        retry: Number of retry attempts on failure (default 0)
        timeout: Per-file timeout in seconds (optional)

    Returns:
        Dictionary with batch metadata and results array
    """
    start_time = time.time()
    dir_path = Path(directory)

    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    # Find all matching files
    files = []
    for ext in extensions:
        pattern = f"**/*.{ext}" if recursive else f"*.{ext}"
        files.extend(dir_path.glob(pattern))

    files = sorted(set(files))  # Remove duplicates, sort

    if not files:
        raise ValueError(f"No audio files found in {directory}")

    # Load checkpoint if resuming
    processed_files = set()
    results = []
    errors = []

    if resume_path:
        try:
            checkpoint = load_checkpoint(resume_path)
            processed_files = set(checkpoint.get('processed_files', []))
            results = checkpoint.get('results', [])
            errors = checkpoint.get('errors', [])
            logger.info(f"Resuming from checkpoint: {len(processed_files)} files already done")
        except FileNotFoundError:
            logger.warning(f"Checkpoint not found: {resume_path}, starting fresh")

    # Filter out already processed files
    files_to_process = [f for f in files if str(f.absolute()) not in processed_files]

    logger.info(f"Found {len(files)} audio files total")
    if processed_files:
        logger.info(f"Skipping {len(files) - len(files_to_process)} already processed")
    logger.info(f"Processing {len(files_to_process)} files with {workers} worker(s)")
    print("=" * 50)

    skipped = []
    processed_paths = list(processed_files)

    # Sequential processing (with validation, retry, timeout support)
    for i, file_path in enumerate(files_to_process, 1):
        file_str = str(file_path.absolute())
        logger.info(f"[{i}/{len(files_to_process)}] {file_path.name}")

        # Validate before processing
        validation = validate_audio_file(str(file_path), min_duration)
        if not validation.get('valid'):
            reason = validation.get('reason', 'Unknown')
            logger.warning(f"  SKIPPED: {reason}")
            skipped.append({"file": file_path.name, "path": file_str, "reason": reason})
            processed_paths.append(file_str)
            # Save checkpoint for skipped files too
            if checkpoint_path:
                save_checkpoint(checkpoint_path, processed_paths, results, errors, str(dir_path))
            continue

        # Analyze with retry/timeout support
        try:
            if timeout:
                result = analyze_with_timeout(
                    str(file_path), timeout=timeout,
                    output_path=None, use_allin1=use_allin1
                )
            elif retry > 0:
                result = analyze_with_retry(
                    str(file_path), max_retries=retry,
                    output_path=None, use_allin1=use_allin1
                )
            else:
                result = analyze_audio(str(file_path), output_path=None, use_allin1=use_allin1)

            results.append(result)
            logger.info(f"  SUCCESS: BPM={result.get('bpm')}, Key={result.get('key_camelot')}")

        except Exception as e:
            error_record = {
                "file": file_path.name,
                "path": file_str,
                "error": str(e)
            }
            errors.append(error_record)
            logger.error(f"  FAILED: {e}")

        processed_paths.append(file_str)

        # Save checkpoint after each file if enabled
        if checkpoint_path:
            save_checkpoint(checkpoint_path, processed_paths, results, errors, str(dir_path))

    elapsed = time.time() - start_time

    return {
        "batch": {
            "directory": str(dir_path.absolute()),
            "total_files": len(files),
            "processed": len(files_to_process),
            "successful": len(results),
            "failed": len(errors),
            "skipped": len(skipped),
            "workers": workers,
            "processing_time_seconds": round(elapsed, 2)
        },
        "results": results,
        "errors": errors if errors else None,
        "skipped": skipped if skipped else None
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze audio for BPM, beats, phrases, and structure"
    )
    parser.add_argument("audio", nargs="?", help="Path to audio file (required unless --batch)")
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
    parser.add_argument(
        "--batch",
        metavar="DIR",
        help="Batch mode: analyze all audio files in directory"
    )
    parser.add_argument(
        "--ext",
        default="mp3,wav,flac,m4a,ogg",
        help="File extensions to include (comma-separated, default: mp3,wav,flac,m4a,ogg)"
    )
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Recursively search subdirectories"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1 = sequential)"
    )
    parser.add_argument(
        "--trim",
        action="store_true",
        help="Create trimmed audio versions (_mix, _drop, _body)"
    )
    parser.add_argument(
        "--trim-output", "--trim-dir",
        metavar="DIR",
        help="Output directory for trimmed files (default: same as input)"
    )

    # Phase 3: Reliability arguments
    parser.add_argument(
        "--min-duration",
        type=float,
        default=30.0,
        metavar="SECS",
        help="Minimum file duration in seconds (default: 30)"
    )
    parser.add_argument(
        "--log",
        metavar="FILE",
        help="Write logs to file"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug output"
    )
    parser.add_argument(
        "--checkpoint",
        metavar="FILE",
        help="Save batch progress to checkpoint file"
    )
    parser.add_argument(
        "--resume",
        metavar="FILE",
        help="Resume batch from checkpoint file"
    )
    parser.add_argument(
        "--retry",
        type=int,
        default=0,
        metavar="N",
        help="Retry failed files N times (default: 0)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        metavar="SECS",
        help="Per-file timeout in seconds (default: no timeout)"
    )

    # API server arguments
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Run as API server"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Server host (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (default: 8000)"
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.serve and not args.batch and not args.audio:
        parser.error("audio file path required (or use --batch for directory, or --serve for API server)")

    # Setup logging
    setup_logging(log_file=args.log, verbose=args.verbose)

    # Server mode
    if args.serve:
        try:
            import uvicorn
            from server import app
            print(f"Starting Beat Analyzer API server on {args.host}:{args.port}")
            print(f"API docs: http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}/docs")
            uvicorn.run(app, host=args.host, port=args.port)
        except ImportError as e:
            print(f"Error: Missing server dependencies - {e}", file=sys.stderr)
            print("Run: pip install fastapi uvicorn python-multipart", file=sys.stderr)
            sys.exit(1)
        return

    try:
        if args.batch:
            # Batch mode
            extensions = [e.strip().lstrip('.') for e in args.ext.split(',')]
            result = batch_analyze(
                args.batch,
                extensions,
                recursive=args.recursive,
                use_allin1=not args.fast,
                workers=args.workers,
                min_duration=args.min_duration,
                checkpoint_path=args.checkpoint,
                resume_path=args.resume,
                retry=args.retry,
                timeout=args.timeout
            )

            # Print summary
            print(f"\n{'='*50}")
            print(f"BATCH COMPLETE")
            print(f"{'='*50}")
            print(f"  Total files: {result['batch']['total_files']}")
            print(f"  Processed: {result['batch']['processed']}")
            print(f"  Successful: {result['batch']['successful']}")
            print(f"  Failed: {result['batch']['failed']}")
            print(f"  Skipped: {result['batch']['skipped']}")
            print(f"  Time: {result['batch']['processing_time_seconds']}s")

            # Save results
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"\nSaved to: {args.output}")

            if args.json:
                print(json.dumps(result, indent=2))
        else:
            # Single file mode
            result = analyze_audio(
                args.audio,
                args.output,
                use_allin1=not args.fast,
                trim=args.trim,
                trim_output_dir=args.trim_output
            )

            if args.json:
                print(json.dumps(result, indent=2))

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Analysis failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
