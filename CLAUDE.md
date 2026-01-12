# Beat Analyzer

Audio analysis service for extracting BPM, beats, downbeats, phrases, and song structure from audio files. Designed for beat-matching in DJ/mixing applications.

## Quick Start

```bash
# Setup (requires Python 3.11+)
python3.11 -m venv venv
source venv/bin/activate
pip install cython wheel
pip install -r requirements.txt
pip install allin1  # For semantic section labels

# Analyze a track (uses allin1 ML model)
python analyze.py /path/to/song.mp3 -o output.json

# Fast mode (skip allin1, use MSAF/SSM fallback)
python analyze.py /path/to/song.mp3 --fast -o output.json
```

## Project Structure

```
beat-analyzer/
├── analyze.py           # Main CLI tool
├── requirements.txt     # Python dependencies
├── CLAUDE.md           # This file
├── TASKS.md            # Current tasks
└── venv/               # Virtual environment (gitignored)
```

## What It Does

Analyzes audio files and extracts:
- **BPM** - Tempo in beats per minute
- **Key** - Musical key with Camelot notation for harmonic mixing (e.g., "A minor (8A)")
- **Beats** - Precise timestamp of every beat
- **Downbeats** - Bar boundaries (first beat of each bar)
- **Phrases** - 4-bar and 8-bar musical phrase groupings
- **Energy** - Phrase classification (buildup/drop/stable/high/low)
- **Sections** - Song structure with semantic labels (intro/verse/chorus/bridge/outro)

## Tech Stack

| Component | Library | Purpose |
|-----------|---------|---------|
| Section Detection | allin1 | ML model for semantic section labels |
| Beat Detection | madmom (RNN) | Neural network beat tracking (fallback) |
| Downbeat Detection | madmom (DBN) | Bar boundary detection (fallback) |
| Audio Loading | librosa | Audio I/O and features |
| Section Fallback | MSAF (scluster) | Spectral clustering segmentation |
| Section Fallback | librosa SSM | Self-similarity matrix (for short files) |

## Key Commands

```bash
# Activate environment
source venv/bin/activate

# Analyze single file
python analyze.py song.mp3 -o analysis.json

# Fast mode (skip allin1)
python analyze.py song.mp3 --fast -o analysis.json

# Output raw JSON to stdout
python analyze.py song.mp3 --json

# Batch mode - analyze directory
python analyze.py --batch /path/to/music -o results.json

# Batch with parallel processing (4 workers)
python analyze.py --batch /path/to/music -w 4 -o results.json

# Batch with extension filter
python analyze.py --batch /path/to/music --ext mp3,wav -o results.json

# Batch recursive search
python analyze.py --batch /path/to/music --recursive -o results.json

# Run tests (when added)
pytest tests/
```

## Output Format

```json
{
  "file": "song.mp3",
  "bpm": 128.0,
  "key": "A",
  "key_scale": "minor",
  "key_camelot": "8A",
  "key_confidence": 0.762,
  "beats": [0.234, 0.702, ...],
  "downbeats": [0.234, 2.104, ...],
  "phrases": [
    {"start": 0.234, "end": 7.734, "bar_count": 4, "phrase_number": 1, "energy_type": "buildup"},
    ...
  ],
  "phrases_8bar": [...],
  "sections": [
    {"start": 0.0, "end": 15.5, "label": "intro"},
    {"start": 15.5, "end": 45.2, "label": "verse"},
    ...
  ]
}
```

## Notes

- **allin1 patched** - We implemented pure PyTorch shims in `dinat.py` to fix natten 0.21+ API incompatibility
- **Short files (<10s)** - MSAF may fail, falls back to SSM
- **Energy analysis** - Uses RMS + spectral centroid to classify phrase dynamics

## Integration with Spinola

This service will run on a server and provide beat analysis data to the Spinola mobile app for:
- Beat-matched music playback during spinning workouts
- Clean mix transitions between songs
- Tempo-synced workout intensity
- Phrase-aligned section transitions

## Autonomous Rules

**Proceed when:**
- Task has clear acceptance criteria
- Tests pass (when added)

**Stop and ask when:**
- New dependencies needed
- Architecture changes
- API design decisions
