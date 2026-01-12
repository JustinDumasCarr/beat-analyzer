# Beat Analyzer

Audio analysis service for extracting BPM, beats, downbeats, and song structure from audio files. Designed for beat-matching in DJ/mixing applications.

## Quick Start

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install cython wheel
pip install -r requirements.txt

# Analyze a track
python analyze.py /path/to/song.mp3 -o output.json
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
- **Beats** - Precise timestamp of every beat
- **Downbeats** - Bar boundaries (first beat of each bar)
- **Sections** - Song structure (intro, verse, chorus, etc.)

## Tech Stack

| Component | Library | Purpose |
|-----------|---------|---------|
| Beat Detection | madmom (RNN) | Neural network beat tracking |
| Downbeat Detection | madmom (DBN) | Bar boundary detection |
| Audio Loading | librosa | Audio I/O and features |
| Section Detection | librosa SSM | Self-similarity segmentation |

## Key Commands

```bash
# Activate environment
source venv/bin/activate

# Analyze single file
python analyze.py song.mp3 -o analysis.json

# Output raw JSON to stdout
python analyze.py song.mp3 --json

# Run tests (when added)
pytest tests/
```

## Current Limitations

1. **Section labels are placeholders** - Uses SSM clustering, not trained section classifier
2. **Requires numpy<2.0** - madmom has compatibility issues with numpy 2.x
3. **allin1 not working** - Better section detection blocked by natten/torch version conflicts

## Future Improvements

1. Get allin1 working (Python 3.10+, proper torch/natten versions)
2. Add API server mode (FastAPI)
3. Add batch processing
4. Improve section labeling with ML model
5. Add key detection (harmonic mixing)

## Integration with Spinola

This service will run on a server and provide beat analysis data to the Spinola mobile app for:
- Beat-matched music playback during spinning workouts
- Clean mix transitions between songs
- Tempo-synced workout intensity

## Autonomous Rules

**Proceed when:**
- Task has clear acceptance criteria
- Tests pass (when added)

**Stop and ask when:**
- New dependencies needed
- Architecture changes
- API design decisions
