# Beat Analyzer

Audio analysis service for extracting BPM, beats, downbeats, and song structure from audio files.

## Features

- **BPM Detection** - Accurate tempo estimation
- **Beat Tracking** - Precise timestamps for every beat (madmom RNN)
- **Downbeat Detection** - Bar boundaries for mix alignment
- **Section Detection** - Song structure segmentation (intro, verse, chorus, etc.)

## Installation

```bash
# Clone and setup
git clone <repo-url>
cd beat-analyzer

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install cython wheel
pip install -r requirements.txt
```

## Usage

```bash
# Analyze a track
python analyze.py /path/to/song.mp3

# Save output to JSON
python analyze.py /path/to/song.mp3 -o analysis.json

# Output JSON to stdout
python analyze.py /path/to/song.mp3 --json
```

## Output Format

```json
{
  "file": "song.mp3",
  "bpm": 128.0,
  "beats": [0.19, 0.66, 1.13, ...],
  "downbeats": [0.19, 2.06, 3.93, ...],
  "segments": [
    {"start": 0.0, "end": 15.5, "label": "intro"},
    {"start": 15.5, "end": 45.2, "label": "verse"},
    ...
  ],
  "beat_count": 258,
  "bars": 65,
  "duration": 154.96
}
```

## Tech Stack

- **librosa** - Audio loading and feature extraction
- **madmom** - Neural network beat/downbeat tracking
- **numpy/scipy** - Scientific computing

## License

MIT
