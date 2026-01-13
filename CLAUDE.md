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
├── analyze.py           # Main CLI tool + analysis functions
├── server.py            # FastAPI HTTP server
├── db.py                # SQLite persistence layer
├── requirements.txt     # Python dependencies
├── Dockerfile           # Multi-stage Docker build
├── docker-compose.yml   # Docker Compose configuration
├── .dockerignore        # Docker build exclusions
├── beat_analyzer.db     # SQLite database (created at runtime)
├── CLAUDE.md           # This file
├── TASKS.md            # Current tasks
└── venv/               # Virtual environment (gitignored)
```

## What It Does

Analyzes audio files and extracts:
- **BPM** - Tempo in beats per minute
- **Key** - Musical key with Camelot notation for harmonic mixing (e.g., "A minor (8A)")
- **Intensity** - Overall song energy score (0-100) classified as warmup/moderate/peak/extreme
- **Loudness** - LUFS measurement with gain suggestion for -14 LUFS target
- **Mix Points** - Optimal timestamps for DJ mixing (mix_in, mix_out, drop_start, buildup_start)
- **Tags** - Searchable metadata (bpm_range, energy, key_family, has_buildup, has_drop)
- **Beats** - Precise timestamp of every beat
- **Downbeats** - Bar boundaries (first beat of each bar)
- **Phrases** - 4-bar and 8-bar musical phrase groupings
- **Energy** - Phrase classification (buildup/drop/stable/high/low)
- **Sections** - Song structure with semantic labels (intro/verse/chorus/bridge/outro)
- **Waveform** - Peak and min/max pairs for audio visualization (1000 points)
- **Vocal Presence** - Segments with vocal detection confidence scores
- **Stems** - Separated tracks (vocals/drums/bass/other) via demucs (optional)

## Tech Stack

| Component | Library | Purpose |
|-----------|---------|---------|
| API Server | FastAPI + uvicorn | HTTP API with async job processing |
| Section Detection | allin1 | ML model for semantic section labels |
| Beat Detection | madmom (RNN) | Neural network beat tracking (fallback) |
| Downbeat Detection | madmom (DBN) | Bar boundary detection (fallback) |
| Audio Loading | librosa | Audio I/O and features |
| Loudness | pyloudnorm | LUFS measurement (EBU R128) |
| Audio Export | pydub | Trimmed version creation |
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

# Create trimmed versions (_mix, _drop, _body)
python analyze.py song.mp3 --trim -o analysis.json

# Trim to specific output directory
python analyze.py song.mp3 --trim --trim-output ./trimmed -o analysis.json

# Batch with reliability features (Phase 3)
# Skip files shorter than 30 seconds
python analyze.py --batch /path/to/music --min-duration 30 -o results.json

# Enable logging to file
python analyze.py --batch /path/to/music --log analysis.log -o results.json

# Verbose debug output
python analyze.py --batch /path/to/music --verbose -o results.json

# Save checkpoint for resumable batches
python analyze.py --batch /path/to/music --checkpoint progress.json -o results.json

# Resume from interrupted batch
python analyze.py --batch /path/to/music --resume progress.json -o results.json

# Retry failed files up to 3 times
python analyze.py --batch /path/to/music --retry 3 -o results.json

# Set per-file timeout (5 minutes)
python analyze.py --batch /path/to/music --timeout 300 -o results.json

# Full reliability setup
python analyze.py --batch /path/to/music \
  --min-duration 30 \
  --log analysis.log \
  --checkpoint progress.json \
  --retry 3 \
  --timeout 300 \
  -o results.json

# Run as API server
python analyze.py --serve --port 8000

# Run tests (when added)
pytest tests/
```

## API Server Mode

Run the analyzer as an HTTP API server for async job processing:

```bash
python analyze.py --serve --host 0.0.0.0 --port 8000
```

**API Documentation:** http://localhost:8000/docs (Swagger UI)

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /analyze | Submit audio file for analysis |
| GET | /jobs/{job_id} | Get job status and results |
| GET | /jobs | List all jobs |
| DELETE | /jobs/{job_id} | Delete a job |
| GET | /health | Health check |

### Example Usage

```bash
# Submit a file for analysis
curl -X POST http://localhost:8000/analyze \
  -F "file=@song.mp3" \
  -F "fast=false"
# Returns: {"job_id": "uuid...", "status": "queued"}

# Check job status
curl http://localhost:8000/jobs/{job_id}
# Returns: {"job_id": "...", "status": "completed", "result": {...}}

# Health check
curl http://localhost:8000/health
# Returns: {"status": "healthy", "allin1_available": true, "msaf_available": true}
```

### Query Parameters

- **POST /analyze**
  - `fast` (bool): Skip ML model for faster processing
  - `trim` (bool): Create trimmed audio versions
  - `waveform` (bool): Include waveform data for visualization (default: true)
  - `vocals` (bool): Include vocal detection (default: true)
  - `stems` (bool): Separate stems via demucs (requires demucs installation)
  - `webhook_url` (str): URL to POST when job completes or fails

- **GET /jobs**
  - `limit` (int): Max jobs to return (default 20)
  - `status` (str): Filter by status (queued, processing, completed, failed)

### Progress Reporting

While processing, jobs include progress information:

```json
{
  "job_id": "...",
  "status": "processing",
  "progress_pct": 60,
  "progress_stage": "section_detection"
}
```

Stages: loading (10%) → key_detection (20%) → beat_detection (40%) → section_detection (60%) → phrase_detection (70%) → intensity (80%) → loudness (85%) → waveform (87%) → vocals (89%) → stems (95%) → complete (100%)

### Webhook Callbacks

When `webhook_url` is provided, a POST request is sent on completion:

```json
{
  "job_id": "...",
  "status": "completed",
  "filename": "song.mp3",
  "completed_at": "2024-01-15T10:30:00",
  "result": {...},
  "error": null
}
```

### Database Persistence

Jobs are stored in SQLite (`beat_analyzer.db`) and survive server restarts. Expired jobs (older than 7 days) are automatically cleaned up on server startup.

## Docker Deployment

### Quick Start

```bash
# Build and run with docker-compose
docker-compose up -d

# Or build manually
docker build -t beat-analyzer:latest .
docker run -d -p 8000:8000 \
  -v beat-analyzer-db:/data/db \
  -v beat-analyzer-uploads:/data/uploads \
  beat-analyzer:latest
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_PATH` | `/data/db/beat_analyzer.db` | SQLite database location |
| `UPLOAD_DIR` | `/data/uploads` | Directory for uploaded files |

### Volumes

| Volume | Purpose |
|--------|---------|
| `/data/db` | Persistent job database |
| `/data/uploads` | Uploaded files and generated stems |
| `/home/appuser/.cache` | ML model cache (allin1, ~1.5GB) |

### Production Tips

- First request with allin1 downloads ~1.5GB models. Pre-warm with a test file or mount a volume with cached models.
- For stem separation, install demucs in a custom Dockerfile (adds ~5.7GB).
- Use `--restart unless-stopped` for automatic recovery.

## Output Format

```json
{
  "file": "song.mp3",
  "bpm": 128.0,
  "key": "A",
  "key_scale": "minor",
  "key_camelot": "8A",
  "key_confidence": 0.762,

  "intensity_score": 75,
  "intensity_class": "peak",

  "loudness_lufs": -8.5,
  "loudness_peak": -0.3,
  "loudness_gain_suggestion": -5.5,

  "mix_points": {
    "mix_in": 15.234,
    "mix_out": 195.456,
    "drop_start": 45.789,
    "buildup_start": 38.123
  },

  "tags": {
    "bpm_range": "fast",
    "energy": "high",
    "key_family": "8",
    "has_buildup": true,
    "has_drop": true
  },

  "trim_points": {
    "intro_end": 15.234,
    "body_start": 15.234,
    "body_end": 195.456,
    "drop_start": 38.123,
    "outro_start": 195.456
  },

  "versions": [
    {"type": "mix", "start": 15.234, "end": 210.0, "file": "song_mix.mp3"},
    {"type": "drop", "start": 38.123, "end": 210.0, "file": "song_drop.mp3"},
    {"type": "body", "start": 15.234, "end": 195.456, "file": "song_body.mp3"}
  ],

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
  ],

  "waveform": {
    "peaks": [0.123, 0.456, ...],
    "minmax": [[0.1, 0.2], [-0.1, 0.15], ...],
    "sample_rate": 22050,
    "duration": 210.0,
    "points": 1000
  },

  "vocal_presence": [
    {"start": 15.2, "end": 45.8, "confidence": 0.85},
    {"start": 60.5, "end": 120.3, "confidence": 0.92}
  ],

  "stems": {
    "vocals": "/path/to/song_vocals.wav",
    "drums": "/path/to/song_drums.wav",
    "bass": "/path/to/song_bass.wav",
    "other": "/path/to/song_other.wav"
  }
}
```

## Notes

- **allin1 patched** - We implemented pure PyTorch shims in `dinat.py` to fix natten 0.21+ API incompatibility
- **Short files (<10s)** - MSAF may fail, falls back to SSM
- **Energy analysis** - Uses RMS + spectral centroid to classify phrase dynamics
- **Vocal detection** - Uses librosa HPSS + pyin for vocal segment detection (no extra dependencies)
- **Stem separation** - Optional: install demucs (~5.7GB models) with `pip install demucs>=4.0.0`
- **Waveform data** - 1000 points of peaks + minmax pairs for visualization (~8KB JSON)

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
