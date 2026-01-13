# Beat Analyzer Tasks

## Current Sprint

### Pending

(none)

## Completed

- [x] **Dockerize**
  - Multi-stage Dockerfile (builder + runtime)
  - docker-compose.yml with persistent volumes
  - Environment variable configuration (DATABASE_PATH, UPLOAD_DIR)
  - Health check endpoint integration
- [x] **Phase 5: Enhanced API Features**
  - SQLite persistence for jobs (survives server restarts)
  - Progress reporting (progress_pct, progress_stage in job status)
  - Webhook callbacks (webhook_url parameter, fires on completion/failure)
  - Waveform data extraction (peaks + minmax pairs for visualization)
  - Vocal detection (HPSS + pyin for vocal segment detection)
  - Stem separation (optional demucs integration for vocals/drums/bass/other)
- [x] **API server mode (Phase 4)**
  - FastAPI-based HTTP API with background job processing
  - POST /analyze - Submit audio file for analysis (returns job_id)
  - GET /jobs/{job_id} - Get job status and results
  - GET /jobs - List all jobs with optional status filter
  - DELETE /jobs/{job_id} - Delete job and cleanup files
  - GET /health - Health check with dependency status
  - CLI integration: --serve, --host, --port flags
- [x] **Initial setup** - librosa + madmom beat/downbeat detection
- [x] **BPM detection** - Working, accurate
- [x] **Beat timestamps** - Working via madmom RNN
- [x] **Downbeat detection** - Working via madmom DBN
- [x] **Basic section detection** - Working with SSM fallback
- [x] **JSON output** - Full analysis saved to JSON
- [x] **MSAF integration** - Better section boundary detection using scluster algorithm
- [x] **Python 3.11 + numpy 2.x support** - madmom from GitHub now works with modern numpy
- [x] **allin1 integration** - Patched natten compatibility, now provides semantic section labels (intro/verse/chorus/etc.)
- [x] **Phrase detection** - 4-bar and 8-bar phrase grouping from downbeats
- [x] **Energy analysis** - Classify phrases as buildup/drop/stable/high/low using RMS + spectral centroid
- [x] **Key detection** - Musical key with Camelot notation for DJ harmonic mixing (Krumhansl-Schmuckler algorithm)
- [x] **Batch processing** - Analyze multiple files from directory with parallel processing support (--batch, --workers)
- [x] **Phase 1: Enhanced Analysis & Tagging**
  - Intensity score (0-100) with warmup/moderate/peak/extreme classification
  - Loudness analysis (LUFS, peak dBFS, gain suggestion for -14 LUFS)
  - Mix points detection (mix_in, mix_out, drop_start, buildup_start)
  - Tags system (bpm_range, energy, key_family, has_buildup, has_drop)
- [x] **Phase 2: Audio Trimming**
  - Create trimmed versions with --trim flag
  - _mix version: starts after intro (mix_in point)
  - _drop version: starts at buildup/drop
  - _body version: intro and outro removed (mix_in to mix_out)
  - Uses pydub for audio export (preserves input format)
- [x] **Phase 3: Reliability**
  - Input validation with --min-duration (skip files too short)
  - Retry logic with exponential backoff (--retry N)
  - Checkpointing for batch processing (--checkpoint, --resume)
  - Per-file timeout with process termination (--timeout)
  - Structured logging to file (--log, --verbose)

## Fixed Issues

### allin1 natten compatibility
- **Issue**: natten 0.21+ removed `natten1dav`, `natten1dqkrpb`, `natten2dav`, `natten2dqkrpb`
- **Fix**: Implemented pure PyTorch compatibility shims in `venv/.../allin1/models/dinat.py`
- **Upstream issue**: https://github.com/mir-aidj/all-in-one/issues/30

## Architecture Notes

### Analysis Pipeline
1. **allin1** (default) - ML model provides BPM, beats, downbeats, and semantic section labels
2. **MSAF** (fallback) - Spectral clustering for section boundaries when allin1 unavailable
3. **SSM** (fallback) - Self-similarity matrix for short files or when MSAF fails
4. **madmom** (fallback) - RNN beat/downbeat detection when allin1 unavailable

### Phrase Detection
- Groups downbeats into 4-bar phrases (standard) and 8-bar phrases (larger structure)
- Uses RMS energy + spectral centroid to classify energy trajectory
- Classifications: buildup, drop, stable, high, low
