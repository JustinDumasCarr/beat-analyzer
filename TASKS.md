# Beat Analyzer Tasks

## Current Sprint

### Pending

- [ ] **Add API server mode**
  - FastAPI endpoints for analysis
  - POST /analyze with audio file upload
  - GET /status for job tracking

- [ ] **Add batch processing**
  - Analyze multiple files at once
  - Output combined JSON/CSV

- [ ] **Add key detection**
  - Detect musical key for harmonic mixing
  - Use librosa or essentia

- [ ] **Dockerize**
  - Create Dockerfile for deployment
  - Include all dependencies

## Completed

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
