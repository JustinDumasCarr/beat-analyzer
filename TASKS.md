# Beat Analyzer Tasks

## Current Sprint

### In Progress

- [ ] **Improve section detection accuracy**
  - Current SSM approach gives poor results
  - Try allin1 with Python 3.10+ fresh environment
  - Or train custom section classifier

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
- [x] **Basic section detection** - Working (but needs improvement)
- [x] **JSON output** - Full analysis saved to JSON

## Research Notes

### allin1 Dependency Issues
- Requires specific torch + natten versions
- natten 0.21.1 doesn't export expected functions
- madmom requires numpy<2.0
- Consider fresh Python 3.10+ environment

### Alternative Section Detection
- MSAF library (Music Structure Analysis Framework)
- CRNN chorus detection model
- Self-trained model on labeled data
