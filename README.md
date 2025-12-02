# ✨🎵 ClearStage

ClearStage is an AI-powered audio engine designed to clean up and enhance live concert recordings.

## Goal & Purpose
The goal of this project is to take a raw live audio recording and intelligently separate the stems (vocals, drums, bass, other) to reduce crowd noise and improve vocal clarity, resulting in a studio-quality mix.

I started this project to assist in Roehampton's Cabaret Night live recordings which were of poor quality.

## Features
- **AI Source Separation**: Uses Demucs (HTDemucs_FT) for high-fidelity stem separation.
- **Hard Center Vocal Extraction**: Removes stereo bleed using Mid-Side processing to isolate vocals.
- **Crowd Taming**: Smooths stereo width and removes harsh transients from the crowd noise.
- **Smart Vocal Ducking**: Automatically lowers backing tracks when vocals are present.
- **Vocal Polishing**: Applies parallel compression and dynamic expansion for studio presence.
- **Auto-Mastering**: Recombines stems with a "ClearStage" mixing profile.

## Usage
```bash
python clearstage_engine.py input_file.mp3
```
