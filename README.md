# ✨🎵 ClearStage

ClearStage is an AI-powered audio engine designed to clean up and enhance live concert recordings.

## Goal & Purpose
The goal of this project is to take a raw live audio recording and intelligently separate the stems (vocals, drums, bass, other) to reduce crowd noise and improve vocal clarity, resulting in a studio-quality mix.

I started this project to assist in Roehampton's Cabaret Night live recordings which were of poor quality.

## Features
- **AI Source Separation**: Uses Demucs to separate audio into 4 stems.
- **AI Smart Masking**: Uses Time-Frequency Ratio Masking and SNR-based auto-calibration to intelligently suppress crowd noise.
- **Vocal Presence**: Applies targeted EQ to boost vocal clarity and presence.
- **Auto-Mastering**: Recombines stems with a "ClearStage" mixing profile.

## Usage
```bash
python clearstage_engine.py input_file.mp3
```
