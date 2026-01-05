🎵 Music Intelligence Analyzer

A Python-based musical structure analysis toolkit for audio and MIDI files.
This project focuses on arrangement, repetition, energy flow, and texture, not genre labels or ML black boxes.

Think of it as X-ray vision for songs.

What This Does (In Plain English)

Given an audio file (WAV / MP3) or MIDI file, the scripts will:

Audio Analysis

Visualize waveform + energy over time

Detect rhythmic impact and transitions

Track spectral texture (brightness / density)

Estimate section boundaries

Measure repetition vs fatigue

Highlight where novelty enters (or doesn’t)

Suggest edit zones (where tightening or escalation could help)

MIDI Analysis

Plot note density over time

Show velocity distribution

Display pitch-class usage

Render a piano roll

This is analysis for musicians, not academic MIR.

Folder Structure
music_intelligence/
│
├── analyze.py              # Main entry point (auto-detects audio vs MIDI)
├── audio_analysis.py       # Audio feature extraction + structure analysis
├── midi_analysis.py        # MIDI feature extraction
├── visuals.py              # All plotting + PNG output
│
└── outputs/                # Generated PNGs (auto-created)

Requirements

macOS

Python 3.14

FFmpeg (for MP3 support)

Python packages:

numpy

librosa

matplotlib

scipy

Install deps (example):

pip install numpy librosa matplotlib scipy


Make sure ffmpeg is available:

ffmpeg -version

How to Run
Analyze an audio file
python3 analyze.py path/to/song.wav
python3 analyze.py path/to/song.mp3

Analyze a MIDI file
python3 analyze.py path/to/song.mid

Output

Each run produces:

Pop-up visualizations (for exploration)

Saved PNGs (for reference / comparison)

PNG files are written to:

outputs/<run_id>_audio_01.png
outputs/<run_id>_audio_02.png
...

Typical Audio Outputs

Waveform + RMS / Onset

Spectral features (centroid / bandwidth / rolloff)

Novelty curve (section-change proxy)

Repetition / fatigue heatmap

Key + chord timeline (low confidence = harmony not dominant)

Section similarity matrix

Edit suggestion overlay

How to Read the Results (Very Important)

This tool does not judge music.

It answers questions like:

Where does new information actually enter?

Where am I repeating by design vs habit?

Do sections sound different or just feel different?

Is length being earned?

Key Interpretations

High repetition ≠ bad

Low novelty ≠ boring

Low key confidence ≠ wrong harmony

Some music (e.g. Tool, ambient, groove-based tracks) is meant to resist novelty.

What This Is Not

❌ A genre classifier

❌ A hit-song predictor

❌ A replacement for ears

This is a decision-support tool, not an authority.

Design Philosophy

Visual-first

Deterministic (no opaque ML)

Musically interpretable

Modular and hackable

Safe to ignore when instinct disagrees

If the graphs disagree with your gut, trust your gut — but ask why.

Typical Use Cases

Tightening arrangements

Comparing versions of the same song

Studying reference tracks

Identifying overlong sections

Learning how repetition functions in your genre

Avoiding accidental stagnation

Final Note

If a song “looks wrong” but feels right, that’s information.

This system doesn’t tell you what to change —
it tells you where you have a choice.

That’s the point.
