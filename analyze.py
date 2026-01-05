import os
import sys
from datetime import datetime

from audio_analysis import analyze_audio
from midi_analysis import analyze_midi


AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aiff", ".aif"}
MIDI_EXTS = {".mid", ".midi"}


def _ensure_output_dir(base_dir: str) -> str:
    out_dir = os.path.join(base_dir, "output")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _timestamp_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python3 analyze.py <path-to-audio-or-midi-file>")
        print("Example: python3 analyze.py samples/song.wav")
        return 2

    in_path = sys.argv[1].strip().strip('"').strip("'")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = _ensure_output_dir(base_dir)

    if not os.path.isfile(in_path):
        print(f"❌ File not found: {in_path}")
        return 2

    ext = os.path.splitext(in_path)[1].lower()
    tag = _timestamp_tag()
    safe_name = os.path.splitext(os.path.basename(in_path))[0]
    run_id = f"{safe_name}_{tag}"

    try:
        if ext in MIDI_EXTS:
            print(f"🎼 Detected MIDI: {in_path}")
            summary = analyze_midi(in_path=in_path, out_dir=out_dir, run_id=run_id, show_plots=True)
        elif ext in AUDIO_EXTS:
            print(f"🎧 Detected Audio: {in_path}")
            summary = analyze_audio(in_path=in_path, out_dir=out_dir, run_id=run_id, show_plots=True)
        else:
            print(f"❌ Unsupported file type: {ext}")
            print(f"Supported audio: {', '.join(sorted(AUDIO_EXTS))}")
            print(f"Supported midi: {', '.join(sorted(MIDI_EXTS))}")
            return 2

        print("\n✅ Analysis complete.\n")
        for k, v in summary.items():
            print(f"- {k}: {v}")
        return 0

    except Exception as e:
        print("\n❌ Analysis failed.")
        print(f"Error: {type(e).__name__}: {e}")
        print("\nIf this is an MP3 decode issue on macOS, try:")
        print("  brew install ffmpeg")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

