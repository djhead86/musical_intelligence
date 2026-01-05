import os
from typing import Dict, Any

import numpy as np
import pretty_midi

from visuals import (
    save_and_show,
    plot_midi_overview,
    plot_midi_pitch_histogram,
    plot_midi_pianoroll,
)


def analyze_midi(in_path: str, out_dir: str, run_id: str, show_plots: bool = True) -> Dict[str, Any]:
    pm = pretty_midi.PrettyMIDI(in_path)

    duration_sec = float(pm.get_end_time())

    # Collect note events across non-drum instruments
    notes = []
    inst_names = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        inst_names.append(inst.name if inst.name else "Instrument")
        for n in inst.notes:
            notes.append((n.start, n.end, n.pitch, n.velocity))

    if not notes:
        raise ValueError("No pitched (non-drum) notes found in this MIDI file.")

    notes = np.array(notes, dtype=float)  # start, end, pitch, velocity
    starts = notes[:, 0]
    ends = notes[:, 1]
    pitches = notes[:, 2].astype(int)
    vels = notes[:, 3]

    # Note density over time (per second bins)
    bin_size = 0.5
    bins = np.arange(0.0, max(duration_sec, 0.001) + bin_size, bin_size)
    density = np.zeros(len(bins) - 1, dtype=float)

    for s, e in zip(starts, ends):
        i0 = int(np.floor(s / bin_size))
        i1 = int(np.floor(e / bin_size))
        i0 = max(i0, 0)
        i1 = min(i1, len(density) - 1)
        density[i0 : i1 + 1] += 1.0

    t_density = bins[:-1] + (bin_size / 2.0)

    # Pitch-class histogram (0=C, 1=C#, ...)
    pitch_classes = pitches % 12
    pc_hist = np.bincount(pitch_classes, minlength=12).astype(float)
    pc_hist = pc_hist / (pc_hist.sum() + 1e-9)

    # Piano roll (aggregated) at a modest fs so it’s readable
    fs = 50  # frames per second
    pr = pm.get_piano_roll(fs=fs)  # 128 x T
    # Keep only pitched region for display
    pr_display = pr[21:109, :]  # A0..C8 region

    figs = []
    figs.append(plot_midi_overview(t_density=t_density, density=density, vels=vels))
    figs.append(plot_midi_pitch_histogram(pc_hist=pc_hist))
    figs.append(plot_midi_pianoroll(pianoroll=pr_display, fs=fs))

    for i, fig in enumerate(figs, start=1):
        fname = os.path.join(out_dir, f"{run_id}_midi_{i:02d}.png")
        save_and_show(fig=fig, png_path=fname, show=show_plots)

    # Basic “musical” summary hooks
    mean_vel = float(np.mean(vels))
    note_count = int(len(notes))
    unique_pitches = int(len(np.unique(pitches)))

    return {
        "type": "midi",
        "file": os.path.basename(in_path),
        "duration_sec": round(duration_sec, 2),
        "note_count_non_drum": note_count,
        "unique_pitches": unique_pitches,
        "mean_velocity": round(mean_vel, 2),
        "instruments_non_drum": ", ".join([n for n in inst_names if n]) if inst_names else "(unknown)",
        "png_output_dir": out_dir,
    }

