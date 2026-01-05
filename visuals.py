from typing import List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt


def save_and_show(fig: plt.Figure, png_path: str, show: bool = True) -> None:
    fig.tight_layout()
    fig.savefig(png_path, dpi=200)
    if show:
        fig.show()
    else:
        plt.close(fig)


def _add_section_lines(fig: plt.Figure, sections: List[float]) -> None:
    for ax in fig.axes:
        for s in sections:
            ax.axvline(s, linestyle="--", alpha=0.35)


# =========================
# AUDIO VISUALS
# =========================

def plot_audio_overview(
    y: np.ndarray,
    sr: int,
    rms: np.ndarray,
    t_rms: np.ndarray,
    onset_env: np.ndarray,
    t_onset: np.ndarray,
    sections: Optional[List[float]] = None,
) -> plt.Figure:
    fig = plt.figure(figsize=(12, 6))

    ax1 = fig.add_subplot(2, 1, 1)
    t = np.arange(len(y)) / sr
    ax1.plot(t, y)
    ax1.set_title("Waveform (Amplitude vs Time)")
    ax1.set_xlabel("Time (sec)")
    ax1.set_ylabel("Amplitude")

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(t_rms, rms, label="RMS Energy")
    ax2.plot(t_onset, onset_env, label="Onset Strength")
    ax2.set_title("Energy + Onset (Change / Impact Proxy)")
    ax2.set_xlabel("Time (sec)")
    ax2.set_ylabel("Value")
    ax2.legend()

    if sections:
        _add_section_lines(fig, sections)

    return fig


def plot_audio_features(
    t_spec: np.ndarray,
    centroid: np.ndarray,
    bandwidth: np.ndarray,
    rolloff: np.ndarray,
    t_onset: np.ndarray,
    novelty: np.ndarray,
    sections: Optional[List[float]] = None,
) -> plt.Figure:
    fig = plt.figure(figsize=(12, 7))

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(t_spec, centroid, label="Centroid")
    ax1.plot(t_spec, bandwidth, label="Bandwidth")
    ax1.plot(t_spec, rolloff, label="Rolloff")
    ax1.set_title("Spectral Features (Brightness / Texture)")
    ax1.set_xlabel("Time (sec)")
    ax1.set_ylabel("Hz")
    ax1.legend()

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(t_onset, novelty, label="Novelty")
    ax2.set_title("Novelty Curve (Section Change Proxy)")
    ax2.set_xlabel("Time (sec)")
    ax2.set_ylabel("Score")
    ax2.legend()

    if sections:
        _add_section_lines(fig, sections)

    return fig


def plot_self_similarity_matrix(sim: np.ndarray, frame_times: np.ndarray) -> plt.Figure:
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(sim, origin="lower", aspect="auto", interpolation="nearest")
    ax.set_title("Repetition / Fatigue Heatmap")
    ax.set_xlabel("Time")
    ax.set_ylabel("Time")
    return fig


def plot_key_and_chords(
    chord_times: np.ndarray,
    chord_labels: List[str],
    key_label: str,
    key_strength: float,
) -> plt.Figure:
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(1, 1, 1)

    uniq = list(dict.fromkeys(chord_labels))
    mapping = {c: i for i, c in enumerate(uniq)}
    y = [mapping[c] for c in chord_labels]

    ax.step(chord_times, y, where="post")
    ax.set_title(f"Key + Chord Timeline — {key_label} ({key_strength:.2f})")
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Chord Index")

    if len(uniq) <= 20:
        ax.set_yticks(range(len(uniq)))
        ax.set_yticklabels(uniq)

    return fig


def plot_section_similarity_matrix(sim: np.ndarray, labels: List[str]) -> plt.Figure:
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(sim, origin="lower", interpolation="nearest")
    ax.set_title("Section Similarity Matrix")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    for i in range(sim.shape[0]):
        for j in range(sim.shape[1]):
            ax.text(j, i, f"{sim[i, j]:.2f}", ha="center", va="center", fontsize=8)

    return fig


def plot_edit_suggestions(
    t_onset: np.ndarray,
    novelty: np.ndarray,
    sections: List[float],
    suggestions: List[Tuple[float, float, str]],
) -> plt.Figure:
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(t_onset, novelty, label="Novelty")

    for s in sections:
        ax.axvline(s, linestyle="--", alpha=0.3)

    for a, b, note in suggestions:
        ax.axvspan(a, b, alpha=0.2)
        ax.text((a + b) / 2, np.max(novelty) * 0.8, note, ha="center", fontsize=9)

    ax.set_title("Edit Suggestions")
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Novelty")
    ax.legend()

    return fig


# =========================
# MIDI VISUALS (RESTORED)
# =========================

def plot_midi_overview(t_density: np.ndarray, density: np.ndarray, vels: np.ndarray) -> plt.Figure:
    fig = plt.figure(figsize=(12, 6))

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(t_density, density)
    ax1.set_title("MIDI Note Density Over Time")
    ax1.set_xlabel("Time (sec)")
    ax1.set_ylabel("Active Notes")

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.hist(vels, bins=32)
    ax2.set_title("Velocity Distribution")
    ax2.set_xlabel("Velocity")
    ax2.set_ylabel("Count")

    return fig


def plot_midi_pitch_histogram(pc_hist: np.ndarray) -> plt.Figure:
    labels = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(range(12), pc_hist)
    ax.set_xticks(range(12))
    ax.set_xticklabels(labels)
    ax.set_title("Pitch Class Histogram")
    return fig


def plot_midi_pianoroll(pianoroll: np.ndarray, fs: int) -> plt.Figure:
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(pianoroll, aspect="auto", origin="lower", interpolation="nearest")
    ax.set_title("Piano Roll")
    ax.set_xlabel("Time (frames)")
    ax.set_ylabel("Pitch")
    return fig
