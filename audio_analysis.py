import os
from typing import Dict, Any, Tuple, List

import numpy as np
import librosa

from visuals import (
    save_and_show,
    plot_audio_overview,
    plot_audio_features,
    plot_self_similarity_matrix,
    plot_key_and_chords,
    plot_section_similarity_matrix,
    plot_edit_suggestions,
)


def _load_audio(path: str, target_sr: int = 22050) -> Tuple[np.ndarray, int]:
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    if y.size == 0:
        raise ValueError("Loaded audio is empty.")
    return y, sr


def _z(x: np.ndarray) -> np.ndarray:
    return (x - np.mean(x)) / (np.std(x) + 1e-9)


def _detect_sections_aggressive(
    novelty: np.ndarray,
    times: np.ndarray,
    z_thresh: float = 1.2,
    min_gap_sec: float = 4.0,
) -> List[float]:
    nz = _z(novelty)
    idxs = np.where(nz > z_thresh)[0]
    if len(idxs) == 0:
        return []

    sections = [float(times[idxs[0]])]
    last_t = sections[0]
    for idx in idxs[1:]:
        t = float(times[idx])
        if (t - last_t) >= min_gap_sec:
            sections.append(t)
            last_t = t
    return sections


def _chroma_for_analysis(y: np.ndarray, sr: int, hop_length: int) -> np.ndarray:
    # CQT chroma is pretty stable for key/chords
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    chroma = np.maximum(chroma, 0.0)
    chroma = chroma / (np.sum(chroma, axis=0, keepdims=True) + 1e-9)
    return chroma


def _krumhansl_key(chroma_mean: np.ndarray) -> Tuple[str, float]:
    # Krumhansl-Schmuckler profiles (major/minor)
    maj = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], dtype=float)
    minr = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17], dtype=float)
    maj = maj / np.sum(maj)
    minr = minr / np.sum(minr)

    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    def best_rot(profile: np.ndarray) -> Tuple[int, float]:
        scores = []
        for r in range(12):
            scores.append(float(np.dot(chroma_mean, np.roll(profile, r))))
        best = int(np.argmax(scores))
        return best, float(scores[best])

    # normalize chroma_mean
    cm = chroma_mean / (np.sum(chroma_mean) + 1e-9)

    maj_root, maj_score = best_rot(maj)
    min_root, min_score = best_rot(minr)

    if maj_score >= min_score:
        return f"{names[maj_root]} major", maj_score
    return f"{names[min_root]} minor", min_score


def _chord_templates() -> Tuple[List[str], np.ndarray]:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    chords = []
    templ = []
    for root in range(12):
        v = np.zeros(12, dtype=float)
        v[root] = 1.0
        v[(root + 4) % 12] = 1.0
        v[(root + 7) % 12] = 1.0
        templ.append(v / (np.linalg.norm(v) + 1e-9))
        chords.append(f"{names[root]}")

    for root in range(12):
        v = np.zeros(12, dtype=float)
        v[root] = 1.0
        v[(root + 3) % 12] = 1.0
        v[(root + 7) % 12] = 1.0
        templ.append(v / (np.linalg.norm(v) + 1e-9))
        chords.append(f"{names[root]}m")

    return chords, np.stack(templ, axis=0)  # (24, 12)


def _estimate_chords_over_time(
    chroma: np.ndarray,
    times: np.ndarray,
    beat_times: np.ndarray,
) -> Tuple[np.ndarray, List[str]]:
    chord_names, templ = _chord_templates()

    # Use beat-synchronous labeling for musical stability
    if beat_times.size < 2:
        # fallback: chunk chroma into ~0.5s bins via times
        step = max(1, int(0.5 / (times[1] - times[0] + 1e-9)))
        idxs = np.arange(0, chroma.shape[1], step, dtype=int)
        chord_times = times[idxs]
        labels = []
        for i in idxs:
            v = chroma[:, i]
            v = v / (np.linalg.norm(v) + 1e-9)
            scores = templ @ v
            labels.append(chord_names[int(np.argmax(scores))])
        return chord_times, labels

    chord_times = beat_times
    labels = []

    # map beat times to nearest chroma frames
    for bt in beat_times:
        idx = int(np.argmin(np.abs(times - bt)))
        v = chroma[:, idx]
        v = v / (np.linalg.norm(v) + 1e-9)
        scores = templ @ v
        labels.append(chord_names[int(np.argmax(scores))])

    return chord_times, labels


def _self_similarity_from_chroma(chroma_sync: np.ndarray) -> np.ndarray:
    X = chroma_sync.T  # (T, 12)
    X = X - np.mean(X, axis=1, keepdims=True)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    sim = X @ X.T
    sim = (sim + 1.0) / 2.0  # to [0,1]
    return sim


def _section_features_from_chroma(
    chroma: np.ndarray,
    times: np.ndarray,
    sections: List[float],
    duration_sec: float,
) -> Tuple[List[str], np.ndarray]:
    # Build section intervals: [0, s1), [s1, s2), ... [slast, end)
    cuts = [0.0] + sorted([float(s) for s in sections if 0.0 < s < duration_sec]) + [float(duration_sec)]
    labels = []
    feats = []
    for i in range(len(cuts) - 1):
        a, b = cuts[i], cuts[i + 1]
        labels.append(chr(ord("A") + (i % 26)))
        idx = np.where((times >= a) & (times < b))[0]
        if idx.size == 0:
            feats.append(np.zeros(12, dtype=float))
        else:
            v = np.mean(chroma[:, idx], axis=1)
            v = v / (np.linalg.norm(v) + 1e-9)
            feats.append(v)
    return labels, np.stack(feats, axis=0)  # (S,12)


def _cosine_sim_matrix(V: np.ndarray) -> np.ndarray:
    # V: (S, D)
    X = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    sim = X @ X.T
    sim = (sim + 1.0) / 2.0
    return sim


def _edit_suggestions(
    t_onset: np.ndarray,
    novelty: np.ndarray,
    sim: np.ndarray,
    frame_times: np.ndarray,
    sections: List[float],
    min_len_sec: float = 8.0,
) -> List[Tuple[float, float, str]]:
    """
    Suggest trims where:
    - novelty is consistently low
    - repetition is high (self-similarity off-diagonal structure)
    """
    nz = _z(novelty)

    # Low novelty windows
    low = nz < -0.3
    if low.size < 10:
        return []

    # contiguous low-novelty segments in onset-time domain
    segments = []
    start = None
    for i in range(low.size):
        if low[i] and start is None:
            start = i
        if (not low[i] or i == low.size - 1) and start is not None:
            end = i if not low[i] else i + 1
            a = float(t_onset[start])
            b = float(t_onset[end - 1])
            if (b - a) >= min_len_sec:
                segments.append((a, b))
            start = None

    if not segments:
        return []

    # Repetition score for a time t: average similarity to other times (excluding near-diagonal)
    n = sim.shape[0]
    rep = np.zeros(n, dtype=float)
    band = max(3, int(3.0 / (frame_times[1] - frame_times[0] + 1e-9))) if frame_times.size > 1 else 3
    for i in range(n):
        j0 = max(0, i - band)
        j1 = min(n, i + band + 1)
        mask = np.ones(n, dtype=bool)
        mask[j0:j1] = False
        rep[i] = float(np.mean(sim[i, mask])) if np.any(mask) else float(np.mean(sim[i]))

    # Convert segments to suggested cuts when repetition is also high
    suggestions = []
    for (a, b) in segments:
        # map to similarity frame indices
        ia = int(np.argmin(np.abs(frame_times - a)))
        ib = int(np.argmin(np.abs(frame_times - b)))
        if ib <= ia:
            continue
        rep_score = float(np.mean(rep[ia:ib]))

        if rep_score < 0.62:
            continue

        # Snap to nearest detected section boundaries for "musical cuts"
        cuts = [0.0] + sorted(sections) + [float(frame_times[-1]) if frame_times.size else b]
        snap_a = min(cuts, key=lambda x: abs(x - a))
        snap_b = min(cuts, key=lambda x: abs(x - b))
        if snap_b <= snap_a:
            snap_a, snap_b = a, b

        note = f"Trim? (rep {rep_score:.2f})"
        suggestions.append((float(snap_a), float(snap_b), note))

    # Deduplicate overlapping suggestions
    suggestions.sort(key=lambda x: (x[0], x[1]))
    merged = []
    for s in suggestions:
        if not merged:
            merged.append(s)
            continue
        a, b, note = s
        pa, pb, pnote = merged[-1]
        if a <= pb:
            merged[-1] = (pa, max(pb, b), pnote)
        else:
            merged.append(s)

    # cap count
    return merged[:6]


def analyze_audio(in_path: str, out_dir: str, run_id: str, show_plots: bool = True) -> Dict[str, Any]:
    y, sr = _load_audio(in_path)
    duration_sec = float(librosa.get_duration(y=y, sr=sr))

    hop_length = 512
    frame_length = 2048

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=hop_length) if beats is not None else np.array([])

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)[0]

    t_rms = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    t_onset = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr, hop_length=hop_length)
    t_spec = librosa.frames_to_time(np.arange(len(centroid)), sr=sr, hop_length=hop_length)

    novelty = _z(np.interp(t_onset, t_rms, _z(rms), left=float(_z(rms)[0]), right=float(_z(rms)[-1]))) + _z(onset_env)

    sections = _detect_sections_aggressive(
        novelty=novelty,
        times=t_onset,
        z_thresh=1.2,
        min_gap_sec=4.0,
    )

    # --- Repetition/Fatigue: self-similarity from beat-synchronous chroma ---
    chroma = _chroma_for_analysis(y=y, sr=sr, hop_length=hop_length)
    chroma_times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop_length)

    if beat_times.size >= 2:
        chroma_sync = librosa.util.sync(chroma, beats, aggregate=np.mean)
        frame_times = beat_times
    else:
        # fallback: downsample frames
        step = max(1, int(0.5 / (chroma_times[1] - chroma_times[0] + 1e-9)))
        idx = np.arange(0, chroma.shape[1], step, dtype=int)
        chroma_sync = chroma[:, idx]
        frame_times = chroma_times[idx]

    sim = _self_similarity_from_chroma(chroma_sync)

    # --- Key + chord estimation ---
    chroma_mean = np.mean(chroma_sync, axis=1)
    key_label, key_strength = _krumhansl_key(chroma_mean)
    chord_times, chord_labels = _estimate_chords_over_time(chroma_sync, frame_times, beat_times)

    # --- Section similarity matrix ---
    section_labels, section_feats = _section_features_from_chroma(
        chroma=chroma,
        times=chroma_times,
        sections=sections,
        duration_sec=duration_sec,
    )
    section_sim = _cosine_sim_matrix(section_feats)

    # --- Edit suggestions ---
    suggestions = _edit_suggestions(
        t_onset=t_onset,
        novelty=novelty,
        sim=sim,
        frame_times=frame_times,
        sections=sections,
        min_len_sec=8.0,
    )

    # --- Visuals (6 PNGs) ---
    figs = []
    figs.append(plot_audio_overview(y=y, sr=sr, rms=rms, t_rms=t_rms, onset_env=onset_env, t_onset=t_onset, sections=sections))
    figs.append(plot_audio_features(t_spec=t_spec, centroid=centroid, bandwidth=bandwidth, rolloff=rolloff, t_onset=t_onset, novelty=novelty, sections=sections))
    figs.append(plot_self_similarity_matrix(sim=sim, frame_times=frame_times))
    figs.append(plot_key_and_chords(chord_times=chord_times, chord_labels=chord_labels, key_label=key_label, key_strength=key_strength))
    figs.append(plot_section_similarity_matrix(sim=section_sim, labels=section_labels))
    figs.append(plot_edit_suggestions(t_onset=t_onset, novelty=novelty, sections=sections, suggestions=suggestions))

    for i, fig in enumerate(figs, start=1):
        fname = os.path.join(out_dir, f"{run_id}_audio_{i:02d}.png")
        save_and_show(fig=fig, png_path=fname, show=show_plots)

    # --- Console summaries (short, useful) ---
    print("\n📍 Detected Section Boundaries (aggressive):")
    if sections:
        for i, s in enumerate(sections, 1):
            print(f"  {i:02d}. {s:6.2f} sec")
    else:
        print("  (none detected)")

    print(f"\n🎼 Estimated Key: {key_label} (strength {key_strength:.2f})")

    print("\n🧩 Chord Timeline (first 20):")
    for t, c in list(zip(chord_times, chord_labels))[:20]:
        print(f"  {t:6.2f}s  {c}")

    print("\n✂️  Edit Suggestions:")
    if suggestions:
        for a, b, note in suggestions:
            print(f"  {a:6.2f}s → {b:6.2f}s  {note}")
    else:
        print("  (none)")

    dyn = float(np.percentile(rms, 95) - np.percentile(rms, 10))
    return {
        "type": "audio",
        "file": os.path.basename(in_path),
        "sample_rate_hz": sr,
        "duration_sec": round(duration_sec, 2),
        "tempo_bpm_est": float(tempo) if np.isfinite(tempo) else None,
        "section_count": len(sections),
        "key_est": key_label,
        "key_strength": round(float(key_strength), 3),
        "edit_suggestions": len(suggestions),
        "rms_dynamic_range_proxy": round(dyn, 6),
        "png_output_dir": out_dir,
    }
