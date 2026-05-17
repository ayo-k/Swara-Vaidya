import librosa
import numpy as np
import json
import whisper as _whisper
from sklearn.preprocessing import StandardScaler
from librosa.sequence import dtw
from difflib import SequenceMatcher
import re
from scipy.ndimage import uniform_filter1d         # Smooth pitch curve
from scipy.signal import find_peaks, resample      # Peak detection, resamp
import torch
# Load Whisper once (shared by direct run and API)
# ── These are safe at module level (just assignments, no I/O) ──
WHISPER_MODEL = None   # loaded lazily in _load_references()
ref_pitch     = None
ref_tempo     = None
ref_mfcc      = None
ref_audio     = None
ref_sr        = None
ref_mantra    = None

def _load_references():
    """
    Called ONCE when the first request arrives (not at import time).
    This lets uvicorn bind the port immediately, then load heavy models.
    """
    global WHISPER_MODEL, ref_pitch, ref_tempo, ref_mfcc
    global ref_audio, ref_sr, ref_mantra

    if WHISPER_MODEL is not None:
        return   # already loaded — skip

    print("Loading Whisper model...")
    WHISPER_MODEL = _whisper.load_model("base")
    print("Whisper ready.")

    with open("reference_features.json", "r") as f:
        reference = json.load(f)

    ref_pitch  = np.array(reference["pitch_contour"])
    ref_tempo  = reference["tempo"]
    ref_mfcc   = np.array(reference["mfcc"])
    ref_mantra = reference["mantra_text"]

    ref_audio_raw, ref_sr_raw = librosa.load("chant1.wav")
    ref_audio_trimmed, _      = librosa.effects.trim(ref_audio_raw)

    ref_audio = ref_audio_trimmed
    ref_sr    = ref_sr_raw

    print("Reference data loaded.")
# -----------------------
# 1. Pitch similarity
# -----------------------
# We compare how similar the pitch movements are between reference and user.
# Steps:
#   1. Keep only voiced frames (where pitch > 0, i.e. voice is active)
#   2. Convert to log scale (human ears hear pitch on a log scale)
#   3. Z-score normalize (removes gender/voice type difference)
#   4. Resample to same length (DTW works better with equal length)
#   5. DTW comparison
#   6. Convert distance to similarity score
 
# =============================================================================
# 1-a PITCH ZONE ANALYSIS (Vedic Accent: Udatta / Anudatta / Svarita)
# =============================================================================
# Based on research by Begus (2016) and BHASHA 2025 paper.
#
# Vedic Sanskrit has 3 pitch accent zones:
#   Anudatta  = low flat region (before the accent)
#   Udatta    = rising region   (the accent peak)
#   Svarita   = falling region  (after the peak)
#
# Instead of comparing just the pitch numbers,
# we detect WHICH ZONE is happening at each moment using slope:
#   slope > 0  → pitch is rising  → Udatta
#   slope < 0  → pitch is falling → Svarita
#   slope ≈ 0  → pitch is flat    → Anudatta
 
def extract_pitch_zones(pitch_contour):
    """
    IMPROVED VERSION — from Document 8 (upgraded pitch zone extraction).
    
    Takes a raw pitch array.
    Returns list of zones detected: anudatta / udatta / svarita
    and the normalized pitch array.
    
    Upgrades over old version:
      - Uses find_peaks() for robust peak detection (handles multiple peaks)
      - Finds dominant peak near CENTER of contour (more reliable for mantras)
      - Handles plateau shapes (long flat sections don't confuse the detector)
      - Slope threshold prevents noise from creating false zone switches
    """
    # Only keep voiced frames (where pitch > 0)
    voiced = [(i, p) for i, p in enumerate(pitch_contour) if p > 0]
 
    if len(voiced) < 6:
        return [], np.array([])
 
    pitches = np.array([v[1] for v in voiced])
 
    # --- Step 1: Smooth pitch ---
    # Moving average with window=5 removes frame-to-frame jitter
    # without destroying the overall shape of the curve
    smoothed = uniform_filter1d(pitches, size=5)
 
    # --- Step 2: Z-score normalize (speaker/gender independent) ---
    # After this step: mean=0, std=1 for both male and female voices
    # DTW comparison becomes gender-neutral
    norm = (smoothed - np.mean(smoothed)) / (np.std(smoothed) + 1e-9)
 
    # --- Step 3: Compute slope (gradient) ---
    # np.gradient = rate of change at each point
    # slope[i] > 0  → pitch rising at frame i
    # slope[i] < 0  → pitch falling at frame i
    # slope[i] ≈ 0  → pitch flat at frame i
    slope = np.gradient(norm)
 
    # --- Step 4: Find dominant peak near center ---
    # find_peaks() finds ALL local maxima in the array
    # We then pick the one closest to the CENTER of the contour
    # Why center? Because in most mantras the main accent peak
    # falls roughly in the middle of the phrase, not at the edges.
    # This is more robust than just taking np.argmax() which can
    # pick an edge artifact.
    peaks, _ = find_peaks(norm)
    if len(peaks) == 0:
        # No peaks found — fall back to global maximum
        peak_idx = int(np.argmax(norm))
    else:
        center   = len(norm) // 2
        # Pick the peak closest to center
        peak_idx = peaks[int(np.argmin(np.abs(peaks - center)))]
 
    # --- Step 5: Classify each frame into a zone using slope ---
    # threshold=0.05 means slope must be meaningfully positive/negative
    # to count as rising/falling. This prevents noise spikes from
    # creating spurious zone changes.
    zones        = []
    current_zone = None
    start        = 0
    threshold    = 0.05
 
    for i in range(len(norm)):
        if slope[i] > threshold:
            zone = "udatta"      # rising → Udatta accent region
        elif slope[i] < -threshold:
            zone = "svarita"     # falling → Svarita (post-peak fall)
        else:
            zone = "anudatta"    # flat/low → Anudatta (pre-accent region)
 
        if current_zone is None:
            current_zone = zone
            start        = i
        elif zone != current_zone:
            zones.append({
                "zone":      current_zone,
                "avg_pitch": float(np.mean(norm[start:i])),
                "length":    i - start
            })
            current_zone = zone
            start        = i
 
    # Save last zone
    zones.append({
        "zone":      current_zone,
        "avg_pitch": float(np.mean(norm[start:])),
        "length":    len(norm) - start
    })
 
    return zones, norm
 
 
def compare_pitch_zones(ref_zones, user_zones):
    """
    Compares Vedic accent zones between reference and user.
    Returns: zone_similarity (0-100), list of feedback messages
    """
    if not ref_zones or not user_zones:
        return 0.0, ["Could not detect pitch zones — recording may be too quiet"]
 
    feedback = []
    scores   = []
 
    # Convert list of zones to dict for easy lookup
    # Note: if same zone appears multiple times, last one wins
    # This is fine for a first pass
    ref_dict  = {z["zone"]: z for z in ref_zones}
    user_dict = {z["zone"]: z for z in user_zones}
 
    for zone in ["anudatta", "udatta", "svarita"]:
        if zone not in ref_dict:
            continue   # reference doesn't have this zone, skip
 
        if zone not in user_dict:
            feedback.append(f"Missing {zone} section in your chant")
            scores.append(0.0)
            continue
 
        ref_z  = ref_dict[zone]
        user_z = user_dict[zone]
 
        # Compare normalized pitch levels
        # Since both are normalized, difference should be small for good match
        pitch_diff = abs(ref_z["avg_pitch"] - user_z["avg_pitch"])
        zone_score = max(0.0, 1.0 - pitch_diff)
        scores.append(zone_score)
 
        # Generate specific feedback
        if zone == "udatta" and zone_score < 0.7:
            feedback.append("Udatta (rising accent) is not rising high enough")
        elif zone == "svarita" and zone_score < 0.7:
            feedback.append("Svarita (falling tone after peak) is not dropping correctly")
        elif zone == "anudatta" and zone_score < 0.7:
            feedback.append("Anudatta (low starting region) pitch is unstable")
 
    # Check peak timing
    # The Beguš paper shows that WHEN the peak happens matters —
    # independent svarita peaks earlier (compressed udatta)
    ref_total  = sum(z["length"] for z in ref_zones)
    user_total = sum(z["length"] for z in user_zones)
 
    if ref_total > 0 and user_total > 0:
        ref_peak_ratio  = ref_dict.get("udatta",  {}).get("length", 0) / ref_total
        user_peak_ratio = user_dict.get("udatta", {}).get("length", 0) / user_total
        timing_diff     = abs(ref_peak_ratio - user_peak_ratio)
 
        if timing_diff > 0.2:
            feedback.append(
                f"Pitch peak timing is off — "
                f"reference peaks at {round(ref_peak_ratio * 100)}% of chant, "
                f"you peaked at {round(user_peak_ratio * 100)}%"
            )
 
    zone_similarity = float(np.mean(scores)) * 100 if scores else 0.0
    return zone_similarity, feedback
 
# =============================================================================
# 1-b — ACCENT LEVEL CLASSIFIER (BHASHA 2025 paper)
# =============================================================================
# The BHASHA paper found that Rigvedic Sanskrit has 3 pitch accent levels:
#   Anudatta = low    (label 1)
#   Neutral  = medium (label 2)
#   Udatta   = high   (label 3)
#
# A wrong accent CHANGES THE MEANING of the Sanskrit word.
# Your pitch contour DTW catches overall shape, but a user could follow
# the general melody while placing the HIGH tone on the WRONG syllable.
# This classifier catches that specific error.
#
# How it works:
#   - Take all voiced (non-zero) pitch values
#   - Find the 33rd percentile (low_thresh) and 66th percentile (high_thresh)
#   - Classify every frame: below low_thresh=Anudatta, above high_thresh=Udatta
#   - Compare the sequence of labels using SequenceMatcher
 
def classify_accent_levels(pitch_contour):
    """
    Classifies each pitch frame into one of 3 Vedic accent levels.
    Returns array of labels: 0=silence, 1=Anudatta(low), 2=neutral, 3=Udatta(high)
    
    Based on BHASHA 2025 paper — accent placement is linguistically meaningful.
    A wrong label at a syllable = wrong accent = possibly wrong meaning.
    """
    voiced = pitch_contour[pitch_contour > 0]
    if len(voiced) == 0:
        return np.array([])
 
    # Percentile thresholds divide pitch range into 3 equal parts
    # 33rd percentile = lower third of pitch range = Anudatta region
    # 66th percentile = upper third of pitch range = Udatta region
    low_thresh  = np.percentile(voiced, 33)
    high_thresh = np.percentile(voiced, 66)
 
    levels = []
    for p in pitch_contour:
        if p == 0:
            levels.append(0)    # silence — not voiced
        elif p < low_thresh:
            levels.append(1)    # Anudatta — low pitch
        elif p < high_thresh:
            levels.append(2)    # neutral — middle pitch
        else:
            levels.append(3)    # Udatta — high pitch (accent peak)
    return np.array(levels)
 
 
# =============================================================================
# 1-c — INTRA-SYLLABLE PITCH SHAPE DETECTOR (Beguš 2016 paper)
# =============================================================================
# The Beguš paper explains that within a single syllable,
# the DIRECTION of pitch movement identifies the accent type:
#
#   Udatta   = rising pitch within the syllable     (low → high)
#   Anudatta = falling pitch within the syllable    (high → low)
#   Svarita  = rising-then-falling (circumflex)     (low → high → low)
#   Flat     = sustained note, no clear direction
#
# This is DIFFERENT from the zone analysis (Section 5a) which looks at
# the full contour. This looks at small windows (individual syllables).
#
# Why does this matter?
# Example: user chants a syllable with a flat tone where
# reference has a rising (Udatta) tone. Zone analysis might miss this
# if the overall contour shape is similar. This detector catches it.
 
def detect_pitch_shape(pitch_segment):
    """
    Classifies the pitch movement WITHIN a short segment (one syllable).
    
    Returns one of: 'rising', 'falling', 'rising_falling', 'flat'
    
    Based on Beguš 2016:
      rising        = Udatta accent
      falling       = Anudatta accent  
      rising_falling = Svarita (independent svarita = compressed udatta)
      flat          = sustained, no accent
    """
    # Need at least 3 frames to detect a shape
    if len(pitch_segment) < 3:
        return 'flat'
 
    mid   = len(pitch_segment) // 2
    first = np.mean(pitch_segment[:mid])    # average pitch in first half
    last  = np.mean(pitch_segment[mid:])    # average pitch in second half
    peak  = np.max(pitch_segment)           # highest point in the segment
 
    # 1.05 threshold = must be 5% higher to count as meaningful change
    # This prevents tiny fluctuations from being classified as movement
    rising          = last  > first * 1.05
    falling         = first > last  * 1.05
    # Peak in middle = pitch went up then came down within this segment
    has_peak_middle = (peak > first * 1.05) and (peak > last * 1.05)
 
    if has_peak_middle:
        return 'rising_falling'   # Svarita — circumflex shape
    elif rising:
        return 'rising'           # Udatta — accent rising
    elif falling:
        return 'falling'          # Anudatta — falling after accent
    else:
        return 'flat'             # sustained, no clear accent
 
 
def analyze_syllable_shapes(pitch_contour, n_segments=8):
    """
    Divides the pitch contour into n_segments equal windows
    and classifies the pitch shape of each window.
    
    Returns list of shape labels like:
    ['flat', 'rising', 'rising_falling', 'falling', 'flat', ...]
    
    n_segments=8 approximates syllable-level analysis for a short mantra.
    For longer mantras you can increase this.
    """
    voiced = pitch_contour[pitch_contour > 0]
    if len(voiced) < n_segments:
        return []
 
    # Split voiced frames into n equal windows
    segments = np.array_split(voiced, n_segments)
    shapes   = [detect_pitch_shape(seg) for seg in segments]
    return shapes
 
 
def shape_sequence_similarity(ref_shapes, user_shapes):
    """
    Compares the sequence of pitch shapes between reference and user.
    Returns similarity score 0-100 and list of feedback messages.
    
    Example:
      ref:  ['flat', 'rising', 'rising_falling', 'falling']
      user: ['flat', 'flat',   'rising_falling', 'falling']
      → segment 2 is wrong: user has 'flat' where ref has 'rising' (Udatta)
    """
    if not ref_shapes or not user_shapes:
        return 0.0, []
 
    # Compare shape sequences using SequenceMatcher
    score    = SequenceMatcher(None, ref_shapes, user_shapes).ratio() * 100
    feedback = []
 
    # Find specific segments that differ
    matcher = SequenceMatcher(None, ref_shapes, user_shapes)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            ref_shape  = ref_shapes[i1]
            user_shape = user_shapes[j1] if j1 < len(user_shapes) else 'unknown'
 
            # Give specific Vedic accent feedback
            if ref_shape == 'rising' and user_shape != 'rising':
                feedback.append(
                    f"Syllable group {i1+1}: should have an Udatta (rising accent) "
                    f"but yours sounds {user_shape}"
                )
            elif ref_shape == 'rising_falling' and user_shape != 'rising_falling':
                feedback.append(
                    f"Syllable group {i1+1}: should have a Svarita "
                    f"(rising-then-falling circumflex) but yours sounds {user_shape}"
                )
            elif ref_shape == 'falling' and user_shape != 'falling':
                feedback.append(
                    f"Syllable group {i1+1}: should have an Anudatta "
                    f"(falling tone) but yours sounds {user_shape}"
                )
 
    return score, feedback
 
 
# --- Pitch slope similarity ---
# The Beguš paper: independent svarita = same pitch targets but STEEPER slope
# So we also compare the rate of pitch change (velocity/slope)
 
def pitch_slope_similarity(ref_p, user_p):
    """
    Compares how fast pitch rises and falls.
    Captures steepness difference between udatta and independent svarita.
    """
    def get_slopes(p):
        voiced = p[p > 0]
        if len(voiced) < 2:
            return np.array([0.0])
        log_p = np.log2(voiced)
        return np.diff(log_p)   # differences between consecutive pitch values
 
    ref_slopes  = get_slopes(ref_p)
    user_slopes = get_slopes(user_p)
 
    if len(ref_slopes) == 0 or len(user_slopes) == 0:
        return 0.0
 
    D, wp = dtw(
        ref_slopes.reshape(1, -1),
        user_slopes.reshape(1, -1),
        metric='euclidean'
    )
    slope_dist = D[-1, -1] / len(wp)
    return 100 * np.exp(-slope_dist / 0.5)
 
# --- Combined pitch score ---
# We now have 5 pitch measurements, each from a different research insight:
#
#   pitch_similarity  — DTW on z-scored pitch contour (overall shape)
#                       catches: melody wrong, wrong key relative movement
#
#   zone_sim          — Udatta/Anudatta/Svarita zone presence & levels
#                       catches: missing the rising/falling arc of the mantra
#                       (from Beguš + BHASHA papers)
#
#   slope_sim         — rate of pitch change (steepness)
#                       catches: svarita compressed vs expanded wrong
#                       (from Beguš 2016 — independent svarita = steeper udatta)
#
#   accent_sim        — frame-level Anudatta/Neutral/Udatta label sequence
#                       catches: Udatta placed on WRONG syllable
#                       (from BHASHA 2025 — wrong accent = wrong meaning)
#
#   shape_sim         — intra-syllable pitch shape (rising/falling/circumflex)
#                       catches: wrong shape on individual syllables
#                       (from Beguš 2016 — svarita = circumflex within syllable)

# =============================================================================
# WORD-LEVEL SVARA MAP
# Maps pitch analysis back to actual words so feedback says:
# "On 'gam' (word 2): expected udatta (rising), you sang anudatta (flat)"
# =============================================================================

HOP_LENGTH = 512  # default librosa hop length

def build_word_svara_map(pitch_contour, word_timestamps, audio_sr):
    """
    For each word, slices the pitch contour using Whisper timestamps
    and classifies dominant_accent + pitch_shape for that word.
    """
    if not word_timestamps or len(pitch_contour) == 0:
        return []

    voiced_vals = pitch_contour[pitch_contour > 0]
    if len(voiced_vals) == 0:
        return []

    p_mean      = np.mean(voiced_vals)
    p_std       = np.std(voiced_vals) + 1e-9
    low_thresh  = np.percentile(voiced_vals, 33)
    high_thresh = np.percentile(voiced_vals, 66)

    word_map = []
    for wt in word_timestamps:
        start_frame = int(wt["start"] * audio_sr / HOP_LENGTH)
        end_frame   = min(int(wt["end"] * audio_sr / HOP_LENGTH), len(pitch_contour) - 1)

        if start_frame >= end_frame:
            word_map.append({"word": wt["word"], "start": wt["start"], "end": wt["end"],
                             "dominant_accent": "silence", "pitch_shape": "flat", "avg_pitch_norm": 0.0})
            continue

        segment    = pitch_contour[start_frame:end_frame]
        voiced_seg = segment[segment > 0]

        if len(voiced_seg) < 3:
            dominant_accent, pitch_shape, avg_norm = "silence", "flat", 0.0
        else:
            counts = {"anudatta": 0, "neutral": 0, "udatta": 0}
            for p in voiced_seg:
                if p < low_thresh:       counts["anudatta"] += 1
                elif p < high_thresh:    counts["neutral"]  += 1
                else:                    counts["udatta"]   += 1
            dominant_accent = max(counts, key=counts.get)
            pitch_shape     = detect_pitch_shape(voiced_seg)  # reuses existing function
            avg_norm        = float((np.mean(voiced_seg) - p_mean) / p_std)

        word_map.append({"word": wt["word"], "start": wt["start"], "end": wt["end"],
                         "dominant_accent": dominant_accent, "pitch_shape": pitch_shape,
                         "avg_pitch_norm": avg_norm})
    return word_map


def build_ref_word_svara_map(ref_pitch_contour, ref_word_list, ref_audio_array, ref_sample_rate):
    """
    Reference has no Whisper timestamps, so divide equally across known words.
    """
    if not ref_word_list or len(ref_pitch_contour) == 0:
        return []
    duration_sec  = len(ref_audio_array) / ref_sample_rate
    word_duration = duration_sec / len(ref_word_list)
    synthetic_ts  = [{"word": w, "start": i * word_duration, "end": (i+1) * word_duration}
                     for i, w in enumerate(ref_word_list)]
    return build_word_svara_map(ref_pitch_contour, synthetic_ts, ref_sample_rate)


def compare_word_svara_maps(ref_map, user_map):
    """
    Compares reference and user word-by-word.
    Returns list of per-word feedback dicts with exact fix messages.
    """
    if not ref_map or not user_map:
        return []

    ACCENT_MEANING = {
        "udatta":   "udatta — pitch must RISE on this syllable",
        "anudatta": "anudatta — pitch stays LOW and flat",
        "neutral":  "neutral — steady middle pitch",
        "silence":  "silence",
    }
    SHAPE_MEANING = {
        "rising":         "rising / udatta (vowel climbs upward)",
        "falling":        "falling / anudatta (vowel descends)",
        "rising_falling": "svarita — rise then fall within the vowel (circumflex)",
        "flat":           "flat / sustained (no accent movement)",
    }

    items = []
    total = max(len(ref_map), len(user_map))

    for i in range(min(len(ref_map), len(user_map))):
        rw, uw = ref_map[i], user_map[i]
        word   = uw["word"]
        wnum   = i + 1

        accent_wrong = rw["dominant_accent"] != uw["dominant_accent"]
        shape_wrong  = rw["pitch_shape"]     != uw["pitch_shape"]

        accent_fix = shape_fix = None

        if accent_wrong:
            ra, ua = rw["dominant_accent"], uw["dominant_accent"]
            if ra == "udatta":
                accent_fix = (f"On '{word}' (word {wnum}/{total}): lift your pitch upward on the vowel. "
                              f"This syllable carries the udatta accent — it must rise clearly. "
                              f"You sang {ACCENT_MEANING.get(ua, ua)} instead.")
            elif ra == "anudatta":
                accent_fix = (f"On '{word}' (word {wnum}/{total}): keep pitch LOW and even. "
                              f"Anudatta is the quiet ground before the accent. "
                              f"You sang {ACCENT_MEANING.get(ua, ua)} instead.")
            elif ra == "neutral":
                accent_fix = (f"On '{word}' (word {wnum}/{total}): hold a steady middle pitch. "
                              f"You sang {ACCENT_MEANING.get(ua, ua)} instead.")

        if shape_wrong:
            rs, us = rw["pitch_shape"], uw["pitch_shape"]
            if rs == "rising_falling":
                shape_fix = (f"On '{word}' (word {wnum}/{total}): this is a SVARITA syllable — "
                             f"your voice must rise then fall within the vowel (circumflex shape). "
                             f"Sing it in two movements: up then immediately down. "
                             f"You sang {SHAPE_MEANING.get(us, us)} instead.")
            elif rs == "rising":
                shape_fix = (f"On '{word}' (word {wnum}/{total}): vowel should climb upward (udatta shape). "
                             f"Let pitch rise as you hold the vowel. "
                             f"You sang {SHAPE_MEANING.get(us, us)} instead.")
            elif rs == "falling":
                shape_fix = (f"On '{word}' (word {wnum}/{total}): vowel should descend (anudatta shape). "
                             f"Let pitch gently fall through the vowel. "
                             f"You sang {SHAPE_MEANING.get(us, us)} instead.")

        items.append({"word": word, "word_num": wnum,
                      "ref_accent": rw["dominant_accent"], "user_accent": uw["dominant_accent"],
                      "ref_shape":  rw["pitch_shape"],     "user_shape":  uw["pitch_shape"],
                      "accent_ok": not accent_wrong, "shape_ok": not shape_wrong,
                      "accent_fix": accent_fix, "shape_fix": shape_fix})
    return items
# -------------------------
# 2. Rhythm similarity
# (Onset envelope — works for slow chants, no beat needed)
# -------------------------
def get_onset_envelope(y, sr):
    """
    Captures syllable attack patterns over time.
    Much better than beat_track for mantras/chants
    which have no drum beats.
    """
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    if onset_env.max() > 0:
        onset_env = onset_env / onset_env.max()
    return onset_env
# --------------------------------------------------------
# 4. Text similarity (No eSpeak / No phonemizer needed)
# --------------------------------------------------------
# -------------------------
# Clean text
# -------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# -------------------------
# Sanskrit syllable splitter
# (much better than eSpeak for mantras)
# -------------------------
def split_syllables(text):
    """
    Splits romanized Sanskrit/mantra text into syllables.
    Captures consonant clusters + vowel + optional trailing consonant.
    Works well for: om, namah, shivaya, ganapataye, etc.
    """
    return re.findall(r'[bcdfghjklmnpqrstvwxyz]*[aeiou]+[bcdfghjklmnpqrstvwxyz]*', text)

# =============================================================================
# 6. FEEDBACK ENGINE
# =============================================================================
# Generates structured, actionable feedback in Vedic chanting terminology.
# Covers all 4 dimensions: pronunciation (ucchāraṇa), pitch (svara),
# rhythm (laya), and voice quality (nāda).
#
# Feedback structure:
#   - overall_grade    : Uttama / Madhyama / Ādi / Prārambhika
#   - dimension_report : per-dimension score + what went wrong + how to fix
#   - priority_issues  : top 3 things to fix first (sorted by impact)
#   - praise           : what the user did well (keeps motivation high)
#   - sadhana_tip      : one focused practice suggestion
# =============================================================================


def score_to_grade(score):
    """Map 0-100 score to a traditional Sanskrit proficiency label."""
    if score >= 85:
        return "Uttama"          # उत्तम  — Excellent
    elif score >= 70:
        return "Madhyama"        # मध्यम  — Good / Middle
    elif score >= 50:
        return "Ādi"             # आदि    — Beginner / Foundation
    else:
        return "Prārambhika"     # प्रारंभिक — Just starting
 
 
def score_to_emoji(score):
    if score >= 85: return "✅"
    elif score >= 70: return "🔶"
    elif score >= 50: return "⚠️"
    else: return "❌"
 
 
# ---------- per-dimension feedback builders ----------------------------------
 
def pronunciation_feedback(text_sim, ref_syllables, user_syllables, ref_words, user_words):
    """
    Ucchāraṇa (उच्चारण) — Pronunciation feedback.
    Finds exactly which syllables or words differ and explains them.
    """
    issues   = []
    praises  = []
    tips     = []
 
    # --- find mismatched words ---
    matcher    = SequenceMatcher(None, ref_words, user_words)
    wrong_words = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ('replace', 'delete'):
            wrong_words.extend(ref_words[i1:i2])
    if wrong_words:
        issues.append(
            f"Incorrect ucchāraṇa (pronunciation) detected for: "
            f"'{', '.join(wrong_words[:4])}'"
            + (" and more..." if len(wrong_words) > 4 else "")
        )
 
    # --- find mismatched syllables ---
    syl_matcher   = SequenceMatcher(None, ref_syllables, user_syllables)
    wrong_syls    = []
    for tag, i1, i2, j1, j2 in syl_matcher.get_opcodes():
        if tag in ('replace', 'delete'):
            wrong_syls.extend(ref_syllables[i1:i2])
    if wrong_syls:
        issues.append(
            f"Akṣara (syllable) errors in: '{', '.join(wrong_syls[:5])}'"
            + (" and more..." if len(wrong_syls) > 5 else "")
        )
 
    # --- specific Sanskrit pronunciation tips for common errors ---
    user_flat = user_clean.replace(" ", "")
    ref_flat  = ref_clean.replace(" ", "")
 
    # Check visarga (ḥ sound) — often dropped by beginners
    if 'h' in ref_flat and 'h' not in user_flat:
        issues.append("Visarga (ḥ) sounds are missing — the soft 'h' at syllable ends must be audible")
        tips.append(
            "Visarga sādhana: practice each visarga syllable in isolation — "
            "e.g., 'namaḥ' → breathe out on the 'ḥ' like a soft echo"
        )
 
    # Check anusvara (ṃ/ṅ nasals) — often replaced with plain 'm'
    if 'ng' in ref_flat or 'nm' in ref_flat:
        if 'ng' not in user_flat and 'nm' not in user_flat:
            issues.append(
                "Anusvāra (ṃ) nasalization is missing — "
                "syllables like 'aṃ', 'oṃ' need a nasal resonance, not a hard 'm'"
            )
            tips.append(
                "Anusvāra sādhana: hum 'mmm' and feel the vibration in your skull — "
                "that nasal resonance is the anusvāra quality"
            )
 
    # Praise if text similarity is high
    if text_sim >= 75:
        praises.append("Your mantra words and syllables are clearly recognizable — shuddha ucchāraṇa (pure pronunciation) foundation is present")
 
    how_to_fix = []
    if wrong_words:
        how_to_fix.append(
            "Repeat each incorrect word 10× in isolation before rejoining it to the mantra — "
            "this is called 'pada-abhyāsa' (word-level practice)"
        )
    if wrong_syls:
        how_to_fix.append(
            "Break the mantra into individual akṣaras (syllables) and chant them with a pause between each — "
            "'krama-pāṭha' technique"
        )
    how_to_fix += tips
 
    return issues, praises, how_to_fix
 
 
def pitch_feedback(pitch_combined, pitch_similarity, zone_sim, accent_sim,
                   shape_sim, slope_sim, zone_feedback, shape_feedback,
                   word_svara_feedback=None):
    """
    Svara feedback — now word-level aware.
    Returns 4 values: issues, per_word_issues, praises, fixes
    """
    issues          = []
    per_word_issues = []  # NEW — word-by-word breakdown
    praises         = []
    fixes           = []

    # A. Overall contour
    if pitch_similarity < 50:
        issues.append("Svara-krama (melodic arc) diverges significantly from reference")
        fixes.append("Listen to reference 3x with eyes closed, trace the pitch mentally, "
                     "then hum melody on 'aa' before adding words")
    elif pitch_similarity < 75:
        issues.append("Svara-krama is partially correct but drifts in some sections")
        fixes.append("Sa-grama sadhana: hum the pitch curve on 'aa' first, then add mantra text")

    # B. Word-by-word svara breakdown  ← THE KEY NEW SECTION
    accent_errors = 0
    shape_errors  = 0
    correct_words = []

    if word_svara_feedback:
        for item in word_svara_feedback:
            if item["accent_ok"] and item["shape_ok"]:
                correct_words.append(item["word"])
                continue
            if not item["accent_ok"]:
                accent_errors += 1
            if not item["shape_ok"]:
                shape_errors += 1

            detail = f"Word {item['word_num']}: '{item['word']}'"
            if not item["accent_ok"]:
                detail += (f"\n     Accent  — expected: {item['ref_accent']}  |  "
                           f"you sang: {item['user_accent']}")
                if item["accent_fix"]:
                    detail += f"\n     Fix: {item['accent_fix']}"
            if not item["shape_ok"]:
                detail += (f"\n     Shape   — expected: {item['ref_shape']}  |  "
                           f"you sang: {item['user_shape']}")
                if item["shape_fix"]:
                    detail += f"\n     Fix: {item['shape_fix']}"
            per_word_issues.append(detail)

        if accent_errors:
            issues.append(f"Svara-sthana errors on {accent_errors} word(s) — "
                          f"wrong udatta/anudatta/svarita placement changes the meaning. "
                          f"See word-by-word breakdown below.")
        if shape_errors:
            issues.append(f"Svara-akrti errors on {shape_errors} word(s) — "
                          f"pitch shape within vowel (rising/falling/circumflex) incorrect.")
        if correct_words:
            praises.append(f"Correct svara on: {', '.join(correct_words)}")
    else:
        # Fallback if Whisper timestamps unavailable
        if accent_sim < 60:
            issues.append("Svara-sthana errors detected — udatta on wrong syllables "
                          "(word-level detail unavailable)")
            fixes.append("Learn svara markers from printed text with accent notation")

    # C. Zone-level (supplementary context)
    for fb in zone_feedback:
        if "Udatta" in fb and "not rising" in fb:
            issues.append("Udatta zone too flat — raised accent must lift above anudatta base")
            fixes.append("On each accented word (see list above): push pitch up suddenly, "
                         "not gradually. The lift must be clear.")
        elif "Svarita" in fb and "not dropping" in fb:
            issues.append("Svarita zone not falling after peak")
            fixes.append("After the peak word: let voice glide DOWN. "
                         "Like a ball thrown up — the fall is part of the accent.")
        elif "Anudatta" in fb and "unstable" in fb:
            issues.append("Anudatta region unsteady — low section before accent must be calm")
            fixes.append("Hold a drone at natural speaking pitch for 1 min — "
                         "this is your anudatta baseline.")
        elif "timing" in fb.lower():
            issues.append("Accent peak timing is off — udatta arriving too early or late")
            fixes.append("Chant at half speed with hand-clap on each syllable. "
                         "Mark the udatta clap and practice landing on it.")

    # D. Slope
    if slope_sim < 50:
        issues.append("Svara-vega (pitch transition speed) is incorrect")
        fixes.append("Mimic reference 5x focusing ONLY on how fast pitch moves, not words")

    # E. Praises
    if pitch_combined >= 75:
        praises.append("Overall svara-sadhana shows good understanding of the melodic arc")
    if zone_sim >= 80:
        praises.append("All three svara zones (anudatta, udatta, svarita) present and well-shaped")

    return issues, per_word_issues, praises, fixes  # now 4 return values
 
 
def rhythm_feedback(tempo_similarity, ref_audio, audio, sr):
    """
    Laya (लय) — Rhythm / Tempo feedback.
    """
    issues  = []
    praises = []
    fixes   = []
 
    ref_duration  = len(ref_audio) / ref_sr
    user_duration = len(audio)     / sr
    duration_ratio = user_duration / (ref_duration + 1e-9)
 
    if tempo_similarity < 50:
        if duration_ratio > 1.3:
            issues.append(
                f"Laya is too slow — your chant is {round(duration_ratio, 1)}× longer than the reference. "
                "In Vedic recitation, dragging the laya dilutes the mantra's śakti (power)"
            )
            fixes.append(
                "Use a tāla (rhythmic clap) or metronome set to the reference tempo — "
                "chant with the tāla until the laya is internalized (at least 21 repetitions)"
            )
        elif duration_ratio < 0.7:
            issues.append(
                f"Laya is too fast — your chant is compressed to {round(duration_ratio, 1)}× the reference length. "
                "Rushing (druta-laya) causes syllable merging and loss of mantra clarity"
            )
            fixes.append(
                "Practice in vilambita-laya (slow tempo) — chant at half your current speed, "
                "holding each vowel for its full mātrā (mora) count before moving on"
            )
        else:
            issues.append(
                "Laya (rhythmic flow) is irregular — syllable durations are uneven "
                "even though the overall length is close"
            )
            fixes.append(
                "Krama-pāṭha sādhana: chant with a steady hand-clap on every syllable — "
                "each clap forces equal duration for each akṣara"
            )
    elif tempo_similarity < 75:
        issues.append(
            "Minor laya deviations detected — some syllables are stretched or compressed "
            "beyond the natural mātrā (mora) proportion"
        )
        fixes.append(
            "Record yourself and listen back counting mātrās: "
            "short vowels = 1 mātrā, long vowels = 2 mātrās. "
            "Ensure this ratio is maintained throughout"
        )
    else:
        praises.append("Laya (rhythmic timing) is well-maintained — syllable flow follows the reference closely")
 
    # Check onset pattern regularity
    user_onset_local = get_onset_envelope(audio, sr)
    onset_std = float(np.std(np.diff(np.where(user_onset_local > 0.5)[0]))) if np.any(user_onset_local > 0.5) else 0
    if onset_std > 20:
        issues.append(
            "Syllable onset pattern is irregular — some syllables attack too hard or too softly, "
            "breaking the samatvam (evenness) of the chant"
        )
        fixes.append(
            "Practice 'mṛdu-pāṭha' (soft reading): chant very softly so you cannot force hard attacks — "
            "this trains even onset strength"
        )
 
    return issues, praises, fixes
 
 
def voice_quality_feedback(mfcc_similarity):
    """
    Nāda (नाद) — Voice quality / timbre feedback.
    """
    issues  = []
    praises = []
    fixes   = []
 
    if mfcc_similarity < 40:
        issues.append(
            "Nāda-guṇa (voice quality / resonance) is significantly different from the reference — "
            "the timbral character of your voice needs adjustment for this mantra"
        )
        fixes.append(
            "Warm up with 'Nāda-sādhana': hum 'mmm' for 2 minutes feeling chest resonance, "
            "then 'nnn' feeling nasal resonance, then 'ṅṅṅ' feeling skull resonance — "
            "find which resonance mode matches the reference chant"
        )
        fixes.append(
            "Check your posture: sit in sukhāsana (comfortable cross-legged position) with spine erect — "
            "slouching compresses the svara-yantra (voice box) and changes the nāda quality"
        )
    elif mfcc_similarity < 65:
        issues.append(
            "Nāda-guṇa has room for improvement — some formants (vowel resonances) "
            "do not match the reference mantra's tonal character"
        )
        fixes.append(
            "Focus on vowel openness: Sanskrit 'a' is an open, round sound (like 'aum' prefix) — "
            "do not collapse it to the English schwa 'uh'"
        )
        fixes.append(
            "Practice the mantra in a bathroom or tile room to hear your resonance clearly — "
            "a rich, ringing nāda indicates correct throat/mouth shaping"
        )
    elif mfcc_similarity < 80:
        issues.append(
            "Minor nāda inconsistencies — vowel quality drifts slightly in the middle of the chant"
        )
        fixes.append(
            "Maintain 'jihvā-mūla' (tongue root) position consistently — "
            "do not let the tongue tense or retreat mid-chant"
        )
    else:
        praises.append(
            "Nāda-guṇa (voice quality) is authentic — your resonance closely mirrors the reference chant's timbral signature"
        )
 
    if mfcc_similarity >= 70:
        praises.append("Voice consistency is good throughout the chant — no major tonal breaks detected")
 
    return issues, praises, fixes
 
 
# ---------- priority ranking -------------------------------------------------
 
def rank_priority_issues(text_sim, pitch_combined, tempo_similarity, mfcc_similarity):
    """
    Identifies which dimension needs the most work,
    ranked by impact on overall score × severity of the gap.
    """
    # weight mirrors the overall score formula
    dimensions = [
    ("Uccharana (Pronunciation)",  text_sim,         0.40),
    ("Nada (Voice Quality)",        mfcc_similarity,  0.25),
    ("Svara (Pitch / Accent)",      pitch_combined,   0.20),
    ("Laya (Rhythm)",               tempo_similarity, 0.15),
    ]
    # Store impact as explicit 4th element — prevents unstable sort ordering
    dimensions = [(name, score, weight, round(weight * (100 - score), 2))
                 for name, score, weight in dimensions]
    ranked = sorted(dimensions, key=lambda x: x[3], reverse=True)
    return ranked  # (name, score, weight, impact)
 
 
# ---------- sadhana tip generator --------------------------------------------
 
def generate_sadhana_tip(overall, text_sim, pitch_combined, tempo_similarity, mfcc_similarity):
    """
    Returns one focused daily-practice recommendation based on the biggest gap.
    """
    priority = rank_priority_issues(text_sim, pitch_combined, tempo_similarity, mfcc_similarity)
    weakest_dim, weakest_score, _, _ = priority[0]
 
    if overall >= 85:
        return (
            "Siddhi-sādhana: you are at Uttama level. Now focus on 'bhāvopāsanā' — "
            "chant with full meditative intention. Correct form + intention = complete mantra-sādhana"
        )
    elif "Pronunciation" in weakest_dim:
        return (
            "Daily sādhana: 10 minutes of 'pada-pāṭha' — chant the mantra word-by-word, "
            "pause after each word and verify it against the reference before continuing. "
            "Do this for 21 days to build correct saṃskāra (mental impression)"
        )
    elif "Svara" in weakest_dim:
        return (
            "Daily sādhana: 10 minutes of 'svara-abhyāsa' — chant only the pitch melody "
            "on a single vowel 'ā' first (no words), then overlay the mantra text. "
            "This isolates svara training from ucchāraṇa"
        )
    elif "Rhythm" in weakest_dim:
        return (
            "Daily sādhana: 10 minutes of 'tāla-pāṭha' — clap on every syllable while chanting. "
            "Use vilambita-laya (slow tempo) for the first 5 minutes, "
            "then madhya-laya (medium) for the next 5"
        )
    else:  # Nāda
        return (
            "Daily sādhana: 10 minutes of 'nāda-dhyāna' — before chanting, hum 'Aum' for 3 minutes "
            "feeling the vibration in chest, throat, and head. "
            "This opens the three resonance chambers needed for authentic mantra nāda"
        )
 
 
# ---------- master feedback function -----------------------------------------
 
def generate_full_feedback(
    text_sim, pitch_combined, tempo_similarity, mfcc_similarity,
    pitch_similarity, zone_sim, accent_sim, shape_sim, slope_sim,
    zone_feedback, shape_feedback,
    ref_syllables, user_syllables, ref_words, user_words,
    overall,
    word_svara_feedback=None
):
    """
    Master feedback generator.
    Returns a structured dict with all feedback data,
    and prints a formatted report.
    """
 
    # --- collect per-dimension feedback ---
    uc_issues, uc_praises, uc_fixes = pronunciation_feedback(
        text_sim, ref_syllables, user_syllables, ref_words, user_words
    )
    sv_issues, sv_per_word, sv_praises, sv_fixes = pitch_feedback(
        pitch_combined, pitch_similarity, zone_sim, accent_sim,
        shape_sim, slope_sim, zone_feedback, shape_feedback,
        word_svara_feedback=word_svara_feedback
    )
    la_issues, la_praises, la_fixes = rhythm_feedback(
        tempo_similarity, ref_audio, audio, sr
    )
    na_issues, na_praises, na_fixes = voice_quality_feedback(mfcc_similarity)
 
    # --- overall grade ---
    overall_grade = score_to_grade(overall)
 
    # --- priority ranking ---
    priority_dims = rank_priority_issues(text_sim, pitch_combined, tempo_similarity, mfcc_similarity)
 
    # --- sadhana tip ---
    sadhana_tip = generate_sadhana_tip(overall, text_sim, pitch_combined, tempo_similarity, mfcc_similarity)
 
    # ---------------------------------------------------------------
    # PRINT REPORT
    # ---------------------------------------------------------------
    sep = "=" * 62
 
    print(f"\n{sep}")
    print(f"  MANTRA CHANTING FEEDBACK REPORT")
    print(f"  समीक्षा प्रतिवेदन  |  Samīkṣā Prativedana")
    print(sep)
 
    print(f"\n  Overall Score  : {round(overall, 1)}%  →  {overall_grade}  {score_to_emoji(overall)}")
    print(f"\n  Mantra         : {ref_mantra}")
    print(sep)
 
    # ---- Dimension scores ----
    print("\n📊 DIMENSION SCORES")
    print(f"  Ucchāraṇa  (Pronunciation)  : {round(text_sim, 1):5.1f}%  {score_to_emoji(text_sim)}")
    print(f"  Nāda       (Voice Quality)  : {round(mfcc_similarity, 1):5.1f}%  {score_to_emoji(mfcc_similarity)}")
    print(f"  Svara      (Pitch/Accent)   : {round(pitch_combined, 1):5.1f}%  {score_to_emoji(pitch_combined)}")
    print(f"    ↳ Contour shape           : {round(pitch_similarity, 1):5.1f}%")
    print(f"    ↳ Zone (U/A/S) accuracy   : {round(zone_sim, 1):5.1f}%")
    print(f"    ↳ Accent placement        : {round(accent_sim, 1):5.1f}%")
    print(f"    ↳ Syllable shape          : {round(shape_sim, 1):5.1f}%")
    print(f"    ↳ Slope / steepness       : {round(slope_sim, 1):5.1f}%")
    print(f"  Laya       (Rhythm/Tempo)   : {round(tempo_similarity, 1):5.1f}%  {score_to_emoji(tempo_similarity)}")
 
    # ---- Priority issues ----
    print(f"\n🎯 FOCUS AREAS  (ranked by impact)")
    for i, (dim, score, weight, impact) in enumerate(priority_dims, 1):
        gap = 100 - score
        print(f"  {i}. {dim:<32} Score: {round(score,1)}%  Gap: {round(gap,1)}%  Impact: {impact}")
    # ---- What you did well ----
    all_praises = uc_praises + sv_praises + la_praises + na_praises
    if all_praises:
        print(f"\n✨ WHAT YOU DID WELL  (Sādhu! साधु!)")
        for p in all_praises:
            print(f"  ✓ {p}")
 
    # ---- Ucchāraṇa ----
    print(f"\n🔤 UCCHĀRAṆA  (Pronunciation)  —  {round(text_sim, 1)}%  {score_to_emoji(text_sim)}")
    if uc_issues:
        print("  Issues:")
        for issue in uc_issues:
            print(f"    • {issue}")
        print("  How to improve:")
        for fix in uc_fixes:
            print(f"    → {fix}")
    else:
        print("  ✓ No significant pronunciation issues detected")
 
    # ---- Svara ----
    print(f"\n🎵 SVARA  (Pitch / Vedic Accent)  —  {round(pitch_combined, 1)}%  {score_to_emoji(pitch_combined)}")
    if sv_issues:
        print("  Summary:")
        for issue in sv_issues:
            print(f"    • {issue}")
    if sv_per_word:
        print("\n  Word-by-word svara breakdown:")
        print("  " + "-" * 54)
        for detail in sv_per_word:
            for line in detail.split("\n"):
                print(f"    {line}")
            print()
    if sv_fixes:
        print("  General practice techniques:")
        for fix in sv_fixes:
            print(f"    → {fix}")
    if not sv_issues and not sv_per_word:
        print("  ✓ Svara accuracy is good — pitch accents are well-placed")
 
    # ---- Laya ----
    print(f"\n🥁 LAYA  (Rhythm / Tempo)  —  {round(tempo_similarity, 1)}%  {score_to_emoji(tempo_similarity)}")
    if la_issues:
        print("  Issues:")
        for issue in la_issues:
            print(f"    • {issue}")
        print("  How to improve:")
        for fix in la_fixes:
            print(f"    → {fix}")
    else:
        print("  ✓ Laya is steady — rhythmic flow matches the reference well")
 
    # ---- Nāda ----
    print(f"\n🔔 NĀDA  (Voice Quality / Resonance)  —  {round(mfcc_similarity, 1)}%  {score_to_emoji(mfcc_similarity)}")
    if na_issues:
        print("  Issues:")
        for issue in na_issues:
            print(f"    • {issue}")
        print("  How to improve:")
        for fix in na_fixes:
            print(f"    → {fix}")
    else:
        print("  ✓ Nāda quality is authentic — resonance matches the reference")
 
    # ---- Sadhana tip ----
    print(f"\n🪔 DAILY SĀDHANA RECOMMENDATION")
    print(f"  {sadhana_tip}")
 
    print(f"\n{sep}")
    print("  Om Shanti  🙏  |  ॐ शान्तिः")
    print(sep + "\n")
 
    # Return structured dict for programmatic use / UI display
    return {
        "overall_score" : round(overall, 2),
        "overall_grade" : overall_grade,
        "mantra"        : ref_mantra,
        "dimensions": {
            "uccharana"  : {"score": round(text_sim, 2),        "issues": uc_issues, "fixes": uc_fixes, "praises": uc_praises},
            "svara"      : {"score": round(pitch_combined, 2),  "issues": sv_issues, "fixes": sv_fixes, "praises": sv_praises},
            "laya"       : {"score": round(tempo_similarity, 2),"issues": la_issues, "fixes": la_fixes, "praises": la_praises},
            "nada"       : {"score": round(mfcc_similarity, 2), "issues": na_issues, "fixes": na_fixes, "praises": na_praises},
        },
        "priority_order"  : [d[0] for d in priority_dims],
        "sadhana_tip"     : sadhana_tip,
        "all_praises"     : all_praises,
    }
# =============================================================================
# 8. ENTRY POINT — called by api.py per request
# =============================================================================
def analyze_chant(user_audio_path: str) -> dict:
    """
    Runs the full analysis on a user audio file.
    All functions above are used as-is — nothing is changed.
    Returns the feedback_result dict for the API to serve as JSON.
    """
    _load_references()
    global audio, sr, pitch, tempo, mfcc   # make these available to feedback functions
                                            # that reference them (e.g. rhythm_feedback)

    # ── Load user chant ──
    audio, sr = librosa.load(user_audio_path)
    audio, _  = librosa.effects.trim(audio)

    # ── Pitch ──
    pitch, _, _ = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7')
    )
    pitch = np.nan_to_num(pitch)

    # ── Tempo ──
    tempo_arr, _ = librosa.beat.beat_track(y=audio, sr=sr)
    tempo = float(tempo_arr[0]) if isinstance(tempo_arr, np.ndarray) else float(tempo_arr)

    # ── MFCC ──
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

    # ── 1. Pitch similarity ──
    pitch_voiced     = pitch[pitch > 0]
    ref_pitch_voiced = ref_pitch[ref_pitch > 0]
    if len(pitch_voiced) == 0 or len(ref_pitch_voiced) == 0:
        pitch_similarity = 0.0
    else:
        pitch_log     = np.log2(pitch_voiced)
        ref_pitch_log = np.log2(ref_pitch_voiced)
        pitch_norm     = (pitch_log - np.mean(pitch_log))         / (np.std(pitch_log)         + 1e-9)
        ref_pitch_norm = (ref_pitch_log - np.mean(ref_pitch_log)) / (np.std(ref_pitch_log)     + 1e-9)
        pitch_resampled     = resample(pitch_norm,     100)
        ref_pitch_resampled = resample(ref_pitch_norm, 100)
        D, wp = dtw(pitch_resampled.reshape(1, -1), ref_pitch_resampled.reshape(1, -1), metric='euclidean')
        dtw_distance     = D[-1, -1] / len(wp)
        pitch_similarity = 100 * np.exp(-dtw_distance / 5)

    # ── 1-a Zone analysis ──
    ref_zones,  ref_norm  = extract_pitch_zones(ref_pitch)
    user_zones, user_norm = extract_pitch_zones(pitch)
    zone_sim, zone_feedback = compare_pitch_zones(ref_zones, user_zones)

    # ── 1-b Accent classifier ──
    ref_accent_levels  = classify_accent_levels(ref_pitch)
    user_accent_levels = classify_accent_levels(pitch)
    if len(ref_accent_levels) > 0 and len(user_accent_levels) > 0:
        accent_sim = SequenceMatcher(None, ref_accent_levels.tolist(), user_accent_levels.tolist()).ratio() * 100
    else:
        accent_sim = 0.0

    # ── 1-c Syllable shape ──
    ref_shapes  = analyze_syllable_shapes(ref_pitch)
    user_shapes = analyze_syllable_shapes(pitch)
    shape_sim, shape_feedback = shape_sequence_similarity(ref_shapes, user_shapes)

    # ── Slope ──
    slope_sim = pitch_slope_similarity(ref_pitch, pitch)

    # ── Combined pitch ──
    pitch_combined = (
        0.35 * pitch_similarity +
        0.25 * zone_sim         +
        0.15 * slope_sim        +
        0.15 * accent_sim       +
        0.10 * shape_sim
    )

    # ── 2. Rhythm ──
    user_onset = get_onset_envelope(audio, sr)
    ref_onset  = get_onset_envelope(ref_audio, ref_sr)
    D, wp = dtw(user_onset.reshape(1, -1), ref_onset.reshape(1, -1), metric='euclidean')
    onset_dist       = D[-1, -1] / len(wp)
    tempo_similarity = 100 * np.exp(-onset_dist / 0.3)

    # ── 3. MFCC ──
    mfcc_scaled     = StandardScaler().fit_transform(mfcc.T).T
    ref_mfcc_scaled = StandardScaler().fit_transform(ref_mfcc.T).T
    delta           = librosa.feature.delta(mfcc_scaled)
    ref_delta       = librosa.feature.delta(ref_mfcc_scaled)
    mfcc_combined     = np.vstack([mfcc_scaled, delta])
    ref_mfcc_combined = np.vstack([ref_mfcc_scaled, ref_delta])
    D, wp = dtw(mfcc_combined, ref_mfcc_combined, metric='euclidean')
    mfcc_distance   = D[-1, -1] / len(wp)
    mfcc_similarity = 100 * np.exp(-mfcc_distance / 50)

    # ── 4. Text / Whisper ──
    word_timestamps_data = []
    try:
        user_result = WHISPER_MODEL.transcribe(
            user_audio_path,
            language                   = "en",
            initial_prompt             = f"Sanskrit mantra: {ref_mantra}",
            fp16                       = False,
            temperature                = 0.0,
            best_of                    = 1,
            beam_size                  = 1,
            condition_on_previous_text = False,
            word_timestamps            = True,
        )
        user_text = user_result["text"].lower().strip()
        for segment in user_result.get("segments", []):
            for w in segment.get("words", []):
                wc = re.sub(r'[^a-z\s]', '', w["word"].lower()).strip()
                if wc:
                    word_timestamps_data.append({
                        "word":  wc,
                        "start": float(w["start"]),
                        "end":   float(w["end"]),
                    })
    except Exception as e:
        print(f"Whisper error: {e}")
        user_text = ref_text

    ref_clean  = clean_text(ref_text)
    user_clean = clean_text(user_text)
    ref_words  = ref_clean.split()
    user_words = user_clean.split()
    ref_syllables  = split_syllables(ref_clean)
    user_syllables = split_syllables(user_clean)

    char_sim = SequenceMatcher(None, ref_clean,  user_clean).ratio()
    word_sim = SequenceMatcher(None, ref_words,  user_words).ratio()
    syl_sim  = SequenceMatcher(None, ref_syllables, user_syllables).ratio()
    text_similarity = (0.20 * char_sim + 0.30 * word_sim + 0.50 * syl_sim) * 100

    # ── 5. Overall ──
    overall = (
        0.40 * text_similarity  +
        0.25 * mfcc_similarity  +
        0.20 * pitch_combined   +
        0.15 * tempo_similarity
    )

    # ── Word svara maps ──
    ref_svara_map       = build_ref_word_svara_map(ref_pitch, ref_words, ref_audio, ref_sr)
    user_svara_map      = build_word_svara_map(pitch, word_timestamps_data, sr)
    word_svara_feedback = compare_word_svara_maps(ref_svara_map, user_svara_map)

    # ── Feedback ──
    feedback_result = generate_full_feedback(
        text_similarity, pitch_combined, tempo_similarity, mfcc_similarity,
        pitch_similarity, zone_sim, accent_sim, shape_sim, slope_sim,
        zone_feedback, shape_feedback,
        ref_syllables, user_syllables, ref_words, user_words,
        overall,
        word_svara_feedback=word_svara_feedback
    )

    # ── Add extra fields the API needs ──
    feedback_result["user_transcript"]   = user_text
    feedback_result["word_timestamps"]   = word_timestamps_data
    feedback_result["svara_word_detail"] = word_svara_feedback
    feedback_result["scores"] = {
        "text_similarity" : round(text_similarity,  2),
        "pitch_combined"  : round(pitch_combined,   2),
        "tempo_similarity": round(tempo_similarity, 2),
        "mfcc_similarity" : round(mfcc_similarity,  2),
        "overall"         : round(overall,          2),
        "pitch_breakdown" : {
            "contour_shape"   : round(pitch_similarity, 2),
            "zone_accuracy"   : round(zone_sim,         2),
            "accent_placement": round(accent_sim,        2),
            "syllable_shape"  : round(shape_sim,         2),
            "slope_steepness" : round(slope_sim,         2),
        }
    }
    feedback_result["priority"] = [
        {"dimension": d, "score": round(s, 2), "impact": imp}
        for d, s, w, imp in rank_priority_issues(
            text_similarity, pitch_combined, tempo_similarity, mfcc_similarity
        )
    ]
    return feedback_result


# =============================================================================
# 9. RUN DIRECTLY (python compare_chant.py) — unchanged behaviour
# =============================================================================
if __name__ == "__main__":
    result = analyze_chant("user_chant1.wav")
    print("\n========== RAW SCORES ==========")
    print("Text Similarity  :", result["scores"]["text_similarity"],  "%")
    print("Pitch Similarity :", result["scores"]["pitch_combined"],   "%")
    print("Rhythm Similarity:", result["scores"]["tempo_similarity"], "%")
    print("Voice Similarity :", result["scores"]["mfcc_similarity"],  "%")
    print("Overall Score    :", result["scores"]["overall"],          "%")
    print("================================")