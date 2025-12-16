import sounddevice as sd
import scipy.io.wavfile as wav
import librosa
import numpy as np
import os
import time

# ==============================
# CONFIG
# ==============================
SAMPLE_RATE = 16000
RECORD_SECONDS = 3
PASSPHRASES = {"open", "close"}
COMMAND_DIR = "commands"
os.makedirs(COMMAND_DIR, exist_ok=True)

# ==============================
# RECORD AUDIO
# ==============================
def record_voice(filename):
    print("Recording...")
    audio = sd.rec(int(RECORD_SECONDS * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    audio = audio.flatten()
    audio = audio / (np.max(np.abs(audio)) + 1e-9)
    wav.write(filename, SAMPLE_RATE, np.int16(audio * 32767))
    print(f"Saved: {filename}")

# ==============================
# BASIC SIGNAL GATES
# ==============================
def energy_gate(y):
    # Temporarily disabled for testing
    return True

def duration_gate(y, sr):
    duration = len(y) / sr
    # Loosened duration gate to accept short speech
    return duration >= 0.1

# ==============================
# PREPROCESS AUDIO
# ==============================
def preprocess_audio(file):
    y, sr = librosa.load(file, sr=SAMPLE_RATE, mono=True)
    # Trim very soft silence, but don't remove too much
    y, _ = librosa.effects.trim(y, top_db=40)

    if len(y) < 50:
        return None, None  # too short after trimming

    if not energy_gate(y) or not duration_gate(y, sr):
        return None, None

    return y, sr

# ==============================
# EXTRACT MFCC + DELTA FEATURES
# ==============================
def extract_mfcc(file):
    y, sr = preprocess_audio(file)
    if y is None:
        return None

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # Compute delta with safe width
    width = min(3, mfcc.shape[1] // 2 * 2 + 1)  # must be odd and <= #frames
    delta = librosa.feature.delta(mfcc, width=width)

    mfcc = np.vstack([mfcc, delta])

    # Normalize per coefficient
    mfcc = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / (
        np.std(mfcc, axis=1, keepdims=True) + 1e-8
    )

    return mfcc.T  # frames x features

# ==============================
# DTW DISTANCE (NO NORMALIZATION)
# ==============================
def dtw_distance(mfcc1, mfcc2):
    D, _ = librosa.sequence.dtw(X=mfcc1.T, Y=mfcc2.T, metric='euclidean')
    return D[-1, -1]

# ==============================
# COMMAND DETECTION
# ==============================
def detect_command(file, abs_threshold=4000, margin_ratio=0.8):
    test_mfcc = extract_mfcc(file)
    if test_mfcc is None:
        print("Rejected: energy/duration gate")
        return None

    distances = []
    templates = {}

    for cmd_file in os.listdir(COMMAND_DIR):
        cmd_name = cmd_file.split("_")[0]
        path = os.path.join(COMMAND_DIR, cmd_file)
        templates.setdefault(cmd_name, []).append(extract_mfcc(path))

    for cmd_name, mfcc_list in templates.items():
        valid_dists = []
        for tmpl in mfcc_list:
            if tmpl is None:
                continue
            valid_dists.append(dtw_distance(test_mfcc, tmpl))

        if valid_dists:
            distances.append((cmd_name, min(valid_dists)))

    distances.sort(key=lambda x: x[1])
    print("DTW distances:", distances)

    if len(distances) < 2:
        return None

    best_cmd, best_dist = distances[0]
    second_dist = distances[1][1]

    # Debug print
    print(f"Best: {best_dist:.1f}, Second: {second_dist:.1f}, Ratio: {best_dist/second_dist:.2f}")

    # Absolute rejection temporarily disabled for calibration
    # if best_dist > abs_threshold:
    #     return None

    # Relative margin rejection
    if best_dist / second_dist > margin_ratio:
        return None

    return best_cmd

# ==============================
# MAIN LOOP
# ==============================
def main():
    while True:
        print("\nVOICE COMMAND DETECTION (MFCC + DTW + REJECTION)")
        print("1 - Record command samples")
        print("2 - Test system")
        print("q - Quit")
        choice = input("Choice: ").lower()

        if choice == "1":
            while True:
                name = input("Command name (open/close or b): ").lower()
                if name == "b":
                    break
                if name not in PASSPHRASES:
                    print("Only open or close allowed")
                    continue
                record_voice(os.path.join(COMMAND_DIR, f"{name}_{int(time.time())}.wav"))

        elif choice == "2":
            record_voice("test.wav")
            cmd = detect_command("test.wav")
            if cmd is None:
                print("❌ INVALID COMMAND")
            else:
                print(f"✅ COMMAND DETECTED → {cmd.upper()}")

        elif choice == "q":
            break

if __name__ == "__main__":
    main()
