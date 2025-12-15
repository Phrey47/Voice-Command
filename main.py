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
FIXED_LEN = SAMPLE_RATE * 1  # 1 second after trimming

COMMAND_DIR = "commands"
OWNER_DIR = "owner"

os.makedirs(COMMAND_DIR, exist_ok=True)
os.makedirs(OWNER_DIR, exist_ok=True)

# ==============================
# RECORD AUDIO (PCM 16-bit)
# ==============================
def record_voice(filename):
    print("Recording...")
    audio = sd.rec(
        int(RECORD_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )
    sd.wait()

    # Convert float32 → PCM16 (CRITICAL FIX)
    audio = audio.flatten()
    audio = audio / (np.max(np.abs(audio)) + 1e-9)
    audio_int16 = np.int16(audio * 32767)

    wav.write(filename, SAMPLE_RATE, audio_int16)
    print(f"Saved: {filename}")

# ==============================
# PREPROCESS AUDIO
# ==============================
def preprocess_audio(file):
    y, sr = librosa.load(file, sr=SAMPLE_RATE, mono=True)

    # Trim silence
    y, _ = librosa.effects.trim(y, top_db=20)

    # Fix length (consistency)
    y = librosa.util.fix_length(y, size=FIXED_LEN)

    # Normalize
    y = librosa.util.normalize(y)

    return y, sr

# ==============================
# FEATURE EXTRACTION
# ==============================
def extract_features(file):
    y, sr = preprocess_audio(file)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=20,
        n_fft=512,
        hop_length=160
    )

    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    feat = np.concatenate([
        np.mean(mfcc, axis=1),
        np.mean(delta, axis=1),
        np.mean(delta2, axis=1)
    ])

    # L2 normalize
    return feat / (np.linalg.norm(feat) + 1e-8)


# ==============================
# COSINE DISTANCE
# ==============================
def cosine_distance(f1, f2):
    return 1.0 - np.dot(f1, f2)

# ==============================
# COMMAND DETECTION (MULTI-SAMPLE)
# ==============================
def detect_command(input_file, threshold=0.45):
    input_feat = extract_features(input_file)

    best_cmd = None
    best_dist = float("inf")

    for cmd_file in os.listdir(COMMAND_DIR):
        path = os.path.join(COMMAND_DIR, cmd_file)
        feat = extract_features(path)
        dist = cosine_distance(input_feat, feat)

        if dist < best_dist:
            best_dist = dist
            best_cmd = cmd_file.split("_")[0].upper()

    confidence = max(0.0, 1.0 - best_dist)

    print(f"Command: {best_cmd} | distance={best_dist:.3f} | confidence={confidence:.2f}")

    if best_dist > threshold:
        return None, best_dist

    return best_cmd, best_dist

# ==============================
# SPEAKER VERIFICATION
# ==============================
def verify_speaker(input_file, threshold=0.40):
    input_feat = extract_features(input_file)

    distances = []

    for owner_file in os.listdir(OWNER_DIR):
        path = os.path.join(OWNER_DIR, owner_file)
        feat = extract_features(path)
        dist = cosine_distance(input_feat, feat)
        distances.append(dist)
        print(f"Distance to {owner_file}: {dist:.3f}")

    best = min(distances)
    avg = np.mean(distances)

    print(f"Best={best:.3f} | Avg={avg:.3f}")

    return best < threshold

# ==============================
# MAIN LOOP
# ==============================
def main():
    while True:
        print("\nVoice Command + Speaker Verification")
        print("1 - Record Command Samples")
        print("2 - Record Owner Samples")
        print("3 - Test System")
        print("q - Quit")

        choice = input("Choice: ").lower()

        if choice == "1":
            while True:
                name = input("Command name (open/close or b): ").lower()
                if name == "b":
                    break
                filename = os.path.join(
                    COMMAND_DIR, f"{name}_{int(time.time())}.wav"
                )
                record_voice(filename)

        elif choice == "2":
            while True:
                name = input("Owner sample name (or b): ").lower()
                if name == "b":
                    break
                filename = os.path.join(
                    OWNER_DIR, f"{name}_{int(time.time())}.wav"
                )
                record_voice(filename)

        elif choice == "3":
            print("Speak command clearly...")
            record_voice("test.wav")

            cmd, _ = detect_command("test.wav")
            if cmd is None:
                print("❌ Command rejected")
                continue

            if verify_speaker("test.wav"):
                print(f"✅ ACCESS GRANTED → {cmd}")
            else:
                print("❌ SPEAKER REJECTED")

        elif choice == "q":
            break

if __name__ == "__main__":
    main()
    3