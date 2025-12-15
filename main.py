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
FIXED_LEN = SAMPLE_RATE * 1  # 1 second

COMMAND_DIR = "commands"
OWNER_DIR = "owner"

os.makedirs(COMMAND_DIR, exist_ok=True)
os.makedirs(OWNER_DIR, exist_ok=True)

# ==============================
# RECORD AUDIO
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

    audio = audio.flatten()
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.9

    wav.write(filename, SAMPLE_RATE, np.int16(audio * 32767))
    print(f"Saved: {filename}")

# ==============================
# PREPROCESS AUDIO
# ==============================
def preprocess_audio(file):
    y, sr = librosa.load(file, sr=SAMPLE_RATE, mono=True)

    y, _ = librosa.effects.trim(y, top_db=35)
    y = librosa.util.fix_length(y, size=FIXED_LEN)

    rms = np.sqrt(np.mean(y ** 2))
    if rms > 0:
        y = y / rms

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
        np.std(mfcc, axis=1),
        np.mean(delta, axis=1),
        np.std(delta, axis=1),
        np.mean(delta2, axis=1),
        np.std(delta2, axis=1)
    ])

    return feat / (np.linalg.norm(feat) + 1e-8)

# ==============================
# COSINE DISTANCE
# ==============================
def cosine_distance(f1, f2):
    return 1.0 - np.dot(f1, f2)

# ==============================
# DISTANCE → SIMILARITY
# ==============================
def distance_to_similarity(d):
    return 100 / (1 + np.exp(10 * (d - 0.3)))

# ==============================
# BUILD COMMAND MODELS
# ==============================
def build_command_models():
    models = {}

    for file in os.listdir(COMMAND_DIR):
        cmd = file.split("_")[0]
        path = os.path.join(COMMAND_DIR, file)
        feat = extract_features(path)
        models.setdefault(cmd, []).append(feat)

    for cmd in models:
        models[cmd] = np.mean(models[cmd], axis=0)

    return models

# ==============================
# COMMAND DETECTION
# ==============================
def detect_command(input_file, models, threshold=0.45):
    input_feat = extract_features(input_file)

    best_cmd = None
    best_dist = float("inf")

    for cmd, centroid in models.items():
        dist = cosine_distance(input_feat, centroid)
        if dist < best_dist:
            best_dist = dist
            best_cmd = cmd.upper()

    sim = distance_to_similarity(best_dist)
    print(f"Command={best_cmd} | similarity={sim:.1f}% | distance={best_dist:.3f}")

    if best_dist > threshold:
        return None

    return best_cmd

# ==============================
# BUILD OWNER MODEL
# ==============================
def build_owner_model():
    feats = []

    for file in os.listdir(OWNER_DIR):
        path = os.path.join(OWNER_DIR, file)
        feats.append(extract_features(path))

    if len(feats) == 0:
        raise RuntimeError("No owner samples found")

    centroid = np.mean(feats, axis=0)
    return centroid, feats

# ==============================
# SPEAKER VERIFICATION
# ==============================
def verify_speaker(input_file, owner_centroid, owner_feats, margin=0.08):
    input_feat = extract_features(input_file)

    d_owner = cosine_distance(input_feat, owner_centroid)
    intra = [cosine_distance(owner_centroid, f) for f in owner_feats]

    threshold = np.mean(intra) + margin

    sim = distance_to_similarity(d_owner)
    req_sim = distance_to_similarity(threshold)

    print(
        f"Speaker similarity={sim:.1f}% | required ≥ {req_sim:.1f}% "
        f"(distance={d_owner:.3f})"
    )

    return d_owner < threshold

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
                name = input("Command name (or b): ").lower()
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
            if not os.listdir(COMMAND_DIR) or not os.listdir(OWNER_DIR):
                print("❌ Please record command and owner samples first.")
                continue

            cmd_models = build_command_models()
            owner_centroid, owner_feats = build_owner_model()

            print("Speak now...")
            record_voice("test.wav")

            cmd = detect_command("test.wav", cmd_models)
            if cmd is None:
                print("❌ Command rejected")
                continue

            if verify_speaker("test.wav", owner_centroid, owner_feats):
                print(f"✅ ACCESS GRANTED → {cmd}")
            else:
                print("❌ SPEAKER REJECTED")

        elif choice == "q":
            break

# ==============================
# ENTRY POINT
# ==============================
if __name__ == "__main__":
    main()
