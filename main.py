import sounddevice as sd
import scipy.io.wavfile as wav
import librosa
import numpy as np
import os
import noisereduce as nr

# ==============================
# CONFIG
# ==============================
SAMPLE_RATE = 16000
DURATION = 3  # seconds

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
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )
    sd.wait()
    wav.write(filename, SAMPLE_RATE, audio)
    print(f"Saved: {filename}")

# ==============================
# PREPROCESS AUDIO
# ==============================
def preprocess_audio(file):
    y, sr = librosa.load(file, sr=SAMPLE_RATE)

    # Trim silence
    y, _ = librosa.effects.trim(y, top_db=20)

    # Noise reduction
    y = nr.reduce_noise(y=y, sr=sr)

    # Normalize volume
    y = librosa.util.normalize(y)

    return y, sr

# ==============================
# FEATURE EXTRACTION
# ==============================
def extract_features(file):
    y, sr = preprocess_audio(file)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

# ==============================
# SIMILARITY METRIC
# ==============================
def similarity(f1, f2):
    return np.linalg.norm(f1 - f2)

# ==============================
# COMMAND DETECTION
# ==============================
def detect_command(input_file):
    input_feat = extract_features(input_file)

    best_match = None
    best_distance = float("inf")

    for file in os.listdir(COMMAND_DIR):
        path = os.path.join(COMMAND_DIR, file)
        feat = extract_features(path)
        dist = similarity(input_feat, feat)

        if dist < best_distance:
            best_distance = dist
            best_match = file.replace(".wav", "").upper()

    print(f"Detected Command: {best_match}  (distance={best_distance:.2f})")
    return best_match, best_distance

# ==============================
# SPEAKER VERIFICATION
# ==============================
def verify_speaker(input_file, threshold=80):
    input_feat = extract_features(input_file)

    for file in os.listdir(OWNER_DIR):
        path = os.path.join(OWNER_DIR, file)
        feat = extract_features(path)
        dist = similarity(input_feat, feat)

        print(f"Distance to {file}: {dist:.2f}")

        if dist < threshold:
            print("Speaker Verified ✅")
            return True

    print("Speaker Rejected ❌")
    return False

# ==============================
# MAIN MENU
# ==============================
def main():
    print("\nVoice Command + Voice Authentication System")
    print("1 - Record Command Samples")
    print("2 - Record Owner Voice Samples")
    print("3 - Live Test System")

    choice = input("Select option: ")

    if choice == "1":
        name = input("Command name (open / close): ").lower()
        record_voice(os.path.join(COMMAND_DIR, f"{name}.wav"))

    elif choice == "2":
        name = input("Owner sample name (owner1 / owner2): ").lower()
        record_voice(os.path.join(OWNER_DIR, f"{name}.wav"))

    elif choice == "3":
        print("\nSpeak a command clearly (OPEN or CLOSE)")
        record_voice("test.wav")

        command, _ = detect_command("test.wav")
        authorized = verify_speaker("test.wav")

        print("\n==============================")
        if authorized:
            print(f"ACCESS GRANTED → EXECUTING COMMAND: {command}")
        else:
            print("ACCESS DENIED!")
        print("==============================")

    else:
        print("Invalid option.")

if __name__ == "__main__":
    main()
