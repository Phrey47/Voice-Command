import sounddevice as sd
import scipy.io.wavfile as wav
import librosa
import numpy as np
import os
import noisereduce as nr

SAMPLE_RATE = 16000
DURATION = 3  # seconds

# ------------------------------
# RECORD VOICE
# ------------------------------
def record_voice(filename):
    print("Recording...")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()
    wav.write(filename, SAMPLE_RATE, audio)
    print("Saved:", filename)

# ------------------------------
# PREPROCESSING + NOISE FILTER
# ------------------------------
def preprocess_audio(file):
    y, sr = librosa.load(file, sr=16000)
    
    # Remove silence
    y, _ = librosa.effects.trim(y, top_db=20)

    # Normalize volume
    y = librosa.util.normalize(y)

    # Noise reduction
    y = nr.reduce_noise(y=y, sr=sr)

    return y, sr

# ------------------------------
# FEATURE EXTRACTION (MFCC)
# ------------------------------
def extract_features(file):
    y, sr = preprocess_audio(file)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

# ------------------------------
# DISTANCE CALCULATION
# ------------------------------
def similarity(f1, f2):
    return np.linalg.norm(f1 - f2)

# ------------------------------
# WORD DETECTION
# ------------------------------
def detect_command(input_file):
    input_feat = extract_features(input_file)
    best_match = None
    lowest_dist = float("inf")

    for file in os.listdir("commands"):
        command_path = os.path.join("commands", file)
        command_feat = extract_features(command_path)

        dist = similarity(input_feat, command_feat)

        if dist < lowest_dist:
            lowest_dist = dist
            best_match = file.replace(".wav", "")

    print("Detected Command:", best_match)
    return best_match

# ------------------------------
# VOICE AUTHENTICATION
# ------------------------------
def verify_speaker(input_file, threshold=60):
    input_feat = extract_features(input_file)

    for file in os.listdir("owner"):
        owner_path = os.path.join("owner", file)
        owner_feat = extract_features(owner_path)

        dist = similarity(input_feat, owner_feat)
        print(f"Distance to {file}: {dist:.2f}")

        if dist < threshold:
            print("Speaker Verified ✅")
            return True

    print("Speaker Rejected ❌")
    return False

# ------------------------------
# MAIN SYSTEM
# ------------------------------
print("1 - Record Command Samples")
print("2 - Record Owner Voice Samples")
print("3 - Test System")
choice = input("Select Option: ")

if choice == "1":
    name = input("Command name: ")
    record_voice(f"commands/{name}.wav")

elif choice == "2":
    name = input("Owner sample name: ")
    record_voice(f"owner/{name}.wav")

elif choice == "3":
    test_file = "test.wav"
    record_voice(test_file)

    command = detect_command(test_file)
    authorized = verify_speaker(test_file)

    if authorized:
        print(f"Executing command: {command}")
    else:
        print("Access Denied.")

else:
    print("Invalid option.")
