# ------------------------------
# 1️⃣ Install required library
# ------------------------------
!pip install noisereduce

# ------------------------------
# 2️⃣ Import libraries
# ------------------------------
import librosa
import numpy as np
import noisereduce as nr
import os

# ------------------------------
# 3️⃣ Audio processing functions
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

def extract_features(file):
    y, sr = preprocess_audio(file)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

def similarity(f1, f2):
    return np.linalg.norm(f1 - f2)

# ------------------------------
# 4️⃣ Command detection
# ------------------------------
def detect_command(input_file):
    input_feat = extract_features(input_file)
    best_match = None
    lowest_dist = float("inf")

    commands = [
        "Wav Files/hum_open.wav",
        "Wav Files/hum_close.wav"
    ]

    for file in commands:
        command_feat = extract_features(file)
        dist = similarity(input_feat, command_feat)
        if dist < lowest_dist:
            lowest_dist = dist
            best_match = os.path.basename(file).replace("hum_", "").replace(".wav", "")

    print(f"\nDetected Command: {best_match.upper()}  (distance={lowest_dist:.2f})")
    return best_match

# ------------------------------
# 5️⃣ Speaker verification
# ------------------------------
def verify_speaker(input_file, threshold=100):
    input_feat = extract_features(input_file)

    owners = [
        "Wav Files/hum_auth.wav",
        "Wav Files/owner_owner2.wav"
    ]

    for file in owners:
        owner_feat = extract_features(file)
        dist = similarity(input_feat, owner_feat)
        print(f"Distance to {os.path.basename(file)}: {dist:.2f}")
        if dist < threshold:
            print("\nSpeaker Verified ✅")
            return True

    print("\nSpeaker Rejected ❌")
    return False

# ------------------------------
# 6️⃣ Main execution
# ------------------------------
print("Running Voice Command + Voice Authentication System...")

test_file = "Wav Files/test.wav"  # Your uploaded test file

command = detect_command(test_file)
authorized = verify_speaker(test_file)

print("\n==============================")
if authorized:
    print(f"ACCESS GRANTED → EXECUTING COMMAND: {command.upper()}")
else:
    print("ACCESS DENIED!")
print("==============================")
