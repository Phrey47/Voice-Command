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
FRAME_LEN = 0.03  # 30 ms
FRAME_STEP = 0.01 # 10 ms
PASSPHRASES = {"open", "close"}

COMMAND_DIR = "commands"
OWNER_DIR = "owner"

os.makedirs(COMMAND_DIR, exist_ok=True)
os.makedirs(OWNER_DIR, exist_ok=True)

# ==============================
# RECORD AUDIO
# ==============================
def record_voice(filename):
    print("Recording...")
    audio = sd.rec(int(RECORD_SECONDS*SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    audio = audio.flatten()
    audio = audio / (np.max(np.abs(audio)) + 1e-9)
    wav.write(filename, SAMPLE_RATE, np.int16(audio*32767))
    print(f"Saved: {filename}")

# ==============================
# PREPROCESS AUDIO
# ==============================
def preprocess_audio(file):
    y, sr = librosa.load(file, sr=SAMPLE_RATE, mono=True)
    y, _ = librosa.effects.trim(y, top_db=20)
    return y, sr

# ==============================
# FRAME-LEVEL FEATURE EXTRACTION
# ==============================
def frame_features(y, sr):
    frame_size = int(FRAME_LEN*sr)
    step_size = int(FRAME_STEP*sr)
    frames = []

    for start in range(0, len(y)-frame_size, step_size):
        frame = y[start:start+frame_size]
        # LPC
        lpc_order = 8
        try:
            lpc_coeffs = librosa.lpc(frame, order=lpc_order)
        except np.linalg.LinAlgError:
            lpc_coeffs = np.zeros(lpc_order+1)
        # Pitch
        f0 = librosa.yin(frame, fmin=80, fmax=400)
        f0 = np.nan_to_num(f0)
        pitch_mean = np.mean(f0)
        # Energy
        energy = np.sum(frame**2)
        feat = np.concatenate([lpc_coeffs[1:], [pitch_mean], [energy]])
        frames.append(feat)
    return np.array(frames)

# ==============================
# CORRELATION BETWEEN FRAMES
# ==============================
def frame_correlation(frames1, frames2):
    min_len = min(len(frames1), len(frames2))
    corr_vals = []
    for i in range(min_len):
        f1 = frames1[i]
        f2 = frames2[i]
        # normalized correlation
        if np.linalg.norm(f1)==0 or np.linalg.norm(f2)==0:
            corr = 0
        else:
            corr = np.dot(f1,f2)/(np.linalg.norm(f1)*np.linalg.norm(f2))
        corr_vals.append(corr)
    return np.mean(corr_vals)

# ==============================
# COMMAND DETECTION
# ==============================
def detect_command(file, threshold=0.25):
    y, sr = preprocess_audio(file)
    test_frames = frame_features(y, sr)

    best_cmd = None
    best_corr = -1
    for cmd_file in os.listdir(COMMAND_DIR):
        path = os.path.join(COMMAND_DIR, cmd_file)
        y_c, sr_c = preprocess_audio(path)
        cmd_frames = frame_features(y_c, sr_c)
        corr = frame_correlation(test_frames, cmd_frames)
        if corr > best_corr:
            best_corr = corr
            best_cmd = cmd_file.split("_")[0]
    print(f"Command detected: {best_cmd} | correlation={best_corr:.3f}")
    if best_corr < threshold:
        return None
    return best_cmd

# ==============================
# SPEAKER VERIFICATION
# ==============================
def verify_speaker(file, command, threshold=0.4):
    y, sr = preprocess_audio(file)
    test_frames = frame_features(y, sr)

    best_corr = -1
    for owner_file in os.listdir(OWNER_DIR):
        if not owner_file.startswith(command): 
            continue
        y_o, sr_o = preprocess_audio(os.path.join(OWNER_DIR, owner_file))
        owner_frames = frame_features(y_o, sr_o)
        corr = frame_correlation(test_frames, owner_frames)
        print(f"Correlation with {owner_file}: {corr:.3f}")
        if corr > best_corr:
            best_corr = corr
    print(f"Best correlation: {best_corr:.3f}")
    return best_corr >= threshold

# ==============================
# MAIN LOOP
# ==============================
def main():
    while True:
        print("\nVOICE COMMAND + SPEAKER RECOGNITION (Frame Correlation S&S)")
        print("1 - Record command samples")
        print("2 - Record owner samples")
        print("3 - Test system")
        print("q - Quit")
        choice = input("Choice: ").lower()

        if choice=="1":
            while True:
                name = input("Command name (open/close or b): ").lower()
                if name=="b": break
                if name not in PASSPHRASES: 
                    print("Only open or close allowed"); continue
                record_voice(os.path.join(COMMAND_DIR,f"{name}_{int(time.time())}.wav"))

        elif choice=="2":
            while True:
                name = input("Owner sample name (open/close or b): ").lower()
                if name=="b": break
                if name not in PASSPHRASES: 
                    print("Only open or close allowed"); continue
                record_voice(os.path.join(OWNER_DIR,f"{name}_{int(time.time())}.wav"))

        elif choice=="3":
            record_voice("test.wav")
            cmd = detect_command("test.wav")
            if cmd is None:
                print("❌ INVALID COMMAND")
                continue
            if verify_speaker("test.wav", cmd):
                print(f"✅ ACCESS GRANTED → {cmd.upper()}")
            else:
                print("❌ SPEAKER REJECTED")

        elif choice=="q":
            break

if __name__=="__main__":
    main()
