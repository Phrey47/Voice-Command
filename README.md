# Voice-Command

## Requirements
- Python 3.x
- Libraries:
  pip install sounddevice scipy librosa numpy noisereduce



## How to Use

### 1. Record Command Samples - kung ganahan mo, pwede kamo mismo mo record sa inyo commands
python main.py  
Select option: 1  
Enter command name (either "open" or "close")  
* Speak the command clearly, hilom na room
* 3 seconds record time ra, i sulti ang word na "open" or "close" after makita ang "Recording..."

### 2. Record Owner Voice Samples
python main.py  
Select option: 2  
Enter sample name (e.g., "owner1" or "owner2")  
Say the phrase consistently every time:
"Access granted for voice authentication."
* I-dali ug sulti and access granted for voice authe kay 3 seconds ra gihapon ni

### 3. Test the System
python main.py  
Select option: 3  
Speak one of the recorded command words in your normal voice 
*try ug sulti ug "open" or "close". Kung ganahon mo, icheck inyong gi record diri sa test.wav
*inconsistent pani, usahay ma recognize ang voice, usahay dili ma recognize ang word, basta libug

