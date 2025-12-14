# Voice-Command

## Requirements
- Python 3.x
- Libraries:
  pip install sounddevice scipy librosa numpy noisereduce



## How to Use

### 1. Record Command Samples 
**You may record your own voice commands if you prefer.**
Run:
- python main.py  

Select option: 1  
Enter command name (either "open" or "close").

Instructions:
- Speak clearly in a quiet room
- Recording duration is 3 seconds
- Say the word “open” or “close” after the “Recording…” message appears

### 2. Record Owner Voice Samples
**This step records authorized speaker samples.**
Run:
- python main.py  

Select option: 2  
Enter sample name (e.g., "owner1" or "owner2").

Say the phrase consistently every time:
**"Access granted for voice authentication."**

Notes:
- Recording duration is 3 seconds
- Speak clearly and at a steady pace 


### 3. Test the System
**This step tests both command recognition and speaker authentication.**
Run:
- python main.py  

Select option: 3  
Instructions:
- Speak one of the recorded command words (open or close) using your normal voice
- You may review the recorded audio in test.wav if needed

Important Note:
- The system may sometimes recognize the command but not the voice
- Other times, the voice may be recognized but not the command
- This behavior is expected due to noise, recording conditions, and short recording duration



