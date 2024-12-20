# Audio Transcription Script Setup and Explanation

This script splits a large audio file into smaller chunks, transcribes them using OpenAI's Whisper model, and combines the results.

## Prerequisites

1. **Python Environment**:
   - Python 3.7 or later installed.
   - Virtual environment (optional but recommended).

2. **Required Libraries**:
   Install the required libraries using `pip`:
   ```bash
   pip install whisper pydub
   ```

3. **ffmpeg Installation**: Ensure `ffmpeg` is installed on your system. You can install it via:
   - MacOS: `brew install ffmpeg`
   - Ubuntu: `sudo apt install ffmpeg`
   - Windows: Download from [ffmpeg.org](ffmpeg.org)

4. **Whisper Model**: OpenAI's Whisper model is used for transcription. This script loads the "tiny" model for faster processing. Ensure your environment supports the necessary dependencies.

## Script Instructions

1. **Input Audio**: Place your audio file in the directory and update the `input_file` variable in the script.

2. **Run the Script**: Execute the script using:
   ```bash
   python transcribe_script.py
   ```

3. **Output**:
   - Transcription is saved as `output_transcription.txt` by default.
   - Temporary files are cleaned up automatically.

## Features

- Converts audio to WAV format for compatibility.
- Splits large audio files into manageable chunks.
- Retries transcription for each chunk to ensure reliability.
- Saves progress incrementally to avoid data loss.

## Known Limitations

- The script does not support real-time serving or a local test environment for audio transcription.
- Processing times may vary based on hardware and the Whisper model size.