import whisper
import subprocess
import os
from pydub import AudioSegment
import time
from pytube import YouTube


def download_youtube_audio(url, output_path):
    try:
        print("Downloading YouTube audio with yt-dlp...")
        subprocess.run(["yt-dlp", "-f", "bestaudio", "-o", output_path, url], check=True)
        print("YouTube audio downloaded successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error downloading YouTube audio: {e}")
        return False


def convert_to_wav(input_path, output_path):
    """Convert audio to WAV format"""
    try:
        print("Converting audio to WAV...")
        command = ["ffmpeg", "-i", input_path, "-ar", "16000", "-ac", "1", "-vn", output_path]
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300, check=True)
        print("Conversion to WAV completed.")
        return True
    except subprocess.TimeoutExpired:
        print("Conversion timed out.")
        return False
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg error: {e}")
        return False

def split_audio(input_file, chunk_duration=180000):  # 3 minutes in milliseconds
    """Split audio file into smaller chunks"""
    print("Loading audio file...")
    audio = AudioSegment.from_file(input_file)
    chunks = []
    
    for i in range(0, len(audio), chunk_duration):
        chunk = audio[i:i + chunk_duration]
        chunk_path = f"chunk_{i//chunk_duration}.wav"
        print(f"Saving chunk {i//chunk_duration}...")
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)
    
    return chunks

def transcribe_chunk(model, file_path):
    """Transcribe a single audio chunk"""
    try:
        print(f"Transcribing {file_path}...")
        result = model.transcribe(
            file_path,
            fp16=False,
            language='en',
            task='transcribe',
            temperature=0,
            best_of=1,
            beam_size=1
        )
        print(f"Transcription for {file_path} completed.")
        return result['text']
    except Exception as e:
        print(f"Error transcribing chunk {file_path}: {e}")
        return "[Transcription Failed]"

def main():
    # Get YouTube URL from user
    youtube_url = input("Enter YouTube video URL: ")
    
    # Set up file paths
    temp_audio = "temp_youtube_audio"
    temp_wav = "temp_audio.wav"
    output_file = "youtube_transcription.txt"
    
    # Clean up any existing temporary files
    print("Cleaning up existing temporary files...")
    for f in os.listdir('.'):
        if f.startswith('chunk_') and f.endswith('.wav'):
            os.remove(f)
    for f in [temp_audio, temp_wav]:
        if os.path.exists(f):
            os.remove(f)
            
    # Download YouTube audio
    if not download_youtube_audio(youtube_url, temp_audio):
        print("Failed to download YouTube audio. Exiting.")
        return

    # Convert to WAV
    if not convert_to_wav(temp_audio, temp_wav):
        print("Error converting file. Exiting.")
        os.remove(temp_audio)
        return

    # Load Whisper model
    print("Loading Whisper model (tiny) on CPU...")
    model = whisper.load_model("tiny", device='cpu')
    print("Model loaded successfully.")

    # Split into chunks
    print("Splitting audio into chunks...")
    chunks = split_audio(temp_wav)
    print(f"Total chunks created: {len(chunks)}")

    # Transcribe each chunk
    full_transcription = []
    for i, chunk in enumerate(chunks):
        print(f"Transcribing chunk {i+1}/{len(chunks)}...")
        
        # Add retry mechanism
        max_retries = 3
        for attempt in range(max_retries):
            try:
                text = transcribe_chunk(model, chunk)
                if text != "[Transcription Failed]":
                    break
            except Exception as e:
                print(f"Attempt {attempt + 1} failed for {chunk}: {e}")
                if attempt < max_retries - 1:
                    print("Retrying in 5 seconds...")
                    time.sleep(5)
                else:
                    print("Max retries reached. Moving to the next chunk.")
                    text = "[Transcription Failed]"
        
        full_transcription.append(text)
        
        # Save progress after each chunk
        with open(output_file, "w", encoding='utf-8') as f:
            f.write("\n".join(full_transcription))
        
        # Clean up chunk file
        os.remove(chunk)
        time.sleep(1)
        print(f"Chunk {i+1} processed and cleaned up.")

    # Clean up temporary files
    print("Cleaning up temporary files...")
    for f in [temp_audio, temp_wav]:
        if os.path.exists(f):
            os.remove(f)
    print("Cleanup completed.")

    print(f"Transcription completed and saved to: {output_file}")

if __name__ == "__main__":
    main()