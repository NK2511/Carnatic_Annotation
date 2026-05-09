import os
import whisper
import warnings

# Suppress FP16 warning
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")

def transcribe_files(file_list, model_size="base"):
    """
    Transcribes a list of audio files and saves the transcript to a .txt file.
    """
    
    # Check for ffmpeg and add to PATH if needed
    # Check for ffmpeg and add to PATH if needed
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ffmpeg_dir = os.path.join(script_dir, "..", "demucs_env", "Scripts")
    if os.path.exists(os.path.join(ffmpeg_dir, "ffmpeg.exe")):
        os.environ["PATH"] += os.pathsep + os.path.abspath(ffmpeg_dir)

    print(f"Loading Whisper model ('{model_size}')...")
    try:
        model = whisper.load_model(model_size)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    for audio_path in file_list:
        if not os.path.exists(audio_path):
            print(f"File not found, skipping: {audio_path}")
            continue
            
        print(f"\nProcessing: {audio_path}")
        
        # Check if transcript already exists
        transcript_file = os.path.splitext(audio_path)[0] + "_transcript.txt"
        if os.path.exists(transcript_file):
            print(f"Transcript already exists: {transcript_file}")
            # Uncomment the next line if you want to skip existing transcripts
            # continue 

        try:
            print("Transcribing... (this make take a few minutes)")
            result = model.transcribe(audio_path, verbose=False)
            text = result['text']
            
            with open(transcript_file, "w", encoding="utf-8") as f:
                f.write(text.strip())
            
            print(f"Saved transcript to: {transcript_file}")
            
        except Exception as e:
            print(f"Error transcribing {audio_path}: {e}")

if __name__ == "__main__":
    # LIST OF FILES TO TRANSCRIBE
    # You can add or remove files from this list
    files_to_transcribe = [
        "../downloaded_mp3s/Mahabharat_ep_2_Of Boons and Curses.mp3",
        "../downloaded_mp3s/Mahabharat_ep_3_Bhisma.mp3",
        "../downloaded_mp3s/Mahabharat_ep_4_Entry of Krishna.mp3",
        "../downloaded_mp3s/Mahabharat_ep_5_The Poison Of Hate.mp3",
        "../downloaded_mp3s/Mahabharat_ep_6_Five Husbands.mp3",
        "../downloaded_mp3s/Mahabharat_ep_7_Indraprastha The City of Woe.mp3",
        "../downloaded_mp3s/Mahabharat_ep_8_The Gamble Where Humanity Was Lost.mp3",
        "../downloaded_mp3s/Mahabharat_ep_9_Vanvasa Parv.mp3",
        "../downloaded_mp3s/Mahabharat_ep_10_Agnathavasa Parv.mp3",
        "../downloaded_mp3s/Mahabharat_ep_11_Kurukshetra Beyond Fair and Unfair.mp3",
        "../downloaded_mp3s/Mahabharat_ep_12_Dance Of Destiny.mp3",
        "../downloaded_mp3s/Mahabharat_ep_13_Vanaprastha - End of an Era.mp3",
        "../downloaded_mp3s/Mahabharat_ep_14_Dharma Adharma.mp3",
        "../downloaded_mp3s/Mahabharat_ep_15_Living through the Story.mp3",
        "../downloaded_mp3s/Mahabharat_ep_16_Karna-The Fate's Child.mp3",
    ]

    transcribe_files(files_to_transcribe)
    print("\nAll tasks completed.")
