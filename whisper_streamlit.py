import os
from pathlib import Path
import openai
import pydub
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Replace with your OpenAI API key
api_key = os.getenv('OPENAI_API_KEY')

# Initialize the OpenAI API client
openai.api_key = api_key

def split_audio_file(input_file_path, chunk_duration_ms=60000):
    """Splits the audio file into smaller chunks"""
    audio = pydub.AudioSegment.from_file(input_file_path)

    chunks = [audio[i:i + chunk_duration_ms] for i in range(0, len(audio), chunk_duration_ms)]
    chunk_files = []
    for i, chunk in enumerate(chunks):
        chunk_file_path = input_file_path.with_name(f"{input_file_path.stem}_chunk_{i}{input_file_path.suffix}")
        chunk.export(chunk_file_path, format="mp3")
        chunk_files.append(chunk_file_path)
    return chunk_files

def transcribe_audio(audio_file_path, output_file_path):
    audio_file_path = Path(audio_file_path)  # Ensure audio_file_path is a Path object
    if not audio_file_path.exists():
        st.error(f"Error: The file {audio_file_path} does not exist.")
        return
    try:
        chunk_files = split_audio_file(audio_file_path)
        transcription_texts = []

        for chunk_file in chunk_files:
            with open(chunk_file, 'rb') as audio_file:
                # Create transcription from audio file chunk
                transcription = openai.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
                transcription_texts.append(transcription.text)
        
        # Combine transcriptions and save to a text file
        with open(output_file_path, 'w') as output_file:
            for text in transcription_texts:
                output_file.write(text + "\n")
        
        st.success(f"Transcription saved to {output_file_path}")

        # Delete chunk files after transcription
        for chunk_file in chunk_files:
            os.remove(chunk_file)

    except openai.OpenAIError as e:
        st.error(f"OpenAI API error: {e}")

def main():
    st.title("Audio Transcription with OpenAI")

    audio_file = st.file_uploader("Upload an MP3 file", type=["mp3"])

    if audio_file is not None:
        file_path = Path("uploaded_audio.mp3")
        with open(file_path, "wb") as f:
            f.write(audio_file.getbuffer())
        
        st.write("File uploaded successfully. Ready to transcribe.")
        
        if st.button("Transcribe"):
            output_file_path = file_path.with_suffix('.txt')
            transcribe_audio(file_path, output_file_path)
            
            with open(output_file_path, 'r') as f:
                transcription = f.read()
            
            st.text_area("Transcription", transcription)
            st.download_button(
                    label="Download Transcription",
                    data=transcription,
                    file_name=output_file_path.name,
                    mime="text/plain"
                )

if __name__ == "__main__":
    main()
