import librosa
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Union
import typer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = typer.Typer()

def calculate_mfcc(audio_path: str) -> Dict:
    """Calculate MFCCs using parameters aligned with Essentia implementation.
    
    Parameters:
    - Sampling rate: 44100 Hz
    - FFT size: 2048
    - Window size: 2048
    - Hop length: 512
    - Window type: Hann
    - Number of MFCCs: 13
    - Number of Mel bands: 40
    - Min frequency: 0 Hz
    - Max frequency: Nyquist frequency (sr/2)
    - Power: Energy (2.0)
    """
    logging.info(f"Processing file: {audio_path}")
    
    # Load audio with specific sampling rate
    y, sr = librosa.load(audio_path, sr=44100)
    
    # Calculate mel spectrogram with specified parameters
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=2048,
        win_length=2048,
        hop_length=512,
        window='hann',
        n_mels=40,
        fmin=0.0,
        fmax=sr/2,
        power=2.0
    )
    
    # Convert to log mel spectrogram
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Calculate MFCCs
    mfcc = librosa.feature.mfcc(
        S=log_mel_spec,  # Use pre-computed log mel spectrogram
        n_mfcc=13
    )
    
    return {
        "filename": audio_path,
        "mfcc": mfcc
    }

def process_audio_files(input_path: Union[str, Path]) -> List[Dict]:
    input_path = Path(input_path)
    logging.info(f"Start processing: {input_path}")
    results = []

    if input_path.is_file():
        results.append(calculate_mfcc(str(input_path)))
    else:
        audio_files = sorted([f for f in input_path.glob('*') if f.suffix.lower() in ['.wav', '.mp3', '.flac']])
        for file_path in audio_files:
            results.append(calculate_mfcc(str(file_path)))

    logging.info(f"Processing completed. {len(results)} files processed.")
    return results

@app.command()
def main(input_path: str):
    return process_audio_files(input_path)

if __name__ == "__main__":
    app()