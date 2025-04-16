import librosa
import numpy as np
import speech_recognition as sr

import os
import subprocess
from moviepy import VideoFileClip

def extract_audio(video_path, audio_output_path):
    # Convert .webm to .mp4 temporarily
    converted_path = video_path.rsplit('.', 1)[0] + '_converted.mp4'
    try:
        subprocess.run([
            'ffmpeg', '-y', '-i', video_path, '-c:v', 'libx264', '-c:a', 'aac', converted_path
        ], check=True)

        # Now extract audio from the .mp4
        clip = VideoFileClip(converted_path)
        clip.audio.write_audiofile(audio_output_path)

        # Optional: clean up
        clip.close()
        os.remove(converted_path)

    except subprocess.CalledProcessError as e:
        print("FFmpeg conversion failed:", e)
        raise

def analyze_voice_confidence(audio_path):
    # Load audio with librosa
    y, sr_val = librosa.load(audio_path)

    # Feature 1: RMS Energy
    rms = librosa.feature.rms(y=y).mean()
    # Typical RMS values may range from 0 to 0.1 in a normalized signal; adjust scaling accordingly
    energy_score = min(rms / 0.1, 1.0) * 100

    # Feature 2: Spectral Flux (changes in energy across frames)
    S = np.abs(librosa.stft(y))
    spectral_flux = np.mean(np.diff(S, axis=1)**2)
    # Normalize (tune factor after your observations; here we assume a max around 1e7)
    flux_score = min(spectral_flux / 1e7, 1.0) * 100

    # Feature 3: Zero Crossing Rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    # Normalization: lower zcr is better in speech; here we invert it on a rough scale (if typical rates around 0.1)
    zcr_score = max(100 - (zcr / 0.1 * 100), 0)

    # Feature 4: Pitch Variability
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr_val)
    pitch_values = pitches[pitches > 0]
    if pitch_values.size > 0:
        pitch_std = np.std(pitch_values)
    else:
        pitch_std = 0
    # Assume a std above 50 Hz is good (tune this based on data)
    pitch_score = min(pitch_std / 50, 1.0) * 100

    # Weighted combination — adjust weights based on experiments.
    final_voice_score = round(
        (energy_score * 0.3) + (flux_score * 0.25) + (zcr_score * 0.2) + (pitch_score * 0.25),
        2
    )

    return final_voice_score