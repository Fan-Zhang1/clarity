from pathlib import Path

import librosa
import soundfile as sf
import numpy as np


WHISPER_SR = 16000

# Find corresponding clean speech
def get_clean_path(enhanced_path, clean_dir):
    """
    Example:

    Enhanced:
        S08508_L0201_E018.wav

    Clean:
        S08508_target.wav
    """

    enhanced_path = Path(enhanced_path)
    clean_dir = Path(clean_dir)

    # S08508_L0201_E018 -> S08508
    scene_id = enhanced_path.stem.split("_")[0]

    clean_path = clean_dir / f"{scene_id}_target.wav"

    if not clean_path.exists():
        raise FileNotFoundError(
            f"Cannot find clean reference:\n{clean_path}"
        )

    return clean_path


def prepare_audio_mono(audio_path):
    """
    Load audio, convert it to mono, and resample to 16 kHz.
    """

    audio, sr = librosa.load(
        audio_path,
        sr=None,
        mono=False,
    )

    print(
        f"{Path(audio_path).name}: "
        f"shape={audio.shape}, sr={sr}"
    )

    # Stereo/multichannel -> mono
    if audio.ndim == 2:
        audio = np.mean(audio, axis=0)

    # Resample for Whisper
    if sr != WHISPER_SR:
        audio = librosa.resample(
            audio,
            orig_sr=sr,
            target_sr=WHISPER_SR,
        )

    return audio