import os
import random
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import librosa

# Configure logging
logger = logging.getLogger(__name__)

# Supported audio extensions for 'Bring Your Own' datasets
SUPPORTED_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".webm", ".aiff"}

# Default SNR levels mimicking the design philosophy
SNR_LEVELS = [20, 10, 5, 0]

def mix_at_snr(speech: np.ndarray, noise: np.ndarray, target_snr_db: float) -> tuple[np.ndarray, dict]:
    """Mix speech and noise at a specific SNR level."""
    # Ensure noise is at least as long as speech
    if len(noise) < len(speech):
        repeats = int(np.ceil(len(speech) / len(noise)))
        noise = np.tile(noise, repeats)
    
    # Take a random window of noise
    start_idx = 0 if len(noise) == len(speech) else random.randint(0, len(noise) - len(speech))
    noise_segment = noise[start_idx : start_idx + len(speech)]

    rms_speech = np.sqrt(np.mean(speech**2))
    rms_noise = np.sqrt(np.mean(noise_segment**2))
    
    # Avoid division by zero
    if rms_noise == 0:
        rms_noise = 1e-10
    if rms_speech == 0:
        rms_speech = 1e-10

    target_rms_noise = rms_speech / (10 ** (target_snr_db / 20))
    scale_factor = target_rms_noise / rms_noise
    
    scaled_noise = noise_segment * scale_factor
    mixed = speech + scaled_noise
    
    clipping_occurred = bool(np.any(np.abs(mixed) > 1.0))
    mixed = np.clip(mixed, -1.0, 1.0)

    # Actual SNR check
    actual_snr_db = 20 * np.log10(rms_speech / np.sqrt(np.mean(scaled_noise**2) + 1e-10))

    stats = {
        "target_snr_db": target_snr_db,
        "actual_snr_db": round(float(actual_snr_db), 2),
        "clipping_occurred": clipping_occurred,
        "scale_factor": float(scale_factor)
    }

    return mixed, stats

def load_audio(path: Path, target_sr: int = 16000) -> np.ndarray:
    """Load audio file and convert to mono at target sample rate."""
    y, _ = librosa.load(path, sr=target_sr, mono=True)
    return y

def write_toml_manifest(manifest_path: Path, entries: list[dict]):
    """Generate the TOML manifest for Tier 2 'Bring Your Own' format."""
    with open(manifest_path, "w") as f:
        f.write("# echotrace-fixtures.toml\n")
        f.write("# Generated from local .flac/.wav open datasets\n\n")
        
        for entry in entries:
            f.write("[[clips]]\n")
            f.write(f'id = "{entry["id"]}"\n')
            f.write(f'audio = "{entry["audio"]}"\n')
            if entry.get("reference"):
                # Escape quotes in reference
                ref = entry["reference"].replace('"', '\\"')
                f.write(f'reference = "{ref}"\n')
            f.write("\n")

def generate_dataset(
    speech_dir: Path,
    out_dir: Path,
    noise_dir: Optional[Path] = None,
    sample_rate: int = 16000,
    max_clips: int = 10,
) -> None:
    """
    Generate synthetic dataset from local unmixed audio files.
    Creates noisy mixtures and a valid `echotrace-fixtures.toml` manifest.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_out_dir = out_dir / "mixed"
    audio_out_dir.mkdir(exist_ok=True)

    # Find speech files
    speech_files = [
        p for p in speech_dir.rglob("*") 
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    ]
    if not speech_files:
        logger.error(f"No valid audio files found in {speech_dir}")
        raise ValueError(f"No valid audio files found in {speech_dir}")
    
    # Shuffle and limit
    random.shuffle(speech_files)
    speech_files = speech_files[:max_clips]

    # Find noise files
    noise_files = []
    if noise_dir and noise_dir.exists():
        noise_files = [
            p for p in noise_dir.rglob("*") 
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
        ]
    
    if not noise_files:
        logger.info("No noise files provided or found. Falling back to synthetic white noise generation.")

    manifest_entries = []

    for i, speech_path in enumerate(speech_files):
        logger.info(f"Processing speech [{i+1}/{len(speech_files)}]: {speech_path.name}")
        speech_audio = load_audio(speech_path, target_sr=sample_rate)

        # Truncate silence
        speech_audio, _ = librosa.effects.trim(speech_audio, top_db=20)

        # Decide which noise to use for this speech clip
        noise_name = "synthetic_white"
        if noise_files:
            noise_path = random.choice(noise_files)
            noise_name = noise_path.stem
            noise_audio = load_audio(noise_path, target_sr=sample_rate)
        else:
            # Generate local white noise
            noise_audio = np.random.normal(0, 1, len(speech_audio)).astype(np.float32)
            noise_audio = noise_audio / np.max(np.abs(noise_audio))

        # We will generate a mix for each SNR
        for snr in SNR_LEVELS:
            out_filename = f"{speech_path.stem}_mixed_with_{noise_name}_snr{snr}.wav"
            out_filepath = audio_out_dir / out_filename

            mixed_audio, stats = mix_at_snr(speech_audio, noise_audio, snr)

            # Write generated mix
            sf.write(out_filepath, mixed_audio, sample_rate, subtype="PCM_16")

            # Try to grab reference text (e.g. from LibriSpeech .txt transcript in same folder)
            reference = None
            transcript_file = speech_path.parent / f"{speech_path.parent.name}.trans.txt" # Common in LibriSpeech
            if transcript_file.exists():
                with open(transcript_file, "r") as f:
                    for line in f:
                        if line.startswith(speech_path.stem):
                            reference = line[len(speech_path.stem):].strip()
                            break
            
            # Form final metadata
            manifest_entries.append({
                "id": f"{speech_path.stem}_{snr}dB",
                "audio": f"mixed/{out_filename}",
                "reference": reference or "No reference transcript found."
            })

    # Write Manifest
    manifest_path = out_dir / "echotrace-fixtures.toml"
    write_toml_manifest(manifest_path, manifest_entries)

    logger.info(f"Generated {len(manifest_entries)} fixtures at {out_dir}")
    logger.info(f"Run tests with: echotrace benchmark --fixtures {manifest_path}")
