import os
import sys
from pathlib import Path
import soundfile as sf
import numpy as np
import random

from dt_scw_chain.app import processing_log
from dt_scw_chain.constants import Constant
# Removed dsp.shuffle import since we'll use frame-order shuffling

print("DEBUG: Skrypt batch_convert_scw.py został uruchomiony.")

SCW_TARGET_SAMPLERATE = Constant.DEFAULT_SAMPLERATE  # 48000 Hz
SCW_TARGET_SAMPLES = 367


def process_single_frame_for_chain(audio_data: np.ndarray, fade_samples: int = 5) -> np.ndarray:
    """
    Process a single audio waveform: mono conversion, resizing to SCW_TARGET_SAMPLES,
    normalization and fade in/out.
    """
    if audio_data.ndim > 1 and audio_data.shape[1] > 1:
        audio_data = audio_data.mean(axis=1)
    original_len = len(audio_data)
    if original_len != SCW_TARGET_SAMPLES:
        audio_data = np.interp(
            np.linspace(0, original_len - 1, SCW_TARGET_SAMPLES),
            np.arange(original_len),
            audio_data
        )
        processing_log.append(f"Resized SCW from {original_len} to {SCW_TARGET_SAMPLES} samples.")
    peak = np.abs(audio_data).max()
    if peak > 0:
        audio_data = audio_data / peak
    if fade_samples > 0:
        window = np.ones_like(audio_data)
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        window[:fade_samples] *= fade_in
        window[-fade_samples:] *= fade_out
        audio_data *= window
    return audio_data.astype(np.float32)


def process_directory_recursively(input_dir: str, output_dir: str):
    """
    Recursively convert all WAV files in input_dir to SCW and save to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    found = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith('.wav'):
                found.append(Path(root) / f)
    if not found:
        print(f"No WAV files in {input_dir}")
        return
    for path in sorted(found):
        rel = path.relative_to(input_dir)
        outp = Path(output_dir) / rel
        outp.parent.mkdir(parents=True, exist_ok=True)
        data, sr = sf.read(path, dtype='float32')
        if sr != SCW_TARGET_SAMPLERATE:
            data = np.interp(
                np.linspace(0, len(data)-1, int(len(data) * SCW_TARGET_SAMPLERATE / sr)),
                np.arange(len(data)),
                data
            )
            processing_log.append(f"Resampled {path.name} from {sr}Hz to {SCW_TARGET_SAMPLERATE}Hz.")
        proc = process_single_frame_for_chain(data)
        sf.write(outp, proc, SCW_TARGET_SAMPLERATE)
        processing_log.append(f"Converted {path.name}")
    print("Individual conversion complete.")


def create_wavetable_chain(
    input_path_for_chain: str,
    output_folder: str,
    output_filename: str = "digitakt_wavetable_chain.wav",
    target_num_frames: int = 64,
    target_scw_samples: int = SCW_TARGET_SAMPLES,
    fade_samples: int = 5,
    perform_shuffle: bool = False,
    shuffle_seed: int = None,
) -> None:
    """
    Create a wavetable chain from WAV files in a directory, optionally shuffling frame order.
    """
    processing_log.clear()
    os.makedirs(output_folder, exist_ok=True)
    out_path = Path(output_folder) / output_filename
    src = Path(input_path_for_chain)
    files = sorted(src.rglob("*.wav"))
    if len(files) < target_num_frames:
        print(f"WARNING: Only {len(files)} files found in '{src}'. Expected {target_num_frames}.")
    # Process frames
    frames = []
    for path in files[:target_num_frames]:
        try:
            data, sr = sf.read(path, dtype='float32')
            if sr != SCW_TARGET_SAMPLERATE:
                data = np.interp(
                    np.linspace(0, len(data)-1, int(len(data)*SCW_TARGET_SAMPLERATE/sr)),
                    np.arange(len(data)), data
                )
                processing_log.append(f"Resampled {path.name} to {SCW_TARGET_SAMPLERATE}Hz.")
            proc = process_single_frame_for_chain(data, fade_samples)
            if len(proc) == target_scw_samples:
                frames.append(proc)
                processing_log.append(f"Processed {path.name}")
        except Exception as e:
            processing_log.append(f"ERROR processing {path.name}: {e}")
    if not frames:
        print("No valid frames to chain.")
        return
    # Shuffle frame order, not sample-level
    if perform_shuffle:
        seed = shuffle_seed if shuffle_seed is not None else random.randint(0, 2**32 - 1)
        print(f"Shuffling frame order with seed {seed}...")
        rnd = random.Random(seed)
        rnd.shuffle(frames)
        processing_log.append(f"Shuffled frame order with seed {seed}.")
    # Concatenate and save
    chain = np.concatenate(frames)
    sf.write(out_path, chain, SCW_TARGET_SAMPLERATE)
    print(f"Saved chain to {out_path}")
    print("--- Chain Log ---")
    for entry in processing_log:
        print(f"- {entry}")


if __name__ == "__main__":
    print("\n--- Digitakt SCW/Wavetable Tool ---")
    print("1. Convert individual SCW Waveforms to SCW format")
    print("2. Create WAVetable Chains from subdirectories")
    choice = input("Select mode (1/2): ").strip()
    if choice == '1':
        inp = input("Input dir (default: input_scw_frames): ").strip() or "input_scw_frames"
        out = input("Output dir (default: output_digitakt_scw): ").strip() or "output_digitakt_scw"
        process_directory_recursively(inp, out)
    elif choice == '2':
        inp = input("Root dir of SCW frame subfolders (default: input_scw_frames): ").strip() or "input_scw_frames"
        out = input("Output dir for chains (default: output_digitakt_chains): ").strip() or "output_digitakt_chains"
        root = Path(inp)
        # Find all subdirectories (including nested) in root directory
        subs = [d for d in root.rglob("*") if d.is_dir()]
        if not subs:
            print(f"No subdirectories in '{inp}'.")
            sys.exit(0)
        print(f"Found {len(subs)} subdirectories:")
        for i, d in enumerate(subs, 1):
            print(f" {i}. {d.name}")
        m = input("(a) first or (b) all? ").strip().lower()
        selected = [subs[0]] if m == 'a' else subs
        shuffle_flag = input("Shuffle frame order? (y/n): ").strip().lower() in ['y', 'yes']
        seed = None
        if shuffle_flag:
            # Prompt for number of groups (not used in frame shuffle but kept for compatibility)
            while True:
                g = input("Enter number of groups for shuffle (this will not affect frame-level shuffle, press Enter to skip): ").strip()
                if g == "":
                    break
                try:
                    groups = int(g)
                    print(f"Groups parameter set to {groups} (not used in frame shuffle)")
                    break
                except ValueError:
                    print("Invalid input. Please enter an integer or press Enter.")
            s = input("Enter shuffle seed (0=default constant, empty=random): ").strip()
            if s == '0':
                seed = Constant.DEFAULT_SEED
            elif s:
                try:
                    seed = int(s)
                except ValueError:
                    seed = random.randint(0, 2**32 - 1)
            else:
                seed = random.randint(0, 2**32 - 1)
        for sub in selected:
            fname = f"{sub.name}.wav"
            print(f"Processing '{sub.name}' -> '{fname}'")
            create_wavetable_chain(
                str(sub), out, fname,
                perform_shuffle=shuffle_flag,
                shuffle_seed=seed
            )
    else:
        print("Invalid mode. Exiting.")
