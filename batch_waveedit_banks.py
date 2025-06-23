import os
import subprocess
import soundfile as sf
import numpy as np
from scipy.signal import resample
import random

# Import DSP functions directly.
try:
    from dt_scw_chain.dsp import fade, maximize, shuffle, processing_log 
    from dt_scw_chain.constants import Constant 
except ImportError as e:
    print(f"Error importing dt_scw_chain DSP modules: {e}")
    print("Please ensure dt_scw_chain is correctly installed in editable mode (pip install -e .) in your active venv.")
    print("If the problem persists, try running the script from the parent directory")
    print("  python okwt/batch_waveedit_banks.py")
    exit(1)

# --- CONFIGURATION ---
input_directory = './input_waveedit_banks/'
output_directory = './output_waveedit_banks/'

# Target wavetable parameters for Digitakt
TARGET_NUM_FRAMES = Constant.DIGITAKT_WAVETABLE_NUM_FRAMES # 64
TARGET_FRAME_SIZE = 367
TARGET_SAMPLERATE = Constant.DEFAULT_SAMPLERATE # 48000 Hz

# Original WaveEdit file parameters (needed for correct resampling)
ORIGINAL_SAMPLERATE = 44100
ORIGINAL_TOTAL_SAMPLES = 16384 # 64 frames * 256 samples/frame

# Allowed input file extensions (case-insensitive)
ALLOWED_EXTENSIONS = ('.wav', '.mp3', '.aif', '.flac', '.ogg')

# --- Processing options (set True/False or values) ---
# Each processed bank will be normalized and faded by default for consistency.
APPLY_FADE = True
FADE_SAMPLES = 5 # Number of samples for fade in/out on each frame

# These will be set by user interaction or default if no interaction
APPLY_SHUFFLE_INTERACTIVE = False 
SHUFFLE_GROUPS_INTERACTIVE = 0 
SHUFFLE_SEED_INTERACTIVE = None 


def convert_to_wav(input_path: str, output_path: str, target_sr: int) -> bool:
    """
    Converts any audio format to a standard WAV file using ffmpeg.
    Args:
        input_path: Path to the input audio file.
        output_path: Path where the WAV file will be saved.
        target_sr: Target sample rate for the output WAV.
    Returns:
        True if conversion is successful, False otherwise.
    """
    command = [
        'ffmpeg', '-i', input_path,
        '-ar', str(target_sr),
        '-ac', '1', # Convert to mono
        '-f', 'wav',
        output_path
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        processing_log.append(f"Converted '{os.path.basename(input_path)}' to WAV.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error converting '{input_path}' to WAV: {e}")
        processing_log.append(f"ERROR: Failed to convert '{input_path}' to WAV: {e}")
        return False
    except FileNotFoundError:
        print("Error: ffmpeg not found. Please install ffmpeg or ensure it's in your PATH.")
        processing_log.append("ERROR: ffmpeg not found.")
        return False


def process_single_file(
    input_file: str,
    output_file: str,
    apply_shuffle_choice: bool,
    shuffle_groups_val: int,
    shuffle_seed_val: int | None
) -> bool:
    """
    Processes a single wavetable bank file.
    1. Converts to WAV if not already.
    2. Resamples to target samplerate and frame size.
    3. Applies DSP (normalize, fade, shuffle).
    4. Saves the processed wavetable.

    Args:
        input_file: Path to the input file.
        output_file: Path for the output processed WAV file.
        apply_shuffle_choice: Boolean to enable/disable shuffling.
        shuffle_groups_val: Number of groups for shuffling.
        shuffle_seed_val: Seed for shuffling.

    Returns:
        True if processing is successful, False otherwise.
    """
    temp_wav_path = None
    try:
        file_extension = os.path.splitext(input_file)[1].lower()

        if file_extension != '.wav':
            temp_wav_path = input_file + '.temp.wav'
            if not convert_to_wav(input_file, temp_wav_path, ORIGINAL_SAMPLERATE):
                return False
            input_audio_path = temp_wav_path
        else:
            input_audio_path = input_file

        audio_data, samplerate = sf.read(input_audio_path, dtype='float32')

        # Ensure mono
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)
            processing_log.append("Converted multi-channel audio to mono.")

        # Resample to target samplerate and total samples
        if samplerate != TARGET_SAMPLERATE or len(audio_data) != (TARGET_NUM_FRAMES * TARGET_FRAME_SIZE):
            num_target_samples = TARGET_NUM_FRAMES * TARGET_FRAME_SIZE
            audio_data = resample(audio_data, num_target_samples)
            processing_log.append(f"Resampled audio from {samplerate}Hz ({len(audio_data)} samples) to {TARGET_SAMPLERATE}Hz ({num_target_samples} samples).")

        # Apply DSP
        audio_data = maximize(audio_data, TARGET_FRAME_SIZE)
        processing_log.append("Maximized overall audio volume.")

        if APPLY_FADE:
            frames = audio_data.reshape(-1, TARGET_FRAME_SIZE)
            faded_frames = []
            fade_in_ramp = np.linspace(0, 1, FADE_SAMPLES)
            fade_out_ramp = np.linspace(1, 0, FADE_SAMPLES)

            for i, frame in enumerate(frames):
                if len(frame) >= FADE_SAMPLES * 2:
                    frame[:FADE_SAMPLES] *= fade_in_ramp
                    frame[len(frame) - FADE_SAMPLES:] *= fade_out_ramp
                elif FADE_SAMPLES > 0 and len(frame) > 0:
                     # Fallback for very short frames
                    frame[:] *= np.linspace(0, 1, len(frame))
                    processing_log.append(f"Warning: Frame {i} too short for full {FADE_SAMPLES}-sample fade. Applied full-frame fade.")
                faded_frames.append(frame)
            audio_data = np.array(faded_frames, dtype=np.float32).flatten()
            processing_log.append(f"Applied {FADE_SAMPLES}-sample fade to each frame.")

        if apply_shuffle_choice:
            audio_data = shuffle(audio_data, TARGET_FRAME_SIZE, shuffle_groups_val, shuffle_seed_val)
            processing_log.append(f"Shuffled frames (groups: {shuffle_groups_val}, seed: {shuffle_seed_val}).")

        # Save the processed WAV file
        sf.write(output_file, audio_data, TARGET_SAMPLERATE, subtype='PCM_16')
        processing_log.append(f"Saved processed file: '{os.path.basename(output_file)}'.")
        return True

    except Exception as e:
        print(f"Error processing '{os.path.basename(input_file)}': {e}")
        processing_log.append(f"ERROR: Failed to process '{os.path.basename(input_file)}': {e}")
        return False
    finally:
        if temp_wav_path and os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)


def main():
    """Main function to iterate through input directory and process files."""
    global APPLY_SHUFFLE_INTERACTIVE, SHUFFLE_GROUPS_INTERACTIVE, SHUFFLE_SEED_INTERACTIVE

    # --- Interactive Shuffle Configuration ---
    while True:
        shuffle_input = input("Apply shuffle? (yes/no): ").lower().strip()
        if shuffle_input in ['yes', 'y']:
            APPLY_SHUFFLE_INTERACTIVE = True
            break
        elif shuffle_input in ['no', 'n']:
            APPLY_SHUFFLE_INTERACTIVE = False
            break
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")

    if APPLY_SHUFFLE_INTERACTIVE:
        while True:
            try:
                groups_input = input("Enter shuffle groups (0 for individual frames): ").strip()
                SHUFFLE_GROUPS_INTERACTIVE = int(groups_input)
                break
            except ValueError:
                print("Invalid input. Please enter an integer.")

        while True:
            seed_input = input("Enter shuffle seed (integer, or 'random' for a new seed): ").strip()
            if seed_input.lower() == 'random':
                SHUFFLE_SEED_INTERACTIVE = random.randint(0, 2**32 - 1)
                print(f"Using generated random seed: {SHUFFLE_SEED_INTERACTIVE}")
                break
            try:
                SHUFFLE_SEED_INTERACTIVE = int(seed_input)
                break
            except ValueError:
                print("Invalid input. Please enter an integer or 'random'.")
    else:
        SHUFFLE_GROUPS_INTERACTIVE = 0 # Ensure default if shuffle is off
        SHUFFLE_SEED_INTERACTIVE = None # Ensure default if shuffle is off

    os.makedirs(output_directory, exist_ok=True)

    all_files = [f for f in os.listdir(input_directory) if os.path.isfile(os.path.join(input_directory, f))]
    files_to_process = [f for f in all_files if f.lower().endswith(ALLOWED_EXTENSIONS)]

    if not files_to_process:
        print(f"No allowed ({', '.join(ALLOWED_EXTENSIONS)}) files found in the input directory '{input_directory}'.")
        return

    print(f"Found {len(files_to_process)} files to process.")

    for i, filename in enumerate(files_to_process):
        base_filename = os.path.splitext(filename)[0]
        # Format filename as 001_originalname.wav
        new_filename_with_original_name = f"{i + 1:03d}_{base_filename}.wav"
        input_file_path = os.path.join(input_directory, filename)
        output_final_file_path = os.path.join(output_directory, new_filename_with_original_name)

        print(f"\n--- Processing [{i+1}/{len(files_to_process)}]: '{filename}' -> '{new_filename_with_original_name}' ---")

        success = process_single_file(input_file_path, output_final_file_path, APPLY_SHUFFLE_INTERACTIVE, SHUFFLE_GROUPS_INTERACTIVE, SHUFFLE_SEED_INTERACTIVE)

        if success:
            print(f"Successfully processed '{filename}'.")
        else:
            print(f"Failed to process '{filename}'. See logs for details.")

    if processing_log:
        print("\n--- DSP Processing Log ---")
        for entry in processing_log:
            print(f"- {entry}")


if __name__ == "__main__":
    main()