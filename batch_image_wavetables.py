import numpy as np
from PIL import Image
from pathlib import Path
import os
import sys
import random
import math 

# Import DSP functions and Constants
try:
    from dt_scw_chain.dsp import (
        fade, # This fade is for images/frames, not audio
        flip,
        interpolate,
        invert_phase,
        maximize, # This maximize is for image data
        normalize, # This normalize is for image data
        processing_log,
        resize,
        reverse,
        shuffle, # This shuffle is for image frames
        sort,
        trim,
    )
    from dt_scw_chain.constants import Constant
    from dt_scw_chain.formats import DIGITAKT_IMAGE_TARGET_FRAME_SIZE
    from dt_scw_chain.utils import write_wav
except ImportError as e:
    print(f"Error importing dt_scw_chain modules: {e}")
    print("Please ensure dt_scw_chain is correctly installed in editable mode (pip install -e .) in your active venv.")
    print("If the problem persists, try running the script from the parent directory")
    print("  python dt_scw_chain/batch_image_wavetables.py")
    sys.exit(1)


# --- CONFIGURATION ---
input_directory_name = "input_images"
output_directory_name = "output_image_wavetables"
TARGET_NUM_FRAMES = Constant.DIGITAKT_WAVETABLE_NUM_FRAMES # 64
TARGET_SAMPLERATE = Constant.DEFAULT_SAMPLERATE # 48000 Hz

# Interactive options will override these defaults
APPLY_SHUFFLE_INTERACTIVE = False
SHUFFLE_GROUPS_INTERACTIVE = 0
SHUFFLE_SEED_INTERACTIVE = None


def process_image(
    image_path: Path,
    output_wav_path: Path,
    target_frame_size: int,
    target_num_frames: int,
    apply_shuffle_choice: bool,
    shuffle_groups: int,
    shuffle_seed: int | None
) -> bool:
    """
    Processes a single image file into a Digitakt-compatible wavetable.
    Converts image to grayscale, resizes, normalizes, applies DSP, and saves as WAV.

    Args:
        image_path: Path to the input image file.
        output_wav_path: Path where the output WAV file will be saved.
        target_frame_size: Desired frame size for the wavetable (e.g., 367).
        target_num_frames: Desired number of frames for the wavetable (e.g., 64).
        apply_shuffle_choice: Boolean to enable/disable shuffling.
        shuffle_groups: Number of groups for shuffling.
        shuffle_seed: Seed for shuffling.

    Returns:
        True if processing is successful, False otherwise.
    """
    try:
        # Load image and convert to grayscale (L mode)
        image = Image.open(image_path)
        image = image.convert("L") # Convert to grayscale

        # Convert image to numpy array, normalize to -1.0 to 1.0 range
        img_array = np.array(image, dtype=np.float32) / 127.5 - 1.0

        original_height, original_width = img_array.shape

        # Resample frames (rows) to TARGET_NUM_FRAMES
        # And resize each frame (row) to TARGET_FRAME_SIZE
        # Combined interpolation as done in DSP resize and interpolate for images
        output_wavetable_data = np.zeros((target_num_frames, target_frame_size), dtype=np.float32)

        for i in range(target_num_frames):
            # Interpolate rows
            src_row_idx_float = (i / (target_num_frames - 1)) * (original_height - 1) if target_num_frames > 1 else 0
            row1_idx = math.floor(src_row_idx_float)
            row2_idx = min(math.ceil(src_row_idx_float), original_height - 1)

            if original_height == 1:
                current_image_row = img_array[0, :]
            elif row1_idx == row2_idx:
                current_image_row = img_array[row1_idx, :]
            else:
                alpha = src_row_idx_float - row1_idx
                current_image_row = (1 - alpha) * img_array[row1_idx, :] + \
                                    alpha * img_array[row2_idx, :]

            # Interpolate samples within the frame (from original_width to target_frame_size)
            interpolated_frame = np.interp(
                np.linspace(0, original_width - 1, target_frame_size),
                np.arange(original_width),
                current_image_row
            )
            output_wavetable_data[i, :] = interpolated_frame

        # Flatten the 2D array into a 1D audio array
        audio_data = output_wavetable_data.flatten()

        # Apply DSP functions
        audio_data = maximize(audio_data, target_frame_size) # Maximize volume
        processing_log.append("Maximized overall volume.")

        # Default fade application (as per previous logic in audio script)
        # Note: The 'fade' DSP function typically expects flattened audio_data
        # and reshapes it internally.
        audio_data = fade(audio_data, target_frame_size, 5) # Fixed 5 samples fade
        processing_log.append("Applied 5-sample fade to each frame.")


        if apply_shuffle_choice:
            audio_data = shuffle(audio_data, target_frame_size, shuffle_groups, shuffle_seed)
            processing_log.append(f"Shuffled frames (groups: {shuffle_groups}, seed: {shuffle_seed}).")

        # Save as WAV
        write_wav(
            filename_out=output_wav_path, # Zmieniono na przekazywanie obiektu Path
            audio_data=audio_data,
            frame_size=target_frame_size,
            num_frames=target_num_frames,
            samplerate=TARGET_SAMPLERATE,
            add_uhwt_chunk=True, # Add UHWT chunk for compatibility
            add_srge_chunk=True, # Add SRGE chunk for compatibility
            comment=f"Generated from {image_path.name}"
        )
        processing_log.append(f"Saved processed wavetable: '{output_wav_path.name}'.")
        return True

    except Exception as e:
        print(f"Error processing '{image_path.name}': {e}")
        processing_log.append(f"ERROR: Failed to process '{image_path.name}': {e}")
        return False


def main():
    """Main function to iterate through input directory and process image files."""
    global APPLY_SHUFFLE_INTERACTIVE, SHUFFLE_GROUPS_INTERACTIVE, SHUFFLE_SEED_INTERACTIVE

    input_dir = Path(input_directory_name)
    output_dir = Path(output_directory_name)

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Looking for images in: {input_dir.resolve()}")

    image_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".gif"]
    found_images = []
    for ext in image_extensions:
        for image_file in input_dir.glob(f"*{ext}"):
            found_images.append(image_file)

    if not found_images:
        print(f"No images ({','.join(image_extensions)}) found in folder '{input_dir.name}'.")
        print("Put images in that folder and try again.")
        sys.exit(1)

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
        SHUFFLE_GROUPS_INTERACTIVE = 0
        SHUFFLE_SEED_INTERACTIVE = None

    print(f"Found {len(found_images)} images to process.")

    for i, image_file_path in enumerate(found_images):
        output_wav_filename = f"{i + 1:03d}_{image_file_path.stem}.wav"
        output_wav_path = output_dir / output_wav_filename

        print(f"\n--- Processing [{i+1}/{len(found_images)}]: '{image_file_path.name}' -> '{output_wav_path.name}' ---")

        success = process_image(
            image_path=image_file_path,
            output_wav_path=output_wav_path,
            target_frame_size=DIGITAKT_IMAGE_TARGET_FRAME_SIZE,
            target_num_frames=TARGET_NUM_FRAMES,
            apply_shuffle_choice=APPLY_SHUFFLE_INTERACTIVE,
            shuffle_groups=SHUFFLE_GROUPS_INTERACTIVE,
            shuffle_seed=SHUFFLE_SEED_INTERACTIVE
        )

        if success:
            print(f"Successfully processed '{image_file_path.name}'.")
        else:
            print(f"Failed to process '{image_file_path.name}'. See logs for details.")

    if processing_log:
        print("\n--- DSP Processing Log ---")
        for entry in processing_log:
            print(f"- {entry}")

if __name__ == "__main__":
    main()