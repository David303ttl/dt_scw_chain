import math
import sys
from pathlib import Path
from pprint import pprint
import numpy as np
from PIL import Image
from dt_scw_chain.cli import get_cli 
from dt_scw_chain.constants import Constant 
from dt_scw_chain.dsp import ( 
    fade,
    flip,
    interpolate,
    invert_phase,
    maximize,
    normalize,
    processing_log,
    resize,
    reverse,
    shuffle,
    sort,
    trim,
)
from dt_scw_chain.formats import InputFile, Picture, DIGITAKT_IMAGE_TARGET_FRAME_SIZE 
from dt_scw_chain.utils import ( 
    get_frame_size_from_hint,
    pad_audio_data,
    write_wav,
    write_wt,
)


def main() -> None:
    cli = get_cli()

    sys.tracebacklimit = 0 if not cli.debug else 1

    infile = cli.infile

    if not Path(infile).exists():
        raise FileNotFoundError(infile)

    infile_obj = InputFile(infile)
    content = infile_obj.recognize_type().parse()

    print("Input file details:")
    pprint(dict(content._asdict()), sort_dicts=False)

    frame_size_hint: int = get_frame_size_from_hint(infile_obj.name)

    if infile_obj.extension in Picture.extensions:
        # If input is an image, process it
        image = Image.open(infile)

        # Convert image to numpy array, handle transparency and normalize to -1 to 1
        img_array = np.array(image.convert("L"), dtype=np.float32) / 127.5 - 1.0

        # Calculate target frame size based on DIGITAKT_IMAGE_TARGET_FRAME_SIZE
        target_frame_size = DIGITAKT_IMAGE_TARGET_FRAME_SIZE

        # Resize image to fit target frame size and number of frames (64)
        # Assuming height is num_frames, width is frame_size
        img_array_resized = np.array(Image.fromarray((img_array + 1.0) * 127.5).resize(
            (target_frame_size, Constant.DIGITAKT_WAVETABLE_NUM_FRAMES), Image.Resampling.LANCZOS
        ), dtype=np.float32) / 127.5 - 1.0


        # Flatten the image data to represent a continuous audio waveform
        audio_data = img_array_resized.flatten()
        samplerate = Constant.DEFAULT_SAMPLERATE # Default samplerate for image-generated wavetables
        frame_size = target_frame_size # Frame size is now based on image processing
        out_num_frames = Constant.DIGITAKT_WAVETABLE_NUM_FRAMES # Fixed 64 frames for Digitakt
        processing_log.append(f"Generated audio data from image '{infile}'. Samplerate: {samplerate}, Frame Size: {frame_size}, Number of Frames: {out_num_frames}.")

    else:
        # Original logic for audio files
        audio_data = content.audio_data
        samplerate = content.samplerate
        frame_size = content.frame_size
        out_num_frames = content.num_frames

    if cli.trim_start or cli.trim_end:
        audio_data = trim(audio_data, cli.trim_start, cli.trim_end)

    if cli.resize:
        audio_data = resize(audio_data, frame_size, cli.resize)
        frame_size = cli.resize # Update frame_size after resize
        out_num_frames = len(audio_data) // frame_size # Update num_frames after resize

    if cli.normalize:
        audio_data = normalize(audio_data)

    if cli.maximize:
        audio_data = maximize(audio_data, frame_size)

    if cli.fade:
        audio_data = fade(audio_data, frame_size, cli.fade)

    if cli.sort:
        audio_data = sort(audio_data, frame_size)

    if cli.shuffle:
        # Use cli.shuffle_groups and cli.shuffle_seed
        shuffle_seed_value = None
        if cli.shuffle_seed.lower() == "random":
            # Generate a truly random seed
            shuffle_seed_value = random.randint(0, 2**32 - 1)
            print(f"Using generated random seed for shuffling: {shuffle_seed_value}")
        else:
            try:
                shuffle_seed_value = int(cli.shuffle_seed)
                print(f"Using provided seed for shuffling: {shuffle_seed_value}")
            except ValueError:
                print("Warning: Invalid value for --shuffle-seed. Must be an integer or 'random'. Using random seed.")
                shuffle_seed_value = random.randint(0, 2**32 - 1)
                print(f"Using generated random seed: {shuffle_seed_value}")

        audio_data = shuffle(audio_data, frame_size, cli.shuffle_groups, shuffle_seed_value)

    if cli.reverse:
        audio_data = reverse(audio_data, frame_size)

    if cli.flip:
        audio_data = flip(audio_data, frame_size)

    if cli.invert_phase:
        audio_data = invert_phase(audio_data, frame_size)

    if cli.interpolate:
        audio_data = interpolate(audio_data, frame_size, cli.interpolate)

    # Pad frames to reach DIGITAKT_WAVETABLE_NUM_FRAMES (64) if needed
    if out_num_frames < Constant.DIGITAKT_WAVETABLE_NUM_FRAMES:
        audio_data, out_num_frames = pad_audio_data(
            audio_data,
            frame_size,
            out_num_frames,
            Constant.DIGITAKT_WAVETABLE_NUM_FRAMES
        )
        processing_log.append(f"Padded to {out_num_frames} frames.")


    # Handle output files
    if cli.outfile or cli.split:
        if cli.split:
            frames = audio_data.reshape(out_num_frames, frame_size)
            for i, frame in enumerate(frames):
                filename_out = Path(infile).stem + f"_{i:03d}.wav"
                filename_out_path = Path(cli.split) / filename_out
                Path(cli.split).mkdir(parents=True, exist_ok=True) # Ensure output directory exists
                write_wav(
                    filename_out=filename_out_path,
                    audio_data=frame,
                    frame_size=frame_size, # For single frame output, frame_size is just len(frame)
                    num_frames=1, # Always 1 for split output
                    samplerate=samplerate,
                    add_uhwt_chunk=False, # Don't add chunks to split files
                    add_srge_chunk=False,
                    comment="",
                )
        else: # This 'else' corresponds to the 'if cli.outfile'
            # Ensure correct indentation for this line 108, which previously caused an issue
            outfile_extension = Path(cli.outfile).suffix
            if outfile_extension == ".wav":
                write_wav(
                    filename_out=cli.outfile,
                    audio_data=audio_data,
                    frame_size=frame_size,
                    num_frames=out_num_frames,
                    samplerate=samplerate,
                    add_uhwt_chunk=cli.add_uhwt,
                    add_srge_chunk=cli.add_srge,
                    comment=cli.comment,
                )
            elif outfile_extension == ".wt":
                write_wt(
                    filename_out=cli.outfile,
                    audio_data=audio_data,
                    frame_size=frame_size,
                    num_frames=out_num_frames,
                    flags=0,
                )
            else:
                raise NotImplementedError

            print("\nOutput file details:")
            outfile = InputFile(cli.outfile).recognize_type().parse()
            pprint(dict(outfile._asdict()), sort_dicts=False)

    else: # This 'else' block is for the case where neither --outfile nor --split is provided
        print("\nNo output file specified (--outfile or --split). No file saved.")

    if processing_log:
        print("\n--- DSP Processing Log ---")
        for entry in processing_log:
            print(f"- {entry}")