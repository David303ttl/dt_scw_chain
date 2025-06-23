import argparse
from dt_scw_chain.constants import Constant

def get_cli():
    parser = argparse.ArgumentParser(
        description="dt_scw_chain: Digitakt Single Cycle Waveform and Wavetable Chain Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("infile", help="Input file (audio or image)")
    parser.add_argument(
        "--outfile", "-o", help="Output .wav or .wt file (e.g., output.wav or output.wt)"
    )
    parser.add_argument(
        "--add-uhwt",
        action="store_true",
        help="Add MicroFreak/MicroFreak Vocoder specific (UHWT) chunk to WAV output",
    )
    parser.add_argument(
        "--add-srge",
        action="store_true",
        help="Add Sample Robot (SRGE) chunk to WAV output",
    )
    parser.add_argument(
        "--comment",
        "-c",
        default="",
        help="Add a comment to the output WAV file metadata",
    )
    parser.add_argument(
        "--split",
        nargs="?",
        const=".",
        help="Split wavetable into individual WAV files (optionally specify directory, default: current directory)",
    )

    # DSP options
    dsp_group = parser.add_argument_group("DSP Options")
    dsp_group.add_argument(
        "--trim-start",
        type=float,
        default=0.0,
        help="Trim audio from the start (as a percentage, e.g., 0.1 for 10%%)",
    )
    dsp_group.add_argument(
        "--trim-end",
        type=float,
        default=0.0,
        help="Trim audio from the end (as a percentage, e.g., 0.1 for 10%%)",
    )
    dsp_group.add_argument(
        "--resize",
        type=int,
        help="Resize frames to a new sample size (e.g., 2048)",
    )
    dsp_group.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize audio to 0dBFS peak (adjusts overall volume)",
    )
    dsp_group.add_argument(
        "--maximize",
        action="store_true",
        help="Maximize each frame individually to 0dBFS peak",
    )
    dsp_group.add_argument(
        "--fade",
        type=int,
        default=0,
        help="Apply a linear fade in/out to each frame (number of samples)",
    )
    dsp_group.add_argument(
        "--sort", action="store_true", help="Sort frames by average amplitude"
    )
    dsp_group.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle frames. Use --shuffle-groups and --shuffle-seed for control.",
    )
    dsp_group.add_argument(
        "--shuffle-groups",
        type=int,
        default=0,
        help="Number of groups for shuffling (0 for individual frames). "
             "If X groups, frames are shuffled within X equal blocks.",
    )
    dsp_group.add_argument(
        "--shuffle-seed",
        type=str, # Changed to string to allow "random"
        default="random", # Changed default to "random"
        help="Seed for shuffling. 'random' for a new seed each run, or an integer for reproducible results.",
    )
    dsp_group.add_argument(
        "--reverse", action="store_true", help="Reverse order of frames"
    )
    dsp_group.add_argument(
        "--flip", action="store_true", help="Flip (reverse) data within each frame"
    )
    dsp_group.add_argument(
        "--invert-phase", action="store_true", help="Invert phase of the waveform"
    )
    dsp_group.add_argument(
        "--interpolate",
        type=int,
        help="Interpolate frames to a new frame count (e.g., 128)",
    )

    parser.add_argument(
        "--debug", action="store_true", help="Enable debug output and full tracebacks"
    )

    # Arguments specific to batch_convert_scw (for chaining SCWs)
    batch_scw_group = parser.add_argument_group("Single Cycle Waveform Chain Options")
    batch_scw_group.add_argument(
        "--batch-scw-input",
        type=str,
        help="Directory containing individual WAV files to be chained as SCWs.",
    )
    batch_scw_group.add_argument(
        "--batch-scw-output",
        type=str,
        help="Output directory for the generated SCW chain WAV file.",
    )
    batch_scw_group.add_argument(
        "--batch-scw-filename",
        type=str,
        default="digitakt_wavetable_chain.wav",
        help="Name of the output SCW chain WAV file.",
    )
    batch_scw_group.add_argument(
        "--scw-fade-samples",
        type=int,
        default=5,
        help="Number of samples for fade in/out on each SCW frame (when chaining).",
    )
    batch_scw_group.add_argument(
        "--scw-target-frames",
        type=int,
        default=Constant.DIGITAKT_WAVETABLE_NUM_FRAMES, # Use Constant for default
        help="Target number of frames for the generated SCW chain.",
    )
    batch_scw_group.add_argument(
        "--scw-target-samples",
        type=int,
        default=367,
        help="Target sample count for each SCW frame (when chaining).",
    )

    # Arguments specific to batch_waveedit_banks (for converting old WaveEdit banks)
    batch_we_group = parser.add_argument_group("WaveEdit Bank Conversion Options")
    batch_we_group.add_argument(
        "--batch-we-input",
        type=str,
        help="Directory containing original WaveEdit bank files (.wav, .mp3, etc.).",
    )
    batch_we_group.add_argument(
        "--batch-we-output",
        type=str,
        help="Output directory for the converted WaveEdit banks.",
    )
    batch_we_group.add_argument(
        "--we-apply-shuffle",
        action="store_true",
        help="Apply shuffling to frames when converting WaveEdit banks.",
    )
    batch_we_group.add_argument(
        "--we-shuffle-groups",
        type=int,
        default=0,
        help="Number of groups for shuffling WaveEdit bank frames.",
    )
    batch_we_group.add_argument(
        "--we-shuffle-seed",
        type=str,
        default="random",
        help="Seed for shuffling WaveEdit bank frames ('random' or integer).",
    )

    # Arguments specific to batch_image_wavetables
    batch_img_group = parser.add_argument_group("Image to Wavetable Options")
    batch_img_group.add_argument(
        "--batch-img-input",
        type=str,
        help="Directory containing image files (.png, .jpg, etc.) to convert to wavetables.",
    )
    batch_img_group.add_argument(
        "--batch-img-output",
        type=str,
        help="Output directory for the generated image wavetables.",
    )


    args = parser.parse_args()

    # Basic validation for mutually exclusive modes
    modes_active = sum([
        1 if args.infile else 0,
        1 if args.batch_scw_input else 0,
        1 if args.batch_we_input else 0,
        1 if args.batch_img_input else 0
    ])

    if modes_active > 1:
        parser.error("Only one mode can be active at a time: "
                     "single file processing (--infile), "
                     "batch SCW chain creation (--batch-scw-input), "
                     "batch WaveEdit bank conversion (--batch-we-input), "
                     "or batch image conversion (--batch-img-input).")
    elif modes_active == 0:
        parser.error("No input specified. Please provide --infile, --batch-scw-input, --batch-we-input, or --batch-img-input.")

    return args