import base64
import re
import shlex
import struct
import subprocess
from hashlib import md5
from pathlib import Path

import numpy as np
import soundfile as sf


def bytes_to_array(audio_bytes: bytes, fmt_chunk):
    """Convert bytes to array."""
    bytes_per_sample = fmt_chunk.block_align // fmt_chunk.num_channels

    # PCM
    if fmt_chunk.codec_id == 1:
        if fmt_chunk.bitdepth in {16, 32}:
            dtype = f"<i{bytes_per_sample}"
        # 24-bit data needs reformatting
        elif fmt_chunk.bitdepth == 24:
            dtype = "V1"
        else:
            raise ValueError("Unsupported bit depth:", fmt_chunk.bitdepth)
    # IEEE float
    elif fmt_chunk.codec_id == 3:
        if fmt_chunk.bitdepth in {32, 64}:
            dtype = f"<f{bytes_per_sample}"
        else:
            raise ValueError("Unsupported bit depth:", fmt_chunk.bitdepth)
    # Extensible
    elif fmt_chunk.codec_id == 65534:
        if fmt_chunk.codec_id_hint == 1:
            dtype = f"<i{bytes_per_sample}"
        elif fmt_chunk.codec_id_hint == 3:
            dtype = f"<f{bytes_per_sample}"
        else:
            raise ValueError("Unsupported codec_id_hint:", fmt_chunk.codec_id_hint)
    else:
        raise ValueError("Unsupported codec ID:", fmt_chunk.codec_id)

    # Convert byte data to numpy array
    audio_data = np.frombuffer(audio_bytes, dtype=dtype)

    # Handle 24-bit PCM: unpack to 32-bit integer
    if fmt_chunk.codec_id == 1 and fmt_chunk.bitdepth == 24:
        audio_data = audio_data.view("<i4") >> 8

    return audio_data


def b64_to_array(audio_b64: str):
    """Convert base64 string to audio array (WaveEdit format specific)."""
    return np.frombuffer(base64.b64decode(audio_b64), dtype="<f4")


def to_float32(audio_data: np.ndarray) -> np.ndarray:
    """Convert audio data to float32, scaling if necessary."""
    if audio_data.dtype != np.float32:
        if audio_data.dtype.kind == 'i': # Integer types
            info = np.iinfo(audio_data.dtype)
            audio_data = audio_data.astype(np.float32) / max(abs(info.min), abs(info.max))
        elif audio_data.dtype.kind == 'f': # Float types (e.g., float64)
            audio_data = audio_data.astype(np.float32)
        else:
            raise TypeError(f"Unsupported audio data type for conversion to float32: {audio_data.dtype}")
    return audio_data


def to_mono(audio_data: np.ndarray) -> np.ndarray:
    """Convert stereo or multi-channel audio to mono."""
    if audio_data.ndim > 1:
        return audio_data.mean(axis=1)
    return audio_data


def get_md5(audio_data: np.ndarray) -> str:
    """Calculate MD5 hash of audio data."""
    return md5(audio_data.tobytes()).hexdigest()


def pad_audio_data(
    audio_data: np.ndarray, frame_size: int, current_num_frames: int, target_num_frames: int
) -> tuple[np.ndarray, int]:
    """
    Pad audio data with silence to reach target_num_frames.
    Args:
        audio_data: The current audio data.
        frame_size: Size of a single frame.
        current_num_frames: Current number of frames in audio_data.
        target_num_frames: Desired total number of frames.
    Returns:
        A tuple: (padded_audio_data, new_num_frames)
    """
    if current_num_frames < target_num_frames:
        silence_samples_needed = (target_num_frames - current_num_frames) * frame_size
        silence_padding = np.zeros(silence_samples_needed, dtype=np.float32)
        audio_data = np.concatenate((audio_data, silence_padding))
        current_num_frames = target_num_frames
    return audio_data, current_num_frames


def get_frame_size_from_hint(filename: str) -> int | None:
    """
    Attempt to extract frame size hint from filename (e.g., 'sample_2048.wav' -> 2048).
    Returns None if no hint is found or if it's invalid.
    """
    match = re.search(r'_(\d+)\.(wav|aif|aiff|mp3|flac|ogg|png|jpg|jpeg|webp|bmp|tiff)$', filename, re.IGNORECASE)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            pass # Not a valid integer frame size
    return None


def write_wav(
    filename_out: Path,
    audio_data: np.ndarray,
    frame_size: int,
    num_frames: int,
    samplerate: int,
    add_uhwt_chunk: bool = False,
    add_srge_chunk: bool = False,
    comment: str = "",
) -> None:
    """Write data array as float32 .wav file with optional custom chunks."""
    # Ensure audio_data is float32 and within -1.0 to 1.0 range
    audio_data = to_float32(audio_data)

    # Use soundfile to write the basic WAV
    sf.write(filename_out, audio_data, samplerate, subtype='PCM_16') # Using PCM_16 for Digitakt compatibility

    # Add custom chunks if requested
    if add_uhwt_chunk or add_srge_chunk or comment:
        # Re-open in binary append mode for custom chunk manipulation
        with open(filename_out, 'r+b') as f:
            f.seek(0, 2) # Go to end of file

            # UHWT Chunk (MicroFreak/MicroFreak Vocoder specific)
            if add_uhwt_chunk:
                uhwt_chunk_id = b'uhwt'
                # For Digitakt compatibility, num_frames=64, frame_size=367 is common
                # These might need to be hardcoded or passed based on Digitakt's actual requirements
                # Here using values derived from input.
                uhwt_data = struct.pack("<II", num_frames, frame_size)
                uhwt_chunk_size = len(uhwt_data)
                f.write(uhwt_chunk_id)
                f.write(struct.pack("<I", uhwt_chunk_size))
                f.write(uhwt_data)

            # SRGE Chunk (Sample Robot specific)
            if add_srge_chunk:
                srge_chunk_id = b'SRGE'
                # Placeholder for SRGE data structure.
                # Actual SRGE format is complex. This is a minimal placeholder.
                # For Digitakt, this might not be strictly necessary, but included for completeness.
                srge_data = b'\x00' * 8 # Example placeholder, actual data would be detailed.
                srge_chunk_size = len(srge_data)
                f.write(srge_chunk_id)
                f.write(struct.pack("<I", srge_chunk_size))
                f.write(srge_data)
            
            # Write comment to a 'LIST' chunk of type 'INFO' (Standard WAV metadata)
            if comment:
                list_chunk_id = b'LIST'
                info_id = b'INFO'
                
                # Prepare IART (Artist) and ICMT (Comment) sub-chunks if needed,
                # or a simple ICMT as requested.
                icmt_id = b'ICMT'
                # Ensure comment is null-terminated and even length for chunk padding
                encoded_comment = comment.encode('utf-8') + b'\x00'
                if len(encoded_comment) % 2 != 0:
                    encoded_comment += b'\x00'

                icmt_size = len(encoded_comment)
                
                info_data = (
                    icmt_id + struct.pack("<I", icmt_size) + encoded_comment
                )
                
                list_chunk_size = len(info_id) + len(info_data) + 4 # 4 bytes for INFO ID
                
                f.write(list_chunk_id)
                f.write(struct.pack("<I", list_chunk_size))
                f.write(info_id)
                f.write(info_data)
    
    # After writing chunks, update the overall RIFF chunk size
    # This involves rewriting the size at the very beginning of the WAV file
    file_size = filename_out.stat().st_size
    with open(filename_out, 'r+b') as f:
        f.seek(4) # RIFF chunk size is at byte offset 4
        f.write(struct.pack("<I", file_size - 8)) # Total file size - "RIFF" (4 bytes) - size (4 bytes)


def write_wt(
    filename_out: Path, audio_data: np.ndarray, frame_size: int, num_frames: int, flags: int = 0
) -> None:
    """Write data array as float32 .wt file (Serum Wavetable format)."""
    header = (b"vawt", frame_size, num_frames, flags)
    header_packed = struct.pack("<4s i H H", *header)

    with open(filename_out, "wb") as outfile:
        outfile.write(header_packed)
        audio_data.tofile(outfile)


def ffprobe_samplerate(
    filename_in: str, fallback_samplerate: int, ffprobe_path: str
) -> int:
    """Probe media file (using 'ffprobe') and find its samplerate."""
    # Ensure ffprobe_path is correctly quoted if it contains spaces
    ffprobe_command = f'"{ffprobe_path}" -v error -select_streams a:0 -show_entries stream=sample_rate -of default=noprint_wrappers=1:nokey=1 "{filename_in}"'
    
    try:
        # Use shell=True for simpler execution of quoted path
        ffprobe_out = subprocess.check_output(
            ffprobe_command, shell=True, stderr=subprocess.PIPE, text=True
        ).strip()
        
        if ffprobe_out.isdigit():
            return int(ffprobe_out)
        else:
            return fallback_samplerate
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Warning: Could not get samplerate using ffprobe for '{filename_in}'. Error: {e}. Using fallback samplerate {fallback_samplerate} Hz.")
        return fallback_samplerate


def ffmpeg_read(
    filename_in: str, samplerate_in: int, ffmpeg_path: str
) -> np.ndarray:
    """Decode compressed media file into float32 mono PCM data using ffmpeg."""
    # Ensure ffmpeg_path is correctly quoted if it contains spaces
    read_command = (
        f'"{ffmpeg_path}" -i "{filename_in}" -f f32le -ac 1 -ar {samplerate_in} -'
    )
    
    try:
        # Use shell=True for simpler execution of quoted path
        proc = subprocess.run(
            read_command,
            shell=True,
            capture_output=True,
            check=True,
        )
        audio_data = np.frombuffer(proc.stdout, dtype=np.float32)
        return audio_data
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise RuntimeError(f"Failed to decode audio file '{filename_in}' with ffmpeg: {e}. "
                           "Ensure ffmpeg is installed and accessible in your PATH.")