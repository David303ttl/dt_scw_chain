import json
import shutil
import struct
import sys
from collections import namedtuple
from functools import cached_property
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from dt_scw_chain.cli import get_cli # Changed from .cli
from dt_scw_chain.constants import Constant # Changed from .constants
from dt_scw_chain.utils import ( # Changed from .utils
    b64_to_array,
    bytes_to_array,
    ffmpeg_read,
    ffprobe_samplerate,
    get_md5,
    to_float32,
    to_mono,
)

# --- CUSTOM MODIFICATIONS FOR DIGITAKT TARGET PARAMETERS (for image conversion) ---
DIGITAKT_IMAGE_TARGET_NUM_FRAMES = 64
DIGITAKT_IMAGE_TARGET_FRAME_SIZE = 367

class InputFile:
    def __init__(self, infile):
        self.infile = Path(infile).resolve()

    @property
    def parent(self) -> Path:
        return self.infile.parent

    @property
    def name(self) -> str:
        return self.infile.name

    @property
    def extension(self) -> str:
        return self.infile.suffix.lower()

    @property
    def size_in_bytes(self) -> int:
        return self.infile.stat().st_size

    @cached_property
    def cache(self) -> bytes | dict:
        if self.extension in Picture.extensions:
            # Image files are not cached as bytes, but processed directly in app.py
            return None # Or raise an error if direct image parsing is not expected here
        with open(self.infile, "rb") as f:
            return f.read()

    def recognize_type(self):
        if self.extension == ".json":
            return WaveEdit(self.infile)
        if self.extension in Picture.extensions:
            return Picture(self.infile) # Return Picture object for image handling
        return RawAudio(self.infile)


class RawAudio:
    # Supported raw audio extensions
    extensions = (".wav", ".aif", ".aiff", ".mp3", ".flac", ".ogg")

    def __init__(self, infile):
        self.infile = infile

    def parse(self) -> namedtuple:
        # Use ffprobe to get the sample rate first
        samplerate = ffprobe_samplerate(self.infile, Constant.DEFAULT_SAMPLERATE, shutil.which("ffprobe")) # Ensure ffprobe_path is determined
        audio_data = ffmpeg_read(self.infile, samplerate, shutil.which("ffmpeg")) # Ensure ffmpeg_path is determined
        audio_data = to_float32(audio_data)
        audio_data = to_mono(audio_data)

        frame_size = Constant.DEFAULT_FRAME_SIZE # Use default or infer if possible
        num_frames = len(audio_data) // frame_size
        md5 = get_md5(audio_data)

        info = namedtuple(
            "info",
            ["frame_size", "num_frames", "samplerate", "audio_data", "md5"],
        )
        return info(frame_size, num_frames, samplerate, audio_data, md5)


class Picture:
    extensions = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff")

    def __init__(self, infile):
        self.infile = infile

    def parse(self) -> namedtuple:
        try:
            with Image.open(self.infile) as img:
                img.verify() # Verify that it is an image
            
            # Placeholder for image-derived audio properties.
            # Actual audio data will be generated in app.py
            frame_size = DIGITAKT_IMAGE_TARGET_FRAME_SIZE
            num_frames = DIGITAKT_IMAGE_TARGET_NUM_FRAMES
            samplerate = Constant.DEFAULT_SAMPLERATE # Use a default samplerate for generated audio
            
            # Return an empty array for audio_data here, as it's generated later
            audio_data = np.array([], dtype=np.float32) 
            md5 = get_md5(audio_data) # MD5 of empty array for now

            info = namedtuple(
                "info",
                ["frame_size", "num_frames", "samplerate", "audio_data", "md5"],
            )
            return info(frame_size, num_frames, samplerate, audio_data, md5)

        except Exception as e:
            raise ValueError(f"Not a valid image file: {self.infile} - {e}")


class WaveEdit:
    extensions = (".json",)

    def __init__(self, infile):
        self.infile = infile
        with open(self.infile, "r", encoding="utf-8") as f:
            self.cache = json.load(f)

    def parse_interleaved_frames(self) -> tuple:
        audio_data = b64_to_array(
            self.cache["groups"][0]["components"][0]["wave_data"]
        )
        audio_data = to_float32(audio_data)
        samplerate = int(
            self.cache["groups"][0]["components"][0]["audio_sample_rate"]
        )
        frame_size = int(
            self.cache["groups"][0]["components"][0]["keyframes"][0][
                "window_size"
            ]
        )
        num_frames = len(audio_data) / frame_size
        md5 = get_md5(audio_data)
        return frame_size, num_frames, samplerate, audio_data, md5

    def parse_separate_frames(self) -> tuple:
        keys = self.cache["groups"][0]["components"][0]["keyframes"]
        frames = []
        for key in keys:
            as_b64 = b64_to_array(key["wave_data"])
            frames.append(as_b64)
        num_frames = len(keys)
        frame_size = 2048 
        samplerate = 48000 
        audio_data = np.concatenate(frames)
        audio_data = to_float32(audio_data)
        md5 = get_md5(audio_data)
        return frame_size, num_frames, samplerate, audio_data, md5

    def parse(self):
        info = namedtuple(
            "info",
            ["frame_size", "num_frames", "samplerate", "audio_data", "md5"],
        )
        # Try parsing as interleaved first, then as separate frames
        try:
            return info(*self.parse_interleaved_frames())
        except (KeyError, TypeError, IndexError):
            # Fallback to parsing separate frames if interleaved fails
            return info(*self.parse_separate_frames())