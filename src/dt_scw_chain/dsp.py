import numpy as np
import random # Added for shuffle seed generation
from scipy.signal.windows import hann
import math

processing_log = [] # Moved to global scope or a dedicated logging module if more complex


def reverse(audio_data: np.ndarray, frame_size: int) -> np.ndarray:
    """Reverse order of frames."""
    processing_log.append("Reverse order of frames.")
    return np.flip(audio_data.reshape(-1, frame_size), axis=0)


def flip(audio_data: np.ndarray, frame_size: int) -> np.ndarray:
    """Reverse data within each frame."""
    audio_data = audio_data.reshape(-1, frame_size)
    processing_log.append("Reverse data within each frame.")
    return np.flip(audio_data, axis=1)


def invert_phase(
    audio_data: np.ndarray,
    frame_size: int,
) -> np.ndarray:
    """Invert phase of the waveform."""
    original_shape = audio_data.shape  # Save original shape to handle 1D arrays correctly
    audio_data = audio_data.reshape(-1, frame_size)
    processing_log.append("Invert phase.")
    inverted_data = audio_data * -1

    if len(original_shape) == 1:
        return inverted_data.flatten().astype(np.float32)
    return inverted_data.astype(np.float32)


def shuffle(
    audio_data: np.ndarray,
    frame_size: int,
    groups: int = 1,
    seed: int | None = None,
) -> np.ndarray:
    """
    Shuffle frames in the audio data.
    If groups is 0, shuffles individual frames.
    If groups > 0, shuffles frames within X equal blocks.
    Args:
        audio_data: 1D numpy array of audio data.
        frame_size: Number of samples per frame.
        groups: Number of groups to shuffle within. 0 for no grouping.
        seed: Seed for the random number generator. None for random seed.
    Returns:
        Shuffled audio data.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed) # Ensure python's random is also seeded if used elsewhere
    else:
        # If no seed explicitly provided, generate a truly random one
        random_seed_value = random.randint(0, 2**32 - 1)
        np.random.seed(random_seed_value)
        random.seed(random_seed_value)
        processing_log.append(f"Using generated random seed for shuffling: {random_seed_value}.")


    num_frames = len(audio_data) // frame_size
    frames = audio_data.reshape(num_frames, frame_size)

    if groups <= 0 or groups > num_frames:
        # Shuffle all frames if groups is 0 or invalid
        np.random.shuffle(frames)
        processing_log.append("Shuffled all frames.")
    else:
        # Shuffle within groups
        frames_per_group = num_frames // groups
        remaining_frames = num_frames % groups

        shuffled_frames_list = []
        start_idx = 0
        for i in range(groups):
            end_idx = start_idx + frames_per_group
            if i < remaining_frames: # Distribute remaining frames to first few groups
                end_idx += 1
            
            group_frames = frames[start_idx:end_idx]
            np.random.shuffle(group_frames)
            shuffled_frames_list.append(group_frames)
            start_idx = end_idx
        
        frames = np.concatenate(shuffled_frames_list, axis=0)
        processing_log.append(f"Shuffled frames within {groups} groups.")

    return frames.flatten().astype(np.float32)


def trim(
    audio_data: np.ndarray,
    start_percent: float,
    end_percent: float,
) -> np.ndarray:
    """Trim audio by percentage from start and end."""
    if not (0 <= start_percent < 1 and 0 <= end_percent < 1):
        raise ValueError("Trim percentages must be between 0 and 1 (exclusive).")
    if start_percent + end_percent >= 1:
        raise ValueError("Sum of trim percentages cannot be 1 or greater.")

    total_samples = len(audio_data)
    start_sample = int(total_samples * start_percent)
    end_sample = total_samples - int(total_samples * end_percent)

    if start_sample >= end_sample:
        return np.array([], dtype=np.float32) # Return empty array if trimmed completely

    trimmed_data = audio_data[start_sample:end_sample]
    processing_log.append(
        f"Trimmed audio: {start_percent*100:.2f}% from start, {end_percent*100:.2f}% from end."
    )
    return trimmed_data.astype(np.float32)


def normalize(audio_data: np.ndarray) -> np.ndarray:
    """Normalize audio data to -1 to 1 peak."""
    peak_value = np.abs(audio_data).max()
    if peak_value == 0:
        processing_log.append("Normalization skipped: Audio data is silent.")
        return audio_data

    normalized_data = audio_data / peak_value
    processing_log.append("Normalized audio to 0 dBFS peak.")
    return normalized_data.astype(np.float32)


def maximize(audio_data: np.ndarray, frame_size: int) -> np.ndarray:
    """Maximize each frame's volume independently to -1 to 1 peak."""
    frames = audio_data.reshape(-1, frame_size)
    maximized_frames = []
    for frame in frames:
        peak_value = np.abs(frame).max()
        if peak_value > 0:
            maximized_frames.append(frame / peak_value)
        else:
            maximized_frames.append(frame) # Keep silent frames silent
    processing_log.append("Maximized each frame's volume.")
    return np.concatenate(maximized_frames).astype(np.float32)


def fade(audio_data: np.ndarray, frame_size: int, fade_samples: int) -> np.ndarray:
    """Apply linear fade in/out to each frame."""
    if fade_samples <= 0:
        processing_log.append("Fade skipped: fade_samples is zero or negative.")
        return audio_data

    frames = audio_data.reshape(-1, frame_size)
    faded_frames = []
    fade_in_ramp = np.linspace(0, 1, fade_samples)
    fade_out_ramp = np.linspace(1, 0, fade_samples)

    for frame in frames:
        if len(frame) < fade_samples * 2:
            # Handle cases where frame is too short for full fade
            frame[:] *= np.linspace(0, 1, len(frame))
            processing_log.append(f"Warning: Frame too short for full {fade_samples}-sample fade. Applied full-frame fade.")
        else:
            frame[:fade_samples] *= fade_in_ramp
            frame[len(frame) - fade_samples :] *= fade_out_ramp
        faded_frames.append(frame)

    processing_log.append(f"Applied {fade_samples}-sample linear fade to each frame.")
    return np.concatenate(faded_frames).astype(np.float32)


def sort(audio_data: np.ndarray, frame_size: int) -> np.ndarray:
    """Sort frames by average absolute amplitude."""
    frames = audio_data.reshape(-1, frame_size)
    average_amplitudes = np.mean(np.abs(frames), axis=1)
    sorted_indices = np.argsort(average_amplitudes)
    sorted_frames = frames[sorted_indices]
    processing_log.append("Sorted frames by average absolute amplitude.")
    return sorted_frames.flatten().astype(np.float32)


def resize(
    audio_data: np.ndarray, in_frame_size: int, out_frame_size: int
) -> np.ndarray:
    """
    Resize each frame to a new sample size using linear interpolation.
    Normalizes the output to -1 to 1.
    """
    num_frames = len(audio_data) // in_frame_size
    frames = audio_data.reshape(num_frames, in_frame_size)

    resized_frames = np.zeros((num_frames, out_frame_size), dtype=np.float32)

    for i in range(num_frames):
        resized_frames[i, :] = np.interp(
            np.linspace(0, in_frame_size - 1, out_frame_size),
            np.arange(in_frame_size),
            frames[i, :],
        )

    # Normalize after resizing
    peak_value = np.abs(resized_frames).max()
    if peak_value > 0:
        resized_frames = resized_frames / peak_value
    
    processing_log.append(f"Resized frames: {in_frame_size} samples -> {out_frame_size} samples.")
    return resized_frames.flatten().astype(np.float32)


def interpolate(
    audio_data: np.ndarray, frame_size: int, out_num_frames: int
) -> np.ndarray:
    """
    Interpolate (or decimate) frames to a new total frame count.
    Each frame is individually interpolated from original frame to the new number of frames.
    """
    num_frames = len(audio_data) // frame_size
    frames = audio_data.reshape(num_frames, frame_size)

    # Create a new array for interpolated frames
    interpolated_frames_array = np.zeros(
        (out_num_frames, frame_size), dtype=np.float32
    )

    for i in range(frame_size):
        # Extract the i-th sample from all frames (creating a "vertical" waveform)
        vertical_waveform = frames[:, i]

        # Interpolate this vertical waveform to the new number of frames
        interpolated_vertical_waveform = np.interp(
            np.linspace(0, num_frames - 1, out_num_frames),
            np.arange(num_frames),
            vertical_waveform,
        )
        # Place it back into the new frame array
        interpolated_frames_array[:, i] = interpolated_vertical_waveform

    # Normalize the entire result
    peak_value = np.abs(interpolated_frames_array).max()
    if peak_value > 0:
        interpolated_frames_array /= peak_value

    processing_log.append(
        f"Interpolated frame count: {num_frames} -> {out_num_frames}."
    )
    return interpolated_frames_array.flatten().astype(np.float32)


def overlap(audio_data: np.ndarray, frame_size: int, overlap_size: float):
    """
    Resize by overlapping frames. (Note: This function appears unused in current scripts)
    """
    audio_data = audio_data.reshape(-1, frame_size)

    num_overlap_samples = int(overlap_size * audio_data.shape[1])

    fade_length = int(num_overlap_samples / 4)
    fade_in = hann(fade_length * 2)[:fade_length]
    fade_out = hann(fade_length * 2)[fade_length:]

    # The rest of this function's logic would need to be re-evaluated for current use.
    # It seems to be an incomplete/unused feature.
    processing_log.append("Overlap function called (functionality may be incomplete).")
    return audio_data.flatten().astype(np.float32) # Returning original for now if not fully implemented

# --- NOWA FUNKCJA DO PRZETWARZANIA DANYCH OBRAZU NA WAVETABLE ---
def resize_image_data_for_wavetable(
    image_data_flat: np.ndarray,
    original_width: int,
    original_height: int,
    target_frame_size: int,
    target_num_frames: int,
) -> np.ndarray:
    """
    Przetwarza jednowymiarowe dane obrazu (flattened) na dane wavetable.
    Obsługuje zmianę rozmiaru i interpolację, aby dopasować do docelowych wymiarów.
    """
    processing_log.append(
        f"Przetwarzanie danych obrazu o wymiarach ({original_width}x{original_height}) "
        f"na wavetable: {target_num_frames} ramek po {target_frame_size} próbek."
    )

    # Krok 1: Przekształć jednowymiarowe dane obrazu na dwuwymiarowe (ramki x próbki)
    # Oryginalny obraz jest (height, width).
    # Po flatten() jest 1D. Musimy go z powrotem podzielić na "linie" (ramki).
    if image_data_flat.size != original_width * original_height:
        raise ValueError(f"Rozmiar spłaszczonych danych ({image_data_flat.size}) nie zgadza się z wymiarami obrazu ({original_width}x{original_height}).")

    # Zakładamy, że każda "linia" obrazu to "ramka"
    image_frames = image_data_flat.reshape(original_height, original_width)

    # Krok 2: Zmiana rozmiaru każdej ramki (szerokości) do target_frame_size
    # oraz interpolacja liczby ramek (wysokości) do target_num_frames
    
    # Utwórz nową tablicę o docelowych wymiarach
    output_wavetable_data = np.zeros(
        (target_num_frames, target_frame_size), dtype=np.float32
    )

    for i in range(target_num_frames):
        # Oblicz, która oryginalna ramka (linia obrazu) odpowiada tej docelowej ramce
        # Używamy interpolacji liniowej dla indeksów ramek
        src_row_idx_float = (i / (target_num_frames - 1)) * (original_height - 1)
        
        # Pobierz odpowiednie oryginalne dane z obrazu
        # Interpolacja między sąsiednimi rzędami, jeśli src_row_idx_float jest float
        if original_height > 1: # Avoid division by zero if image has only one row
            row1_idx = math.floor(src_row_idx_float)
            row2_idx = min(math.ceil(src_row_idx_float), original_height - 1) # Ensure index stays within bounds
            
            if row1_idx == row2_idx: # No interpolation needed, it's an exact row
                current_image_row = image_frames[row1_idx, :]
            else:
                alpha = src_row_idx_float - row1_idx
                current_image_row = (1 - alpha) * image_frames[row1_idx, :] + \
                                    alpha * image_frames[row2_idx, :]
        else: # If original_height is 1, just use the first (and only) row
            current_image_row = image_frames[0, :]

        # Teraz interpoluj próbki w ramce (z original_width na target_frame_size)
        interpolated_frame = np.interp(
            np.linspace(0, original_width - 1, target_frame_size), # Docelowe punkty
            np.arange(original_width), # Oryginalne punkty
            current_image_row # Wartości do interpolacji
        )
        output_wavetable_data[i, :] = interpolated_frame

    # Normalizuj całe dane, aby zapobiec clippingowi
    peak_value = np.abs(output_wavetable_data).max()
    if peak_value > 0:
        output_wavetable_data /= peak_value

    processing_log.append(
        f"Obraz przekształcony do wavetable o kształcie ({target_num_frames}, {target_frame_size})."
    )
    return output_wavetable_data.flatten().astype(np.float32)