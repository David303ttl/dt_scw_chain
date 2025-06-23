import soundfile as sf
import numpy as np
import os
from datetime import datetime

def analyze_all_wav_files_in_directory(directory_path, fixed_num_frames=64, output_file=None):
    """
    Scans the specified directory, analyzes all WAV files,
    and displays their wavetable properties, saving results to a file.

    Args:
        directory_path (str): Path to the directory containing WAV files.
        fixed_num_frames (int): Constant, expected number of frames in the wavetable.
                                This will be used to calculate the frame size.
        output_file (file object): File object to write the results.
                                   If None, results are printed to the console.
    """
    def write_output(text):
        if output_file:
            output_file.write(text + '\n')
        print(text)

    if not os.path.isdir(directory_path):
        write_output(f"Error: Directory '{directory_path}' does not exist or is not a directory.")
        return

    write_output(f"--- Starting WAV file analysis in directory: '{directory_path}' ---")

    all_files = os.listdir(directory_path)
    wav_files = [f for f in all_files if f.lower().endswith('.wav') and os.path.isfile(os.path.join(directory_path, f))]

    if not wav_files:
        write_output(f"No WAV files found in '{directory_path}'.")
        write_output("--- Analysis complete ---")
        return

    for i, filename in enumerate(wav_files):
        filepath = os.path.join(directory_path, filename)
        write_output(f"\n--- Analyzing file [{i+1}/{len(wav_files)}]: '{filename}' ---")
        
        try:
            info = sf.info(filepath)
            
            write_output(f"File Name: {filename}")
            write_output(f"Sample Rate: {info.samplerate} Hz")
            write_output(f"Channels: {info.channels}")
            write_output(f"Total Samples (all channels): {info.frames}")
            write_output(f"Duration: {info.duration:.2f} seconds")

            # Calculate actual frame size based on fixed_num_frames
            actual_frame_size = info.frames / fixed_num_frames
            write_output(f"Calculated frame size based on {fixed_num_frames} frames: {actual_frame_size:.2f} samples")

            if actual_frame_size.is_integer():
                write_output(f"\n[OK] File can be perfectly divided into {fixed_num_frames} frames of {int(actual_frame_size)} samples each.")
            else:
                write_output(f"\n[X] File with {info.frames} samples cannot be evenly divided into {fixed_num_frames} frames.")
                write_output(f"  Calculated frame size ({actual_frame_size:.2f} samples) is not an integer.")
                write_output(f"  This may indicate uneven frames or a different number of frames than {fixed_num_frames}.")

        except Exception as e:
            write_output(f"An error occurred while analyzing file '{filename}': {e}")
        
        write_output("-" * 50)

# --- Example script usage ---
if __name__ == "__main__":
    directory_to_scan = '.' # Directory to scan ('.': current directory)
    
    # Generate output filename with a timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_filename = f"wav_analysis_report_{timestamp}.txt"
    
    # Open file for writing
    with open(output_filename, 'w') as f:
        analyze_all_wav_files_in_directory(directory_to_scan, fixed_num_frames=64, output_file=f)
    
    print(f"\nAnalysis report saved to '{output_filename}'")
    print("--- All WAV file analysis complete ---")