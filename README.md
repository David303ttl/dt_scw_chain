# DT_SCW_CHAIN

## A Tool for Converting and Creating Wavetable Chains for Elektron Digitakt

DT_SCW_CHAIN is a Python-based tool designed for batch processing WaveEdit banks, Single Cycle Waveforms (SCW), and images, then combining them into wavetable chains optimized for the Elektron Digitakt sampler.

**Key Features and Adaptations for Elektron Digitakt:**

* **Digitakt Standards**: Elektron Digitakt requires individual Single Cycle Waveforms (SCW) to be 367 samples long at a 48 kHz sample rate, corresponding to the C3 note.
* **Versatile Input Conversion**:
    * **WaveEdit Banks**: The tool converts WaveEdit banks (typically 64 frames of 256 samples at 44.1 kHz) into the required 64 frames of 367 samples at 48 kHz format.
    * **SCW**: Single Cycle Waveforms (e.g., AKWF's 600-sample waveforms) are also converted to 367 samples.
    * **Images**: Capability to convert images directly into wavetable chains.
* **Process Automation**: Ensures efficient conversion of large archives (e.g., over 700 WaveEdit banks, over 4000 AKWF files) to Digitakt's specifications.
* **Audio Processing**:
    * Conversion to 48 kHz, 16-bit mono.
    * Automatic adjustment of sample length to 367, regardless of input file length.
    * Volume normalization.
    * Optional fade in/out to prevent clicks.
    * Ability to shuffle (randomize) frames within wavetable chains.

### Project Origin

The DT_SCW_CHAIN project is a fork of the [drzhnn/okwt](https://github.com/drzhnn/okwt) repository. It has been adapted and extended to meet the specific requirements of the Elektron Digitakt and broaden its functionality.

## Repository Structure

The main directories and files in this repository are:

* `dt_scw_chain/`: A directory containing the project's internal modules:
    * `app.py`: Contains core audio processing and wavetable chain creation functions.
    * `constants.py`: Defines project-wide constants, such as the target sample rate (`DEFAULT_SAMPLERATE`), target samples per SCW (`SCW_TARGET_SAMPLES = 367`), or default number of frames in a chain (`DEFAULT_NUM_FRAMES = 64`).
    * `dsp.py`: A module containing digital signal processing functions, including the `shuffle` function for wavetable frames.

* `batch_convert_scw.py`: The main user interface script for converting individual SCWs and creating wavetable chains from audio files.
* `batch_image_wavetables.py`: A script for converting graphic files into wavetable chains.
* `batch_waveedit_banks.py`: A script for processing entire WaveEdit banks.

* `input_scw_frames/`: Default input directory for individual SCW files for conversion or chain creation.
* `output_digitakt_scw/`: Default output directory for individually converted SCW files.
* `output_digitakt_chains/`: Default output directory for wavetable chains created from individual SCWs.

* `input_waveedit_banks/`: Default input directory for WaveEdit banks.
* `output_waveedit_banks/`: Default output directory for processed WaveEdit banks.

* `input_images/`: Default input directory for image files to be converted to wavetables.
* `output_images_wavetables/`: Default output directory for wavetables created from images.

* `pyproject.toml`: The Poetry configuration file, defining project metadata and dependencies.
* `poetry.lock`: A Poetry-generated file that locks the exact versions of dependencies, ensuring a reproducible environment.


````markdown
## Installation and Usage

The project uses [Poetry](https://python-poetry.org/) for dependency management. Below are instructions on how to set up the environment and run the tool.

### Prerequisites

* Python 3.x (as specified in `pyproject.toml`, currently `>=3.12,<3.14`)
* Poetry (Recommended for environment and dependency management)

### Step 1: Clone the Repository

```bash
git clone https://github.com/David303ttl/dt_scw_chain
cd DT_SCW_CHAIN
````

### Step 2: Install Dependencies using Poetry

The easiest way is to use Poetry. Make sure you have Poetry installed (instructions available on the [official Poetry website](https://www.google.com/search?q=https://python-poetry.org/docs/%23installation)).

```bash
poetry install
```

This command will install all dependencies listed in `pyproject.toml` and locked in `poetry.lock` into a dedicated virtual environment managed by Poetry.

### Step 3: Running the Tool

To ensure your scripts execute within the correct virtual environment with all necessary dependencies, it is recommended to run them using `poetry run`.

#### General Execution Command:

```bash
poetry run python your_script_name.py
```

#### Mode 1: Convert and Create SCW Chains from Audio Files (`batch_convert_scw.py`)

This script allows you to convert individual SCWs to the Digitakt format or combine them into 64-frame wavetable chains.

```bash
poetry run python batch_convert_scw.py
```

Upon execution, you will be prompted to select a mode:

  * `1`: Convert individual SCWs (output to `output_digitakt_scw` directory).
  * `2`: Create a wavetable chain from 64 SCW frames (output to `output_digitakt_chains` directory).

You will then be asked to provide input and output directory paths, and for chain creation, the output filename and shuffling options (number of groups, shuffle seed).
The input directory is read recursively\!
Option `2` reads the first 64 files in the specified directory\!

#### Mode 2: Convert WaveEdit Banks (`batch_waveedit_banks.py`)

This script is used to process entire WaveEdit banks.

```bash
poetry run python batch_waveedit_banks.py
```

You will be prompted to provide input directories (default: `input_waveedit_banks`) and output directories (default: `output_waveedit_banks`). The script will process all WaveEdit banks found in the input directory, converting them to the Digitakt format (64 frames x 367 samples @ 48 kHz).

#### Mode 3: Convert Images to Wavetable Chains (`batch_image_wavetables.py`)

This script enables creating wavetable chains directly from graphic files.

```bash
poetry run python batch_image_wavetables.py
```

You will be prompted to provide input directories (default: `input_images`) and output directories (default: `output_images_wavetables`). The script will process the graphic files, generating 64-frame wavetable chains from them.

-----

**Important Notes:**

  * Ensure your input files (.wav, .png/.jpg for images, .json for WaveEdit) are located in the appropriate input directories.
  * Output directories will be created automatically if they do not exist.

