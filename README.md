# HEIC to JPG Converter

Fork of [dragonGR/PyHEIC2JPG](https://github.com/dragonGR/PyHEIC2JPG). I've made the following changes:

1. *Add recursion*: recursively convert every HEIC in a directory.
2. *Concurrency*: Modify the concurrency model to maintain performance after the recursion implementation.
3. *Color profiles*: Extract the color profile from HEIC images and convert them to their sRGB equivalent for the JPGs.
Before this the colors in the JPG did not match the HEIC.
4. *New options*: Add the `--dry` and `--remove-originals` options to perform dry runs and remove the original images after conversion, respectively.
5. *Change defaults*: for quality and the new options.
6. *Logging*: Change the logging format.

---

This Python script efficiently converts HEIC (High-Efficiency Image Format) files to JPG format, now with parallel processing for faster conversions, enhanced user experience, and automatic preservation of EXIF metadata.

## Key Improvements
- Parallel Processing: Utilizes ``ThreadPoolExecutor`` for concurrent conversion of HEIC files, improving the speed of batch conversions.
- Command-Line Interface (CLI): Interact with the script directly using command-line arguments for easier execution and fine-tuning.
- Optimized Image Processing: Leverages the ``pillow-heif`` library for efficient HEIC processing.
- Improved Error Handling: More robust error management ensures smoother execution.
- Progress Tracking: Track the conversion progress for large file sets.
- Enhanced Folder Management: Confirms user intent before potentially overwriting existing output directories.
- Quality Control: Customize the output JPG quality (1-100) with a simple command-line option.
- EXIF Metadata Preservation: Retains important metadata such as camera details, timestamps, and location information in the converted JPG files.

## What this code does ?

This script allows you to seamlessly convert multiple HEIC files into JPG format with control over the quality and parallel processing options.
- Automated Output Directory: Creates a dedicated folder named "ConvertedFiles" within the HEIC directory to store the converted JPGs.
- Parallel Conversion: Process multiple files simultaneously by specifying the number of workers using the ``-w`` argument.
- Quality Control: Specify the desired JPG image quality using the ``-q``  argument.
- EXIF Metadata Preservation: Automatically extracts and includes metadata from HEIC files into the output JPGs.
- Error Handling: Skips individual files that encounter errors without halting the entire process.

## Installation

1. Install the required dependency: `pip install pillow pillow-heif`
2. Run the Script:
```bash
python main.py <path/to/your/heic/directory>
```

### Set JPG quality (1-100):
```bash
python main.py -q 90 <path/to/your/heic/directory>
```

### Set the number of parallel workers:
```bash
python main.py -w 8 <path/to/your/heic/directory>
```

### Combine quality and parallel workers:
```bash
python main.py -q 90 -w 8 <path/to/your/heic/directory>
```

## Features
- Parallel Processing: Convert multiple HEIC files concurrently for faster performance.
- User-friendly CLI for efficient conversion.
- Optimized HEIC processing for faster performance.
- Robust error handling for a smooth user experience.
- Informative progress tracking.
- Flexible output directory management.
- Quality control for JPG output.
- EXIF Metadata Preservation: Retains critical metadata like camera details, timestamps, and location data in the converted files.

This script provides a highly efficient and flexible solution for converting HEIC images to JPG format, making it ideal for both small and large-scale conversions.