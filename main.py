import io
import os
import logging
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image, ImageCms, UnidentifiedImageError
from pillow_heif import register_heif_opener


def convert_single_file(heic_path, jpg_path, output_quality, dry) -> tuple:
    """
    Convert a single HEIC file to JPG format.
    
    #### Args:
        - heic_path (str): Path to the HEIC file.
        - jpg_path (str): Path to save the converted JPG file.
        - output_quality (int): Quality of the output JPG image.

    #### Returns:
        - tuple: Path to the HEIC file and conversion status.
    """

    try:
        if not dry:
            with Image.open(heic_path) as image:
                # Automatically handle and preserve EXIF metadata
                exif_data = image.info.get("exif")
                icc_profile_data = image.info.get("icc_profile")
                srgb_profile = ImageCms.createProfile("sRGB")

                if icc_profile_data:
                    # Load the source ICC profile
                    input_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc_profile_data))
                    # Convert image to sRGB
                    image = ImageCms.profileToProfile(
                        image,
                        input_profile,
                        srgb_profile,
                        renderingIntent=ImageCms.Intent.PERCEPTUAL,
                    )
                image = image.convert("RGB")
                image.save(
                    jpg_path,
                    "JPEG",
                    quality=output_quality,
                    exif=exif_data,
                    icc_profile=ImageCms.ImageCmsProfile(srgb_profile).tobytes(),
                    keep_rgb=True,
                    optimize=True,
                    subsampling=0,
                )
                # Preserve the original access and modification timestamps
                heic_stat = os.stat(heic_path)
                os.utime(jpg_path, (heic_stat.st_atime, heic_stat.st_mtime))
        return heic_path, True  # Successful conversion
    except (UnidentifiedImageError, FileNotFoundError, OSError) as e:
        logging.error(f"Error converting '{heic_path}': {e}")
        return heic_path, False  # Failed conversion


def convert_heic_to_jpg_async(executor, futures, heic_dir, output_quality, dry) -> (int, int):
    """
    Converts HEIC images in a directory to JPG format using parallel processing.

    #### Args:
        - heic_dir (str): Path to the directory containing HEIC files.
        - output_quality (int, optional): Quality of the output JPG images (1-100). Defaults to 50.
        - max_workers (int, optional): Number of parallel threads. Defaults to 4.
    """

    submits = 0
    skips = 0
    if not os.path.isdir(heic_dir):
        logging.error(f"Directory '{heic_dir}' does not exist.")
        return submits, skips

    sub_dirs = [dir for dir in os.listdir(heic_dir) if os.path.isdir(os.path.join(heic_dir, dir))]
    for sub_dir in sub_dirs:
        sub_submits, sub_skips = convert_heic_to_jpg_async(
            executor, futures, os.path.join(heic_dir, sub_dir), output_quality, dry)
        submits += sub_submits
        skips += sub_skips

    # Get all HEIC files in the specified directory
    heic_files = [file for file in os.listdir(heic_dir) if file.lower().endswith(".heic")]
    total_files = len(heic_files)

    if total_files == 0:
        return submits, skips

    # Prepare file paths for conversion
    tasks = []
    for file_name in heic_files:
        heic_path = os.path.join(heic_dir, file_name)
        jpg_path = os.path.join(heic_dir, os.path.splitext(file_name)[0] + ".jpg")

        # Skip conversion if the JPG already exists
        if os.path.exists(jpg_path):
            skips += 1
            logging.info(f"Skipping '{heic_path}' as the JPG already exists.")
            continue

        # Convert HEIC files to JPG in parallel using ThreadPoolExecutor
        task = executor.submit(convert_single_file, heic_path, jpg_path, output_quality, dry)
        futures[task] = heic_path
        submits += 1

    return submits, skips


def convert_heic_to_jpg(executor, heic_dir, output_quality, dry) -> None:
    futures = {}
    submits, skips = convert_heic_to_jpg_async(executor, futures, heic_dir, output_quality, dry)

    converts = 0
    errors = 0
    for future in as_completed(futures):
        heic_file = futures[future]
        try:
            _, success = future.result()
            if success:
                converts += 1
        except Exception as e:
            errors += 1
            logging.error(f"Error occurred during conversion of '{heic_file}': {e}")

    logging.info(f"Conversion completed. {submits} files, {converts} OK, {skips} SKIP, {errors} ERR.")


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format="%(asctime)s\t| %(levelname)s\t| %(message)s")

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Converts HEIC images to JPG format.",
                                     usage="%(prog)s [options] <heic_directory>",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("heic_dir", type=str, help="Path to the directory containing HEIC images.")
    parser.add_argument("-q", "--quality", type=int, default=90, help="Output JPG image quality (1-100). Default is 50.")
    parser.add_argument("-w", "--workers", type=int, default=4, help="Number of parallel workers for conversion.")
    parser.add_argument("-d", "--dry", type=bool, default=False, help="Dry run mode. Do not execute conversion.")

    parser.epilog = """
    Example usage:
      %(prog)s /path/to/your/heic/images -q 90 -w 8
    """

    # If no arguments provided, print help message
    try:
        args = parser.parse_args()
    except SystemExit:
        print(parser.format_help())
        exit()

    register_heif_opener()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Convert HEIC to JPG with parallel processing
        convert_heic_to_jpg(executor, args.heic_dir, args.quality, args.dry)
