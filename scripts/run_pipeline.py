#!/usr/bin/env python3
"""
Lucid Calibration Pipeline

This script orchestrates the full calibration pipeline:
1. Discovers scenes and cameras in input data
2. Creates a working directory (preserving original data)
3. Crops wide-angle camera images (cam02, cam05, cam07)
4. Converts Lucid calibration JSON to CalibAnything format (with crop adjustments)
5. Generates SAM masks for segmentation (or uses pre-made masks)
6. Processes masks to filter and refine them
7. Runs the CalibAnything C++ calibration program
8. Outputs final calibration results

Prerequisites:
    Run ./setup.sh first to install dependencies (CalibAnything, lucid-sam)

Usage:
    python scripts/run_pipeline.py -i <input_data> -o <output_dir>
    python scripts/run_pipeline.py -i data/scenes -o output --skip-calibration
    python scripts/run_pipeline.py -i data/scenes -o output --force-sam
    python scripts/run_pipeline.py -i data/scenes -o output --masks-dir /path/to/masks
    python scripts/run_pipeline.py -i data/scenes -o output --processed-masks-dir /path/to/masks

The pipeline:
- Automatically crops wide-angle cameras before SAM processing
- Adjusts calibration intrinsics for crop offsets
- Supports caching at each step
- Supports pre-made masks (external or in source data)

Mask source priority (highest to lowest):
1. --processed-masks-dir (external processed masks, skips SAM + processing)
2. --masks-dir (external raw masks, skips SAM, runs processing)
3. Source data processed_masks (scene/camera/processed_masks/)
4. Source data masks (scene/camera/masks/)
5. Generate with SAM (default)

Expected mask directory structure:
    masks_dir/
    ├── image_stem_1/     # Matches image filename without extension
    │   ├── 0001.png
    │   ├── 0002.png
    │   └── ...
    └── image_stem_2/
        └── ...
"""

import argparse
import json
import os
import subprocess
import shutil
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from PIL import Image


# =============================================================================
# Configuration
# =============================================================================

# Paths relative to this script
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
EXTERNAL_DIR = PROJECT_ROOT / "external"
CALIBANYTHING_DIR = EXTERNAL_DIR / "CalibAnything"
LUCID_SAM_DIR = EXTERNAL_DIR / "lucid-sam"

# Default max dimension for SAM when not using CUDA (to reduce memory usage)
SAM_NON_CUDA_MAX_DIMENSION = 1024

# Wide-angle cameras that need cropping
WIDE_CAMERAS = ["cam02", "cam05", "cam07"]

# Crop configurations for wide-angle cameras
# For fixed resolution cameras: (left, top, right, bottom) pixel coordinates
# For variable resolution cameras: percentage-based with "type": "percent"
CROP_CONFIGS = {
    "cam02": {
        "type": "percent",
        "left": 7.5,
        "top": 16.0,
        "right": 93.0,
        "bottom": 75.0,
    },
    "cam05": {
        "type": "fixed",
        "left": 175,
        "top": 200,
        "right": 3560,
        "bottom": 2260,
    },
    "cam07": {
        "type": "fixed",
        "left": 175,
        "top": 200,
        "right": 3560,
        "bottom": 2260,
    },
}


# =============================================================================
# Environment Checks
# =============================================================================

def check_virtual_environment() -> bool:
    """Check if the script is running inside a virtual environment."""
    in_venv = sys.prefix != sys.base_prefix
    has_venv_var = 'VIRTUAL_ENV' in os.environ
    return in_venv or has_venv_var


def enforce_virtual_environment():
    """Exit with an error if not running in a virtual environment."""
    if not check_virtual_environment():
        print("=" * 60)
        print("ERROR: This script must be run inside a virtual environment!")
        print("=" * 60)
        print()
        print("Please run the setup script first:")
        print()
        print("  ./setup.sh")
        print("  source .venv/bin/activate")
        print()
        print("Then run this script again:")
        print("  python scripts/run_pipeline.py -i <input> -o <output>")
        print()
        sys.exit(1)


# =============================================================================
# Dependency Checks
# =============================================================================

def check_dependencies() -> bool:
    """Check that all required dependencies are installed. Exit with instructions if not."""
    missing = []

    # Check CalibAnything
    if not CALIBANYTHING_DIR.exists():
        missing.append("CalibAnything")
    
    # Check lucid-sam
    if not LUCID_SAM_DIR.exists():
        missing.append("lucid-sam")
    else:
        # Check lucid-sam venv is set up
        lucid_sam_venv = LUCID_SAM_DIR / ".venv" / "bin" / "python"
        if not lucid_sam_venv.exists():
            missing.append("lucid-sam virtual environment")

    if missing:
        print("=" * 60)
        print("ERROR: Missing dependencies!")
        print("=" * 60)
        print()
        print(f"Missing: {', '.join(missing)}")
        print()
        print("Please run the setup script first:")
        print()
        print("  ./setup.sh")
        print("  source .venv/bin/activate")
        print()
        print("Then run this script again.")
        print()
        return False
    
    return True


# =============================================================================
# Scene/Camera Discovery
# =============================================================================

def find_scenes(input_dir: Path) -> List[Path]:
    """Find all scene directories in the input directory."""
    scenes = []
    for item in input_dir.iterdir():
        if item.is_dir():
            scenes.append(item)
    return sorted(scenes)


def find_cameras(scene_dir: Path) -> List[str]:
    """Find all camera directories in a scene."""
    cameras = []
    for item in scene_dir.iterdir():
        if item.is_dir() and item.name.startswith('cam'):
            cameras.append(item.name)
    return sorted(cameras)


def find_pcd_files(scene_dir: Path) -> List[Path]:
    """Find all PCD files in a scene directory (not in subdirectories)."""
    pcd_files = []
    for item in scene_dir.iterdir():
        if item.is_file() and item.suffix == '.pcd':
            pcd_files.append(item)
    return sorted(pcd_files)


def is_wide_camera(camera_name: str) -> bool:
    """Check if a camera is a wide-angle camera that needs cropping."""
    return camera_name in WIDE_CAMERAS


# =============================================================================
# Image Cropping
# =============================================================================

def get_crop_box(camera_name: str, img_width: int, img_height: int) -> Tuple[int, int, int, int]:
    """
    Get crop box for a camera.

    Returns:
        Tuple of (left, top, right, bottom) pixel coordinates
    """
    config = CROP_CONFIGS.get(camera_name)
    if not config:
        raise ValueError(f"No crop config for camera: {camera_name}")

    if config.get("type") == "percent":
        left = int(img_width * config["left"] / 100)
        top = int(img_height * config["top"] / 100)
        right = int(img_width * config["right"] / 100)
        bottom = int(img_height * config["bottom"] / 100)
        return (left, top, right, bottom)
    else:
        return (config["left"], config["top"], config["right"], config["bottom"])


def crop_image_file(input_path: Path, output_path: Path, camera_name: str) -> Tuple[int, int]:
    """
    Crop an image and save to output path.

    Returns:
        Tuple of (crop_offset_x, crop_offset_y) - the top-left crop coordinates
    """
    with Image.open(input_path) as img:
        crop_box = get_crop_box(camera_name, img.width, img.height)
        cropped = img.crop(crop_box)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cropped.save(output_path, quality=95)
        return (crop_box[0], crop_box[1])  # left, top offsets


# =============================================================================
# SAM Mask Generation
# =============================================================================

def generate_sam_masks_for_image(
    image_path: Path,
    output_masks_dir: Path,
    lucid_sam_dir: Path = LUCID_SAM_DIR,
    max_dimension: Optional[int] = None
) -> bool:
    """Generate SAM masks for a single image using lucid-sam."""
    image_stem = image_path.stem
    image_path_abs = image_path.resolve()
    mask_output_dir_abs = (output_masks_dir / image_stem).resolve()
    lucid_sam_dir_abs = lucid_sam_dir.resolve()

    venv_python = lucid_sam_dir_abs / ".venv" / "bin" / "python"
    sam_script = lucid_sam_dir_abs / "sam.py"

    if not venv_python.exists():
        print(f"    Error: lucid-sam venv not found at {venv_python}")
        return False

    if not sam_script.exists():
        print(f"    Error: sam.py not found at {sam_script}")
        return False

    python_code = f'''
import sys
import os
import torch
import cv2
sys.path.insert(0, "{lucid_sam_dir_abs}")
from sam import process_image_with_sam
from models import SAMModel

# Read original image dimensions
original_img = cv2.imread("{image_path_abs}")
if original_img is None:
    raise FileNotFoundError(f"Failed to read image: {image_path_abs}")
orig_h, orig_w = original_img.shape[:2]
print(f"Original image size: {{orig_w}}x{{orig_h}}")

# Determine max_dimension: use provided value, or auto-downsample if not on CUDA
max_dim = {max_dimension if max_dimension else 'None'}
needs_upscale = False
if max_dim is None and not torch.cuda.is_available():
    max_dim = {SAM_NON_CUDA_MAX_DIMENSION}
    if max(orig_h, orig_w) > max_dim:
        needs_upscale = True
        print(f"CUDA not available, processing at max dimension {{max_dim}} (will upscale masks)")
elif max_dim is not None and max(orig_h, orig_w) > max_dim:
    needs_upscale = True
    print(f"Processing at max dimension {{max_dim}} (will upscale masks)")

masks = process_image_with_sam(
    image_path="{image_path_abs}",
    selected_model=SAMModel.VIT_L,
    visualize=False,
    output_masks=True,
    visualization_output_path=None,
    output_masks_dir="{mask_output_dir_abs}",
    mask_name_digits=4,
    mask_start_index=1,
    max_dimension=max_dim
)
print(f"Generated {{len(masks)}} masks")

# Upscale masks to original resolution if needed
if needs_upscale:
    print(f"Upscaling masks to original resolution {{orig_w}}x{{orig_h}}...")
    mask_dir = "{mask_output_dir_abs}"
    for mask_file in os.listdir(mask_dir):
        if mask_file.endswith('.png'):
            mask_path = os.path.join(mask_dir, mask_file)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                upscaled = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(mask_path, upscaled)
    print(f"Upscaled {{len(os.listdir(mask_dir))}} masks")
'''

    try:
        result = subprocess.run(
            [str(venv_python), '-c', python_code],
            check=True,
            capture_output=True,
            text=True,
            cwd=str(lucid_sam_dir_abs)
        )
        if result.stdout.strip():
            for line in result.stdout.strip().split('\n'):
                print(f"    [SAM] {line}")
        print(f"    Generated masks for {image_path.name} -> {mask_output_dir_abs}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"    Error generating masks for {image_path.name}: {e}")
        if e.stdout and e.stdout.strip():
            print(f"    stdout: {e.stdout}")
        if e.stderr and e.stderr.strip():
            print(f"    stderr: {e.stderr}")
        return False


def masks_exist_for_image(masks_dir: Path, image_name: str) -> bool:
    """Check if masks already exist for an image."""
    mask_subdir = masks_dir / image_name
    if not mask_subdir.exists():
        return False
    png_files = list(mask_subdir.glob("*.png"))
    return len(png_files) > 0


def all_masks_exist_for_images(images_dir: Path, masks_dir: Path) -> bool:
    """Check if all images have corresponding masks."""
    image_extensions = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'}
    image_files = [f for f in images_dir.iterdir()
                   if f.is_file() and f.suffix in image_extensions]

    if not image_files:
        return True

    for image_file in image_files:
        if not masks_exist_for_image(masks_dir, image_file.stem):
            return False
    return True


def generate_sam_masks_for_camera(
    images_dir: Path,
    masks_dir: Path,
    lucid_sam_dir: Path = LUCID_SAM_DIR,
    max_dimension: Optional[int] = None,
    force: bool = False
) -> bool:
    """Generate SAM masks for all images in a camera's images directory."""
    image_extensions = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'}
    image_files = [f for f in images_dir.iterdir()
                   if f.is_file() and f.suffix in image_extensions]
    image_files.sort()

    if not image_files:
        print(f"    No images found in {images_dir}")
        return True

    print(f"    Generating SAM masks for {len(image_files)} images...")

    all_success = True
    for image_file in image_files:
        image_stem = image_file.stem

        if not force and masks_exist_for_image(masks_dir, image_stem):
            print(f"    Masks already exist for {image_file.name}, skipping")
            continue

        if not generate_sam_masks_for_image(
            image_file, masks_dir, lucid_sam_dir, max_dimension
        ):
            all_success = False

    return all_success


# =============================================================================
# Working Directory Structure
# =============================================================================

def create_working_structure(output_dir: Path, scene_name: str, camera_name: str) -> Dict[str, Path]:
    """
    Create working directory structure for a camera.

    Structure:
        output/scene_cam/
        ├── working/            # Data for CalibAnything
        │   ├── images/         # Cropped images (wide) or copies (narrow)
        │   ├── masks/          # SAM-generated masks
        │   ├── processed_masks/# Filtered masks
        │   ├── pc/             # Point clouds
        │   └── calib.json      # Adjusted calibration
        └── results/            # Calibration outputs
    """
    base_dir = output_dir / f"{scene_name}_{camera_name}"
    base_dir.mkdir(parents=True, exist_ok=True)

    working_dir = base_dir / "working"
    results_dir = base_dir / "results"

    working_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    # Working subdirectories
    images_dir = working_dir / "images"
    masks_dir = working_dir / "masks"
    pc_dir = working_dir / "pc"
    processed_masks_dir = working_dir / "processed_masks"

    images_dir.mkdir(exist_ok=True)
    masks_dir.mkdir(exist_ok=True)
    pc_dir.mkdir(exist_ok=True)
    processed_masks_dir.mkdir(exist_ok=True)

    return {
        'base_dir': base_dir,
        'working_dir': working_dir,
        'results_dir': results_dir,
        'images_dir': images_dir,
        'masks_dir': masks_dir,
        'pc_dir': pc_dir,
        'processed_masks_dir': processed_masks_dir
    }


def prepare_images_and_pcds(
    scene_dir: Path,
    camera_name: str,
    dirs: Dict[str, Path],
    pcd_files: List[Path],
    file_names: List[str]
) -> Tuple[int, int]:
    """
    Prepare images and PCD files for processing.

    For wide cameras: crops images and returns crop offsets
    For narrow cameras: copies images as-is

    Returns:
        Tuple of (crop_offset_x, crop_offset_y) - (0, 0) for narrow cameras
    """
    cam_source_dir = scene_dir / camera_name
    crop_offset_x, crop_offset_y = 0, 0

    if not cam_source_dir.exists():
        print(f"  Warning: Camera directory does not exist: {cam_source_dir}")
        return (0, 0)

    # Find all image files
    image_extensions = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'}
    image_files = []

    for item in cam_source_dir.rglob('*'):
        if item.is_file() and item.suffix in image_extensions:
            image_files.append(item)

    image_files.sort()

    if len(image_files) == 0:
        print(f"  Warning: No image files found in {cam_source_dir}")
    else:
        for image_file in image_files:
            # Process to working folder (crop or copy)
            working_dest = dirs['images_dir'] / image_file.name

            if not working_dest.exists():
                if is_wide_camera(camera_name):
                    # Crop wide-angle camera images
                    crop_offset_x, crop_offset_y = crop_image_file(
                        image_file, working_dest, camera_name
                    )
                    print(f"  Cropped image: {image_file.name} (offset: {crop_offset_x}, {crop_offset_y})")
                else:
                    # Copy narrow camera images as-is
                    shutil.copy2(image_file, working_dest)
                    print(f"  Copied image: {image_file.name}")

    # Copy PCD files to working directory
    for pcd_file, file_name in zip(pcd_files, file_names):
        dest_pcd = dirs['pc_dir'] / f"{file_name}.pcd"
        if not dest_pcd.exists():
            shutil.copy2(pcd_file, dest_pcd)
            print(f"  Copied PCD: {pcd_file.name}")

    return (crop_offset_x, crop_offset_y)


def convert_calib_file(
    lucid_calib_path: Path,
    output_calib_path: Path,
    camera_name: str,
    file_names: List[str],
    crop_offset: Tuple[int, int] = (0, 0),
    params_file: Optional[Path] = None
) -> bool:
    """
    Convert lucid_calib.json to calib.json with crop offset adjustments.

    For cropped images, the principal point (cx, cy) needs to be adjusted
    by subtracting the crop offset.
    """
    convert_script = SCRIPT_DIR / "convert_lucid_calib.py"

    if not convert_script.exists():
        print(f"Error: convert_lucid_calib.py not found at {convert_script}")
        return False

    cmd = [
        'python3', str(convert_script),
        '-i', str(lucid_calib_path),
        '-o', str(output_calib_path),
        '-c', camera_name,
        '--files'
    ] + file_names

    # Apply crop offset adjustments to principal point
    if crop_offset[0] != 0 or crop_offset[1] != 0:
        # Note: cx_offset and cy_offset in convert_lucid_calib.py ADD to cx/cy
        # For cropping, we need to SUBTRACT the crop offset
        cmd.extend(['--cx-offset', str(-crop_offset[0])])
        cmd.extend(['--cy-offset', str(-crop_offset[1])])
        print(f"  Adjusting intrinsics for crop: cx_offset={-crop_offset[0]}, cy_offset={-crop_offset[1]}")

    if params_file and params_file.exists():
        cmd.extend(['--params-file', str(params_file)])
        print(f"  Using params from: {params_file}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"  Converted calibration file: {output_calib_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Error converting calibration file: {e}")
        print(f"  stdout: {e.stdout}")
        print(f"  stderr: {e.stderr}")
        return False


def process_masks(masks_dir: Path, processed_masks_dir: Path) -> bool:
    """Process masks using process_masks.py."""
    if not masks_dir.exists():
        return True

    mask_subdirs = [d for d in masks_dir.iterdir() if d.is_dir()]
    if len(mask_subdirs) == 0:
        print(f"  No mask subdirectories found in {masks_dir}, skipping")
        return True

    process_script = SCRIPT_DIR / "process_masks.py"

    if not process_script.exists():
        print(f"Warning: process_masks.py not found at {process_script}")
        return False

    # Remove processed_masks_dir if it exists
    if processed_masks_dir.exists():
        print(f"  Removing existing processed_masks directory")
        shutil.rmtree(processed_masks_dir)

    total_masks = sum(len(list(d.glob("*.png"))) for d in mask_subdirs)
    print(f"  Processing {len(mask_subdirs)} mask directories ({total_masks} total masks)...")

    cmd = [
        'python3', '-u', str(process_script),  # -u for unbuffered output
        '-i', str(masks_dir),
        '-o', str(processed_masks_dir)
    ]

    try:
        # Stream output in real-time
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        for line in process.stdout:
            print(f"    {line.rstrip()}")
        
        process.wait()
        
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)
        
        print(f"  Processed masks saved to: {processed_masks_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Error processing masks (exit code: {e.returncode})")
        return False


def processed_masks_exist(processed_masks_dir: Path) -> bool:
    """Check if processed masks directory has content."""
    if not processed_masks_dir.exists():
        return False
    for subdir in processed_masks_dir.iterdir():
        if subdir.is_dir():
            if any(subdir.glob("*.png")):
                return True
    return False


# =============================================================================
# External/Pre-made Mask Support
# =============================================================================

def copy_masks_directory(src_dir: Path, dest_dir: Path, image_stems: List[str]) -> int:
    """
    Copy masks from source directory to destination.
    
    Expects source structure:
        src_dir/
        ├── image_stem_1/
        │   ├── 0001.png
        │   └── ...
        └── image_stem_2/
            └── ...
    
    Args:
        src_dir: Source masks directory
        dest_dir: Destination masks directory
        image_stems: List of image stems to copy masks for (if empty, copy all)
    
    Returns:
        Number of mask directories copied
    """
    if not src_dir.exists():
        return 0
    
    copied_count = 0
    for subdir in src_dir.iterdir():
        if not subdir.is_dir():
            continue
        
        # If image_stems specified, only copy matching directories
        if image_stems and subdir.name not in image_stems:
            continue
        
        # Check if has mask files
        mask_files = list(subdir.glob("*.png"))
        if not mask_files:
            continue
        
        # Copy the directory
        dest_subdir = dest_dir / subdir.name
        if dest_subdir.exists():
            shutil.rmtree(dest_subdir)
        shutil.copytree(subdir, dest_subdir)
        copied_count += 1
    
    return copied_count


def find_source_masks(scene_dir: Path, camera_name: str) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Look for pre-existing masks in the source scene/camera directory.
    
    Checks for:
        scene/camera/processed_masks/  (takes priority)
        scene/camera/masks/
    
    Args:
        scene_dir: Scene directory path
        camera_name: Camera name (e.g., 'cam01')
    
    Returns:
        Tuple of (masks_path, processed_masks_path) - either may be None
    """
    cam_dir = scene_dir / camera_name
    
    masks_path = None
    processed_masks_path = None
    
    # Check for processed_masks first (higher priority)
    candidate = cam_dir / "processed_masks"
    if candidate.exists() and candidate.is_dir():
        # Verify it has content
        has_masks = any(
            subdir.is_dir() and any(subdir.glob("*.png"))
            for subdir in candidate.iterdir()
        )
        if has_masks:
            processed_masks_path = candidate
    
    # Check for raw masks
    candidate = cam_dir / "masks"
    if candidate.exists() and candidate.is_dir():
        has_masks = any(
            subdir.is_dir() and any(subdir.glob("*.png"))
            for subdir in candidate.iterdir()
        )
        if has_masks:
            masks_path = candidate
    
    return masks_path, processed_masks_path


def get_image_stems(images_dir: Path) -> List[str]:
    """Get list of image stems (filenames without extension) from images directory."""
    image_extensions = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'}
    stems = []
    for f in images_dir.iterdir():
        if f.is_file() and f.suffix in image_extensions:
            stems.append(f.stem)
    return sorted(stems)


def run_calibration(
    working_dir: Path,
    results_dir: Path,
    scene_name: str,
    camera_name: str
) -> Tuple[bool, Path]:
    """Run the CalibAnything C++ calibration executable on working directory."""
    executable = CALIBANYTHING_DIR / "bin" / "run_lidar2camera"
    calib_json_path = working_dir / "calib.json"

    if not executable.exists():
        print(f"  Error: Calibration executable not found at {executable}")
        print(f"  Please build CalibAnything first:")
        print(f"    cd {CALIBANYTHING_DIR}")
        print(f"    mkdir -p build && cd build")
        print(f"    cmake .. && make")
        return False, results_dir

    if not calib_json_path.exists():
        print(f"  Error: calib.json not found at {calib_json_path}")
        return False, results_dir

    # Convert to absolute paths before changing directory
    calib_json_path_abs = calib_json_path.resolve()
    results_dir_abs = results_dir.resolve()

    calib_data = json.load(open(calib_json_path_abs))
    file_names = calib_data.get('file_name', ['000000'])

    project_root = CALIBANYTHING_DIR
    original_cwd = Path.cwd()

    try:
        os.chdir(project_root)

        print(f"    Processing {len(file_names)} file(s): {', '.join(file_names)}")
        print(f"    Working directory: {working_dir}")
        print(f"    Running calibration (output streamed below)...")

        cmd = [str(executable), str(calib_json_path_abs)]
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        for line in process.stdout:
            print(f"    [Calib] {line.rstrip()}")

        process.wait()

        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)

        # Copy outputs to results directory
        output_files = ['extrinsic.txt', 'refined_proj_seg.png', 'refined_proj.png',
                        'init_proj.png', 'init_proj_seg.png']
        copied_files = []

        for filename in output_files:
            src = project_root / filename
            if src.exists():
                dst = results_dir_abs / filename
                shutil.copy2(src, dst)
                copied_files.append(filename)

        print(f"  Calibration complete. Results saved to: {results_dir_abs}")
        print(f"    Copied files: {', '.join(copied_files)}")
        return True, results_dir

    except subprocess.CalledProcessError as e:
        print(f"  Error running calibration (exit code: {e.returncode})")
        return False, results_dir
    finally:
        os.chdir(original_cwd)


# =============================================================================
# Scene Processing
# =============================================================================

def process_scene(
    scene_dir: Path,
    output_dir: Path,
    skip_calibration: bool = False,
    force_sam: bool = False,
    force_process_masks: bool = False,
    sam_max_dimension: Optional[int] = None,
    external_masks_dir: Optional[Path] = None,
    external_processed_masks_dir: Optional[Path] = None,
    camera_filter: Optional[List[str]] = None
):
    """
    Process a single scene.
    
    Mask source priority (highest to lowest):
        1. --processed-masks-dir (external processed masks)
        2. --masks-dir (external raw masks, will be processed)
        3. Source data processed_masks (scene/camera/processed_masks/)
        4. Source data masks (scene/camera/masks/)
        5. Generate with SAM (default)
    
    Args:
        camera_filter: If provided, only process cameras in this list
    """
    scene_name = scene_dir.name
    print(f"\n{'='*60}")
    print(f"Processing scene: {scene_name}")
    print(f"{'='*60}")

    # Find PCD files
    pcd_files = find_pcd_files(scene_dir)
    if len(pcd_files) == 0:
        print(f"  Warning: No PCD files found in {scene_dir}")
        return

    file_names = [pcd.stem for pcd in pcd_files]
    print(f"  Found {len(pcd_files)} PCD files: {file_names}")

    # Find cameras
    cameras = find_cameras(scene_dir)
    if len(cameras) == 0:
        print(f"  Warning: No camera directories found in {scene_dir}")
        return

    # Filter cameras if specified
    if camera_filter:
        available_cameras = cameras
        cameras = [c for c in cameras if c in camera_filter]
        if len(cameras) == 0:
            print(f"  Warning: No cameras matched filter: {camera_filter}")
            print(f"  Available cameras: {available_cameras}")
            return
        print(f"  Filtered to {len(cameras)} camera(s): {cameras}")
    else:
        print(f"  Found {len(cameras)} cameras: {cameras}")

    # Process each camera
    for camera_name in cameras:
        print(f"\n  Processing camera: {camera_name}")
        camera_type = "wide-angle" if is_wide_camera(camera_name) else "narrow"
        print(f"  Camera type: {camera_type}")

        # Create working directory structure
        dirs = create_working_structure(output_dir, scene_name, camera_name)

        # Prepare images (crop for wide cameras, copy for narrow)
        crop_offset = prepare_images_and_pcds(
            scene_dir, camera_name, dirs, pcd_files, file_names
        )

        # Convert calibration file with crop adjustments
        lucid_calib_path = scene_dir / camera_name / "lucid_calib.json"
        output_calib_path = dirs['working_dir'] / "calib.json"

        params_file = scene_dir / camera_name / "params.json"
        if not params_file.exists():
            params_file = None

        if not lucid_calib_path.exists():
            print(f"  Warning: lucid_calib.json not found at {lucid_calib_path}")
            continue

        if not convert_calib_file(
            lucid_calib_path, output_calib_path,
            camera_name, file_names, crop_offset, params_file
        ):
            print(f"  Failed to convert calibration file for {camera_name}")
            continue

        # Get image stems for mask matching
        image_stems = get_image_stems(dirs['images_dir'])
        
        # Determine mask source with priority
        # Priority: external_processed > external_raw > source_processed > source_raw > generate
        use_external_processed = False
        use_external_raw = False
        use_source_processed = False
        use_source_raw = False
        skip_sam = False
        skip_processing = False
        
        # Check for source masks (auto-detect)
        source_masks_path, source_processed_masks_path = find_source_masks(scene_dir, camera_name)
        
        # Priority 1: External processed masks (--processed-masks-dir)
        if external_processed_masks_dir and external_processed_masks_dir.exists():
            use_external_processed = True
            skip_sam = True
            skip_processing = True
            print(f"  Using external processed masks from: {external_processed_masks_dir}")
        
        # Priority 2: External raw masks (--masks-dir)
        elif external_masks_dir and external_masks_dir.exists():
            use_external_raw = True
            skip_sam = True
            print(f"  Using external masks from: {external_masks_dir}")
        
        # Priority 3: Source processed masks (scene/camera/processed_masks/)
        elif source_processed_masks_path:
            use_source_processed = True
            skip_sam = True
            skip_processing = True
            print(f"  Found processed masks in source: {source_processed_masks_path}")
        
        # Priority 4: Source raw masks (scene/camera/masks/)
        elif source_masks_path:
            use_source_raw = True
            skip_sam = True
            print(f"  Found masks in source: {source_masks_path}")
        
        # Copy external/source masks if applicable
        if use_external_processed:
            count = copy_masks_directory(
                external_processed_masks_dir, dirs['processed_masks_dir'], image_stems
            )
            print(f"  Copied {count} processed mask directories from external source")
        
        elif use_external_raw:
            count = copy_masks_directory(
                external_masks_dir, dirs['masks_dir'], image_stems
            )
            print(f"  Copied {count} mask directories from external source")
        
        elif use_source_processed:
            count = copy_masks_directory(
                source_processed_masks_path, dirs['processed_masks_dir'], image_stems
            )
            print(f"  Copied {count} processed mask directories from source")
        
        elif use_source_raw:
            count = copy_masks_directory(
                source_masks_path, dirs['masks_dir'], image_stems
            )
            print(f"  Copied {count} mask directories from source")
        
        # Generate SAM masks if needed (Priority 5: default)
        if not skip_sam:
            need_sam = force_sam or not all_masks_exist_for_images(
                dirs['images_dir'], dirs['masks_dir']
            )

            if need_sam:
                print(f"  Generating SAM masks...")
                generate_sam_masks_for_camera(
                    dirs['images_dir'],
                    dirs['masks_dir'],
                    lucid_sam_dir=LUCID_SAM_DIR,
                    max_dimension=sam_max_dimension,
                    force=force_sam
                )
            else:
                print(f"  SAM masks already exist, skipping (use --force-sam to regenerate)")

        # Process masks if needed
        if not skip_processing:
            need_process = force_process_masks or not processed_masks_exist(dirs['processed_masks_dir'])

            if need_process:
                process_masks(dirs['masks_dir'], dirs['processed_masks_dir'])
            else:
                print(f"  Processed masks already exist, skipping (use --force-process-masks to reprocess)")

        # Update calib.json to use processed_masks
        if dirs['processed_masks_dir'].exists():
            mask_subdirs = [d for d in dirs['processed_masks_dir'].iterdir() if d.is_dir()]
            if len(mask_subdirs) > 0:
                calib_data = json.load(open(output_calib_path))
                calib_data['mask_folder'] = 'processed_masks'
                json.dump(calib_data, open(output_calib_path, 'w'), indent=2)
                print(f"  Updated calib.json to use processed_masks")

        # Run calibration
        if not skip_calibration:
            print(f"  Running calibration...")
            success, results_path = run_calibration(
                dirs['working_dir'], dirs['results_dir'],
                scene_name, camera_name
            )
        else:
            print(f"  Skipping calibration (--skip-calibration flag)")


# =============================================================================
# Main
# =============================================================================

def main():
    enforce_virtual_environment()

    parser = argparse.ArgumentParser(
        description='Lucid Calibration Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (generates SAM masks)
  python scripts/run_pipeline.py -i data/scenes -o output

  # Process only a specific scene
  python scripts/run_pipeline.py -i data/scenes -o output --scene scene1

  # Process multiple specific scenes
  python scripts/run_pipeline.py -i data/scenes -o output --scene scene1 --scene scene2

  # Process only specific cameras within scenes
  python scripts/run_pipeline.py -i data/scenes -o output --camera cam02 --camera cam03

  # Combine scene and camera filters
  python scripts/run_pipeline.py -i data/scenes -o output --scene scene1 --camera cam02

  # Use pre-made processed masks (skips SAM and processing)
  python scripts/run_pipeline.py -i data/scenes -o output --processed-masks-dir /path/to/masks

  # Use pre-made raw masks (skips SAM, runs processing)
  python scripts/run_pipeline.py -i data/scenes -o output --masks-dir /path/to/masks

  # Skip calibration step (only prepare data and generate masks)
  python scripts/run_pipeline.py -i data/scenes -o output --skip-calibration

  # Force regenerate SAM masks
  python scripts/run_pipeline.py -i data/scenes -o output --force-sam

  # Force reprocess masks
  python scripts/run_pipeline.py -i data/scenes -o output --force-process-masks

  # Limit SAM image size (useful for large images or limited memory)
  python scripts/run_pipeline.py -i data/scenes -o output --sam-max-dimension 1024

Mask source priority (highest to lowest):
  1. --processed-masks-dir (external processed masks)
  2. --masks-dir (external raw masks)
  3. scene/camera/processed_masks/ (auto-detected in source)
  4. scene/camera/masks/ (auto-detected in source)
  5. Generate with SAM (default)

Expected mask directory structure:
  masks_dir/
  ├── image_stem_1/       # Matches image filename without extension
  │   ├── 0001.png
  │   └── ...
  └── image_stem_2/
      └── ...

Filtering:
  --scene NAME    Process only specified scene(s). Can be repeated.
  --camera NAME   Process only specified camera(s). Can be repeated.

Pipeline steps:
  1. Discover scenes and cameras (apply filters if specified)
  2. Create working directory (preserving original data)
  3. Crop wide-angle camera images (cam02, cam05, cam07)
  4. Convert lucid_calib.json to calib.json (with crop adjustments)
  5. Load masks (from external/source) or generate with SAM
  6. Process masks to filter and refine (if using raw masks)
  7. Run CalibAnything C++ calibration
  8. Output results to results/ directory

Output structure:
  output/scene_camera/
  ├── working/            # Data for CalibAnything
  │   ├── images/         # Cropped (wide) or copied (narrow)
  │   ├── masks/
  │   ├── processed_masks/
  │   ├── pc/
  │   └── calib.json
  └── results/            # Calibration outputs
      ├── extrinsic.txt
      └── refined_proj*.png
        """
    )
    parser.add_argument('-i', '--input', required=True, type=Path,
                        help='Input directory containing scene folders')
    parser.add_argument('-o', '--output', required=True, type=Path,
                        help='Output directory for results')
    parser.add_argument('--scene', action='append', dest='scenes', metavar='NAME',
                        help='Process only specified scene(s). Can be used multiple times.')
    parser.add_argument('--camera', action='append', dest='cameras', metavar='NAME',
                        help='Process only specified camera(s). Can be used multiple times.')
    parser.add_argument('--skip-calibration', action='store_true',
                        help='Skip the calibration step')
    parser.add_argument('--force-sam', action='store_true',
                        help='Force regenerate SAM masks even if they exist')
    parser.add_argument('--force-process-masks', action='store_true',
                        help='Force reprocess masks even if processed_masks exist')
    parser.add_argument('--sam-max-dimension', type=int, default=None,
                        help='Maximum image dimension for SAM processing')
    parser.add_argument('--masks-dir', type=Path, default=None,
                        help='Path to external raw masks directory (skips SAM generation)')
    parser.add_argument('--processed-masks-dir', type=Path, default=None,
                        help='Path to external processed masks directory (skips SAM and processing)')

    args = parser.parse_args()

    # Validate input
    if not args.input.exists():
        print(f"Error: Input directory does not exist: {args.input}")
        return 1

    # Check dependencies are installed
    if not check_dependencies():
        return 1

    # Find scenes
    scenes = find_scenes(args.input)
    if len(scenes) == 0:
        print(f"Error: No scene directories found in {args.input}")
        return 1

    # Filter scenes if --scene specified
    if args.scenes:
        scene_filter = set(args.scenes)
        scenes = [s for s in scenes if s.name in scene_filter]
        if len(scenes) == 0:
            print(f"Error: No scenes matched filter: {args.scenes}")
            print(f"Available scenes: {[s.name for s in find_scenes(args.input)]}")
            return 1
        print(f"\nFiltered to {len(scenes)} scene(s): {[s.name for s in scenes]}")
    else:
        print(f"\nFound {len(scenes)} scene(s): {[s.name for s in scenes]}")

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Process each scene
    for scene_dir in scenes:
        process_scene(
            scene_dir, args.output,
            skip_calibration=args.skip_calibration,
            force_sam=args.force_sam,
            force_process_masks=args.force_process_masks,
            sam_max_dimension=args.sam_max_dimension,
            external_masks_dir=args.masks_dir,
            external_processed_masks_dir=args.processed_masks_dir,
            camera_filter=args.cameras
        )

    print(f"\n{'='*60}")
    print(f"Pipeline complete! Results saved to: {args.output}")
    print(f"{'='*60}")

    return 0


if __name__ == '__main__':
    exit(main())
