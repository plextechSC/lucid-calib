# Lucid Calibration Pipeline

A calibration pipeline for processing Lucid camera data using SAM segmentation and CalibAnything optimization.

## Pipeline Overview

```
Input Data → Convert Calibration JSON → Crop Wide Cameras (optional) →
SAM Mask Generation → Mask Processing → CalibAnything C++ Calibration → Final Results
```

## Prerequisites

### System Dependencies

The CalibAnything C++ calibration requires the following libraries:

**macOS (Homebrew):**
```bash
brew install cmake pcl opencv eigen boost jsoncpp
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install cmake libpcl-dev libopencv-dev libeigen3-dev libboost-all-dev libjsoncpp-dev
```

**Other requirements:**
- Python 3.8+
- Git

## Quick Start

```bash
# 1. Clone this repository
git clone <this-repo-url> lucid-calib
cd lucid-calib

# 2. Run setup (creates venv, clones dependencies, builds C++, installs packages)
./setup.sh

# 3. Activate the virtual environment
source .venv/bin/activate

# 4. Run the pipeline
python scripts/run_pipeline.py -i <input_data> -o <output_dir>
```

## Input Data Format

The pipeline expects data in the following structure:

```
input_data/
├── scene1/
│   ├── 000000.pcd
│   ├── 000001.pcd
│   ├── ...
│   ├── cam02/
│   │   ├── 000000.png
│   │   ├── 000001.png
│   │   └── lucid_calib.json
│   ├── cam03/
│   │   ├── 000000.png
│   │   └── lucid_calib.json
│   └── ...
└── scene2/
    └── ...
```

## Output Structure

```
output/
├── scene1_cam02/
│   ├── images/           # Copied input images
│   ├── pc/               # Copied point clouds
│   ├── masks/            # SAM-generated masks
│   ├── processed_masks/  # Filtered masks
│   └── calib.json        # Converted calibration
├── scene1_cam03/
│   └── ...
└── calibration_results/
    ├── scene1_cam02_000000/
    │   ├── extrinsic.txt
    │   ├── refined_proj.png
    │   └── refined_proj_seg.png
    └── ...
```

## Usage

### Full Pipeline

```bash
python scripts/run_pipeline.py -i data/scenes -o output
```

### Skip Calibration (prepare data and masks only)

```bash
python scripts/run_pipeline.py -i data/scenes -o output --skip-calibration
```

### Force Regenerate Masks

```bash
python scripts/run_pipeline.py -i data/scenes -o output --force-sam
```

### Limit SAM Image Size (for memory constraints)

```bash
python scripts/run_pipeline.py -i data/scenes -o output --sam-max-dimension 1024
```

## Individual Scripts

### Convert Calibration JSON

```bash
python scripts/convert_lucid_calib.py -i input/lucid_calib.json -o output/calib.json
```

### Crop Wide-Angle Images

```bash
python scripts/crop_images.py --data-dir data --output-dir cropped
```

### Process Masks

```bash
python scripts/process_masks.py -i masks_dir -o processed_masks_dir
```

## Camera Mapping

| Camera ID | Name  | Type | Description              |
|-----------|-------|------|--------------------------|
| cam-02    | FWC_C | Wide | Front Wide Camera Center |
| cam-03    | FNC   | Narrow | Front Narrow Camera    |
| cam-04    | RNC_R | Narrow | Rear Narrow Right      |
| cam-05    | FWC_R | Wide | Front Wide Camera Right  |
| cam-06    | RNC_C | Narrow | Rear Narrow Center     |
| cam-07    | FWC_L | Wide | Front Wide Camera Left   |
| cam-08    | RNC_L | Narrow | Rear Narrow Left       |

## Caching

The pipeline automatically caches intermediate results:

- **SAM masks**: Stored in `masks/` directory, skipped if they exist
- **Processed masks**: Stored in `processed_masks/`, skipped if they exist

Use `--force-sam` or `--force-process-masks` to regenerate.

## Dependencies

### System Libraries (install before running setup.sh)

| Library | Purpose | macOS | Ubuntu |
|---------|---------|-------|--------|
| CMake | Build system | `brew install cmake` | `apt install cmake` |
| PCL | Point Cloud Library | `brew install pcl` | `apt install libpcl-dev` |
| OpenCV | Computer vision | `brew install opencv` | `apt install libopencv-dev` |
| Eigen | Linear algebra | `brew install eigen` | `apt install libeigen3-dev` |
| Boost | C++ utilities | `brew install boost` | `apt install libboost-all-dev` |
| JsonCpp | JSON parsing | `brew install jsoncpp` | `apt install libjsoncpp-dev` |

### External Repositories (automatically cloned by setup.sh)

- [CalibAnything](https://github.com/OpenCalib/CalibAnything) - C++ calibration optimization
- [lucid-sam](https://github.com/plextechSC/lucid-sam) - SAM mask generation

### Python Packages (automatically installed by setup.sh)

- numpy >= 1.21.0
- opencv-python >= 4.5.0
- scipy >= 1.7.0
- Pillow >= 8.0.0
- PyTorch (installed in lucid-sam venv for SAM)

## Directory Structure

```
lucid-calib/
├── .venv/                    # Python virtual environment
├── external/
│   ├── CalibAnything/        # Cloned CalibAnything repo
│   └── lucid-sam/            # Cloned lucid-sam repo
├── scripts/
│   ├── run_pipeline.py       # Main pipeline orchestrator
│   ├── convert_lucid_calib.py
│   ├── crop_images.py
│   └── process_masks.py
├── requirements.txt
├── setup.sh
└── README.md
```
