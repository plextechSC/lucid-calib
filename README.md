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
│   │   ├── lucid_calib.json
│   │   ├── masks/              # (optional) Raw SAM masks
│   │   │   ├── 000000.png
│   │   │   └── ...
│   │   └── processed_masks/    # (optional) Pre-processed masks
│   │       ├── 000000.png
│   │       └── ...
│   ├── cam03/
│   │   ├── 000000.png
│   │   └── lucid_calib.json
│   └── ...
└── scene2/
    └── ...
```

### Using Pre-made Masks

If you have pre-generated masks, place them in the input data folder and the pipeline will automatically detect and use them:

- **`processed_masks/`** - Skips both SAM generation and mask processing (fastest)
- **`masks/`** - Skips SAM generation, still runs mask processing

This is useful when you have manually created masks or want to reuse masks from a previous run.

## Output Structure

```
output/
├── scene1_cam02/
│   ├── working/              # Data for CalibAnything
│   │   ├── images/           # Cropped (wide) or copied (narrow) images
│   │   ├── pc/               # Copied point clouds
│   │   ├── masks/            # SAM-generated masks
│   │   ├── processed_masks/  # Filtered masks for calibration
│   │   └── calib.json        # Calibration config (TUNE THIS!)
│   └── results/              # Calibration outputs
│       ├── extrinsic.txt     # Refined extrinsic matrix
│       ├── init_proj.png     # Initial projection visualization
│       ├── init_proj_seg.png
│       ├── refined_proj.png  # Refined projection visualization
│       └── refined_proj_seg.png
├── scene1_cam03/
│   └── ...
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

### Use Pre-made Masks

The easiest way is to place masks directly in your input data (see [Input Data Format](#input-data-format)). Alternatively, specify an external masks directory:

```bash
# Use external processed masks (skips SAM and mask processing)
python scripts/run_pipeline.py -i data/scenes -o output --processed-masks-dir /path/to/masks

# Use external raw masks (skips SAM, runs processing)
python scripts/run_pipeline.py -i data/scenes -o output --masks-dir /path/to/masks
```

## Tuning Calibration Parameters

CalibAnything requires **scene-specific parameter tuning** for optimal results. The key parameters are in the `params` section of `output/<scene>_<camera>/working/calib.json`:

```json
"params": {
  "min_plane_point_num": 2000,
  "cluster_tolerance": 0.25,
  "search_num": 4000,
  "search_range": {
    "rot_deg": 5,
    "trans_m": 0.5
  },
  "point_range": {
    "top": 0.0,
    "bottom": 1.0
  },
  "down_sample": {
    "is_valid": false,
    "voxel_m": 0.05
  },
  "thread": {
    "is_multi_thread": true,
    "num_thread": 8
  }
}
```

### Parameter Reference

| Parameter | Description | Default | Typical Range |
|-----------|-------------|---------|---------------|
| `min_plane_point_num` | Minimum points required to detect a plane | 2000 | 500-5000 |
| `cluster_tolerance` | Distance threshold for point clustering (meters) | 0.25 | 0.1-0.5 |
| `search_num` | Number of random search iterations | 4000 | 1000-10000 |
| `search_range.rot_deg` | Rotation search range (degrees) | 5 | 1-10 |
| `search_range.trans_m` | Translation search range (meters) | 0.5 | 0.1-1.0 |
| `point_range.top` | Top percentage of points to exclude (0=none) | 0.0 | 0.0-0.5 |
| `point_range.bottom` | Bottom percentage of points to include (1=all) | 1.0 | 0.5-1.0 |
| `down_sample.is_valid` | Enable voxel grid downsampling | false | true/false |
| `down_sample.voxel_m` | Voxel size for downsampling (meters) | 0.05 | 0.02-0.1 |
| `thread.num_thread` | Number of parallel threads | 8 | 1-16 |

### Tuning Tips

- **Initial calibration is far off**: Increase `search_range` (rot_deg: 10, trans_m: 1.0)
- **Calibration is close but noisy**: Decrease `search_range` and increase `search_num`
- **Processing is too slow**: Enable `down_sample` or reduce `search_num`
- **Too few planes detected**: Decrease `min_plane_point_num`
- **Focus on ground plane**: Set `point_range.top: 0.3` to exclude sky/upper points

### Re-running Calibration

After modifying `calib.json`, re-run the pipeline (masks are cached, only calibration runs):

```bash
python scripts/run_pipeline.py -i data/scenes -o output
```

Or run CalibAnything directly:

```bash
cd external/CalibAnything
./bin/run_lidar2camera /absolute/path/to/output/scene_cam/working/calib.json
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
