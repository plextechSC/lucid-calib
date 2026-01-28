# Data Format Specification

This document describes the expected input data format for the Lucid Calibration Pipeline.

## Input Data Format

The pipeline expects data organized with point clouds at the scene level and images within camera subdirectories:

```
input_data/
├── scene1/
│   ├── 000000.pcd              # Point cloud frame 0
│   ├── 000001.pcd              # Point cloud frame 1
│   ├── ...
│   ├── cam02/
│   │   ├── 000000.png          # Image frame 0
│   │   ├── 000001.png          # Image frame 1
│   │   ├── ...
│   │   └── lucid_calib.json    # Camera calibration parameters
│   ├── cam03/
│   │   ├── 000000.png
│   │   ├── ...
│   │   └── lucid_calib.json
│   └── cam04/
│       ├── 000000.png
│       ├── ...
│       └── lucid_calib.json
├── scene2/
│   └── ...
└── ...
```

## File Formats

### Point Cloud Files (`.pcd`)

Standard PCD (Point Cloud Data) format with intensity values. The calibration algorithm uses intensity information for plane extraction and matching.

### Image Files (`.png`)

RGB images in PNG format. File naming should match the corresponding point cloud frames (e.g., `000000.pcd` corresponds to `000000.png`).

### Input Calibration File (`lucid_calib.json`)

JSON file containing camera intrinsic and extrinsic parameters from Lucid cameras:

```json
{
  "intrinsic_params": {
    "fx": 4567.32,
    "fy": 4566.75,
    "cx": 1915.45,
    "cy": 1103.16,
    "k1": -0.056582,
    "k2": 0.091158,
    "k3": 0.103344,
    "k4": 0.261091,
    "k5": 0,
    "k6": 0,
    "p1": -0.000124,
    "p2": 0.000981,
    "camera": "fnc_c",
    "camera_model": "normal"
  },
  "extrinsic_params": {
    "roll": -89.22,
    "pitch": 0.10,
    "yaw": -89.71,
    "px": 2.46,
    "py": -0.01,
    "pz": -0.53,
    "quaternion": {
      "x": -0.497,
      "y": 0.500,
      "z": -0.502,
      "w": 0.501
    },
    "camera_coordinate": "OPTICAL",
    "translation_error": 0.043,
    "rotation_error": 2.52,
    "reprojection_error": 34.44
  }
}
```

#### Intrinsic Parameters

| Parameter | Description |
|-----------|-------------|
| `fx`, `fy` | Focal lengths in pixels |
| `cx`, `cy` | Principal point coordinates in pixels |
| `k1`-`k6` | Radial distortion coefficients (OpenCV convention) |
| `p1`, `p2` | Tangential distortion coefficients |
| `camera` | Camera identifier string |
| `camera_model` | Camera model type (`"normal"` for standard pinhole) |

#### Extrinsic Parameters

| Parameter | Description |
|-----------|-------------|
| `roll`, `pitch`, `yaw` | Euler angles in degrees |
| `px`, `py`, `pz` | Translation vector in meters |
| `quaternion` | Alternative rotation representation (x, y, z, w) |
| `camera_coordinate` | Coordinate frame convention (`"OPTICAL"` for standard camera frame) |

## Camera Mapping

| Camera ID | Name  | Type   | Description              |
|-----------|-------|--------|--------------------------|
| cam-02    | FWC_C | Wide   | Front Wide Camera Center |
| cam-03    | FNC   | Narrow | Front Narrow Camera      |
| cam-04    | RNC_R | Narrow | Rear Narrow Right        |
| cam-05    | FWC_R | Wide   | Front Wide Camera Right  |
| cam-06    | RNC_C | Narrow | Rear Narrow Center       |
| cam-07    | FWC_L | Wide   | Front Wide Camera Left   |
| cam-08    | RNC_L | Narrow | Rear Narrow Left         |

Wide cameras are automatically cropped by the pipeline to reduce distortion effects.

## Naming Conventions

- **Scene directories**: Use descriptive names (e.g., `scene1`, `daytimetraffic`, `parking_lot`)
- **Camera directories**: Use consistent naming matching the camera IDs (e.g., `cam02`, `cam03`)
- **Frame numbering**: Use zero-padded 6-digit numbers (e.g., `000000`, `000001`)
- **File extensions**: `.pcd` for point clouds, `.png` for images, `.json` for calibration

## Format Conversion

The pipeline automatically converts `lucid_calib.json` to the `calib.json` format required by CalibAnything. This conversion is handled by `scripts/convert_lucid_calib.py` and produces output in each working directory.

To manually convert a calibration file:

```bash
python scripts/convert_lucid_calib.py -i input/lucid_calib.json -o output/calib.json
```

See the main [README.md](README.md) for details on the output `calib.json` format and tuning parameters.
