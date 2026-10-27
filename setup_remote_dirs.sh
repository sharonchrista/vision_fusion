#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
#  LiquidFusionNet — Remote server directory setup
#  Server: sharon@172.21.1.138
#  Run this FIRST on the remote server:
#      bash setup_remote_dirs.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e
PROJECT_ROOT="$HOME/vision_fusion"

echo "Creating LiquidFusionNet project tree at $PROJECT_ROOT ..."

# ── Data directories ──────────────────────────────────────────────────────────
mkdir -p "$PROJECT_ROOT/data/raw"
mkdir -p "$PROJECT_ROOT/data/extracted/camera"
mkdir -p "$PROJECT_ROOT/data/extracted/radar"
mkdir -p "$PROJECT_ROOT/data/extracted/lidar"
mkdir -p "$PROJECT_ROOT/data/extracted/imu_gps"
mkdir -p "$PROJECT_ROOT/data/extracted/multimodal"
mkdir -p "$PROJECT_ROOT/data/processed/camera"
mkdir -p "$PROJECT_ROOT/data/processed/radar"
mkdir -p "$PROJECT_ROOT/data/processed/lidar"
mkdir -p "$PROJECT_ROOT/data/processed/imu_gps"
mkdir -p "$PROJECT_ROOT/data/shards/train"
mkdir -p "$PROJECT_ROOT/data/shards/val"
mkdir -p "$PROJECT_ROOT/data/shards/test"
mkdir -p "$PROJECT_ROOT/data/splits"

# ── Source directories ────────────────────────────────────────────────────────
mkdir -p "$PROJECT_ROOT/src/preprocessing"
mkdir -p "$PROJECT_ROOT/src/models/encoders"
mkdir -p "$PROJECT_ROOT/src/models/fusion"
mkdir -p "$PROJECT_ROOT/src/models/heads"
mkdir -p "$PROJECT_ROOT/src/training"
mkdir -p "$PROJECT_ROOT/src/evaluation"
mkdir -p "$PROJECT_ROOT/src/utils"
mkdir -p "$PROJECT_ROOT/src/data"

# ── Other directories ─────────────────────────────────────────────────────────
mkdir -p "$PROJECT_ROOT/configs"
mkdir -p "$PROJECT_ROOT/scripts"
mkdir -p "$PROJECT_ROOT/notebooks"
mkdir -p "$PROJECT_ROOT/tests"
mkdir -p "$PROJECT_ROOT/results/checkpoints"
mkdir -p "$PROJECT_ROOT/results/logs"
mkdir -p "$PROJECT_ROOT/results/metrics"
mkdir -p "$PROJECT_ROOT/results/figures"

# ── __init__.py files (make every src subfolder a Python package) ─────────────
touch "$PROJECT_ROOT/src/__init__.py"
touch "$PROJECT_ROOT/src/preprocessing/__init__.py"
touch "$PROJECT_ROOT/src/models/__init__.py"
touch "$PROJECT_ROOT/src/models/encoders/__init__.py"
touch "$PROJECT_ROOT/src/models/fusion/__init__.py"
touch "$PROJECT_ROOT/src/models/heads/__init__.py"
touch "$PROJECT_ROOT/src/training/__init__.py"
touch "$PROJECT_ROOT/src/evaluation/__init__.py"
touch "$PROJECT_ROOT/src/utils/__init__.py"
touch "$PROJECT_ROOT/src/data/__init__.py"
touch "$PROJECT_ROOT/tests/__init__.py"

echo ""
echo "✓  Directory tree created at $PROJECT_ROOT"
echo ""
echo "Full structure:"
find "$PROJECT_ROOT" -type d | sort | sed "s|$PROJECT_ROOT|vision_fusion|"
