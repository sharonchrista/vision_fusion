#!/bin/bash
# Watches for blob files and extracts + preprocesses them automatically
# Run in background: nohup bash scripts/watch_and_extract_blobs.sh &

NUSCENES_DIR="$HOME/vision_fusion/data/nuscenes"
SHARDS_OUT="$HOME/vision_fusion/data/shards/nuscenes_trainval"
LOG="$HOME/vision_fusion/results/logs/blob_extract.log"

mkdir -p "$SHARDS_OUT"
cd "$NUSCENES_DIR"

echo "[$(date)] Blob watcher started" >> "$LOG"

BLOBS="v1.0-trainval_meta.tgz v1.0-trainval01_blobs.tgz v1.0-trainval02_blobs.tgz v1.0-trainval03_blobs.tgz v1.0-trainval04_blobs.tgz"

for blob in $BLOBS; do
    while [ ! -f "$NUSCENES_DIR/$blob" ]; do
        echo "[$(date)] Waiting for $blob ..." >> "$LOG"
        sleep 60
    done
    echo "[$(date)] Found $blob — extracting ..." >> "$LOG"
    tar -xzf "$blob" && echo "[$(date)] $blob extracted OK" >> "$LOG"
done

echo "[$(date)] All blobs extracted. Starting trainval preprocessing ..." >> "$LOG"

cd "$HOME/vision_fusion"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate liquidfusion

python src/preprocessing/nuscenes/preprocess_nuscenes.py \
    --dataroot data/nuscenes \
    --version  v1.0-trainval \
    --output   "$SHARDS_OUT" \
    >> "$LOG" 2>&1

echo "[$(date)] Trainval preprocessing complete." >> "$LOG"
echo "[$(date)] Shards: $(ls $SHARDS_OUT/*.h5 | wc -l)" >> "$LOG"
