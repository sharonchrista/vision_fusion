#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
#  transfer_to_server.sh
#  Run this from YOUR LOCAL MACHINE (where you downloaded the tar.gz)
#  Transfers all project files to sharon@172.21.1.138
#
#  Usage:
#      bash transfer_to_server.sh /path/to/liquidfusionnet_project.tar.gz
# ─────────────────────────────────────────────────────────────────────────────

set -e

SERVER="sharon@172.21.1.138"
REMOTE_HOME="/home/sharon"
TAR_PATH="${1:-./liquidfusionnet_project.tar.gz}"

if [ ! -f "$TAR_PATH" ]; then
    echo "ERROR: tar.gz not found at $TAR_PATH"
    echo "Usage: bash transfer_to_server.sh /path/to/liquidfusionnet_project.tar.gz"
    exit 1
fi

echo "── Step 1: Copy setup script to server ──────────────────────────────"
scp vision_fusion/setup_remote_dirs.sh $SERVER:$REMOTE_HOME/

echo "── Step 2: Run directory setup on server ────────────────────────────"
ssh $SERVER "bash $REMOTE_HOME/setup_remote_dirs.sh"

echo "── Step 3: Upload project archive ───────────────────────────────────"
scp "$TAR_PATH" $SERVER:$REMOTE_HOME/

echo "── Step 4: Extract archive on server ────────────────────────────────"
ssh $SERVER "cd $REMOTE_HOME && tar -xzf liquidfusionnet_project.tar.gz"

echo "── Step 5: Upload your zip data files ───────────────────────────────"
echo "  (Skipping — do this manually with the commands below)"
echo ""
echo "  Manual data upload commands — run from your local machine:"
echo "  scp /path/to/DATASET.zip                      $SERVER:$REMOTE_HOME/vision_fusion/data/raw/"
echo "  scp /path/to/Terrestrial_Radar.zip            $SERVER:$REMOTE_HOME/vision_fusion/data/raw/"
echo "  scp /path/to/TIAND_LiDAR-object\ detection.zip $SERVER:$REMOTE_HOME/vision_fusion/data/raw/"
echo "  scp /path/to/TIAND_MULTIMODAL_DATASET.zip     $SERVER:$REMOTE_HOME/vision_fusion/data/raw/"
echo ""
echo "✓  Transfer complete."
