#!/bin/bash
# run_experiments.sh

# Exit immediately if a command fails
set -e

echo "=========================================="
echo " Activating Virtual Environment"
echo "=========================================="
source venv/bin/activate

echo "=========================================="
echo " [STAGE 1] Automated Data Preprocessing"
echo "=========================================="

echo "--> Preprocessing TUSZ Dataset (Target: 40 Patients)"
python main.py --dataset tusz --stage preprocess --max-patients 40

echo "--> Preprocessing CHB-MIT Dataset (Target: All Sessions)"
python main.py --dataset chbmit --stage preprocess

# Define the mounted path for the training stage
DATA_DIR="~/transcend_drive/data/processed"

echo "=========================================="
echo " [STAGE 2] Automated GPU Training Pipeline"
echo "=========================================="

echo "--> [1/4] Running TUSZ CNN Baseline"
python src/training/train.py --data $DATA_DIR/tusz/ --max-patients 25 --model cnn --epochs 30 --batch-size 128

echo "--> [2/4] Running TUSZ TCN Model"
python src/training/train.py --data $DATA_DIR/tusz/ --max-patients 25 --model tcn --epochs 50 --batch-size 128

echo "--> [3/4] Running CHB-MIT CNN Baseline"
python src/training/train.py --data $DATA_DIR/chbmit/ --max-patients 1000 --model cnn --epochs 30 --batch-size 128

echo "--> [4/4] Running CHB-MIT TCN Model"
python src/training/train.py --data $DATA_DIR/chbmit/ --max-patients 1000 --model tcn --epochs 50 --batch-size 128

echo "=========================================="
echo " All pipelines (Preprocessing -> Training) completed successfully!"
echo "=========================================="