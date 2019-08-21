#!/usr/bin/env bash
export PYTHONPATH=/data/code/crop
python infer.py --data-dir ./data/qc/20190805/ \
    --model-path ./results/2019-07-31_06-01-03/saved_models/save_30.pth \
    --batch-size 128 \
    --gpu-id 0 \
    --num-workers 2 \
    --out-file ./data/qc/20190805/output/task_output/crop_20190805.csv \
    --map-file ./data/qc/20190805/combined_20190805.csv
