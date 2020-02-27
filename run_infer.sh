#!/usr/bin/env bash
source ~/.profile
source ~/.bashrc
export PATH="/home/raman/Software/anaconda3/bin:$PATH"
#conda init bash
source activate py37dl
export PYTHONPATH=/data/code/crop
#python infer.py --data-dir ./data/qc/20190805/ \
#    --model-path ./results/2019-07-31_06-01-03/saved_models/save_30.pth \
#    --batch-size 128 \
#    --gpu-id 0 \
#    --num-workers 2 \
#    --out-file ./data/qc/20190805/output/task_output/crop_20190805.csv \
#    --map-file ./data/qc/20190805/combined_20190805.csv

#python infer.py --data-dir ./data/qc/20190805/ \
#    --model-path ./results/2019-09-03_03-12-17/saved_models/save_40.pth \
#    --batch-size 32 \
#    --gpu-id 0 \
#    --num-workers 2 \
#    --out-file ./data/qc/20190805/output/task_output/crop_2019-09-03_03-12-17.csv \
#    --map-file ./data/qc/20190805/combined_2019-09-03_03-12-17.csv

python infer.py --data-dir ./data/qc/20190805/ \
    --model-path ./results/2019-09-02_03-44-21/saved_models/save_40.pth \
    --batch-size 32 \
    --gpu-id 0 \
    --num-workers 2 \
    --out-file ./data/qc/20190805/output/task_output/crop_2019-09-02_03-44-21.csv \
    --map-file ./data/qc/20190805/combined_2019-09-02_03-44-21.csv
