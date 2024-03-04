#!/bin/bash
cd $(dirname $0)
echo $(pwd)

export PYTHONPATH="$(pwd)/../../../../":$PYTHONPATH
experitment_time=$(date "+%Y-%m-%d-%H-%M-%S")
save_path="/egr/research-dselab/liujin33/graph_scaling_law/models/mol/pruning/results"

log_dir="$save_path/GNN-mol-$experitment_time"
mkdir -p $log_dir
log_file="$log_dir/log.txt"
checkpoint_dir="$log_dir/checkpoints"
echo "Outputs redirected to $log_file"
mkdir -p $checkpoint_dir
python main_pyg.py > $log_file 2>&1
wait
