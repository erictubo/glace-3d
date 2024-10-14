#!/usr/bin/env bash

# Find the path to the root of the repo.
SCRIPT_PATH=$(dirname $(realpath -s "$0"))
REPO_PATH=$(realpath -s "${SCRIPT_PATH}/..")


scenes=("notre_dame_closer" "notre_dame")
data_folder="/home/johndoe/Documents/data/GLACE"


training_exe="${REPO_PATH}/train_ace.py"
testing_exe="${REPO_PATH}/test_ace.py"

datasets_folder="${REPO_PATH}/datasets"
out_dir="${REPO_PATH}/output"
check_dir="$out_dir/checkpoint"


mkdir -p "$out_dir"
mkdir -p "$check_dir"

for scene in ${scenes[*]}; do
  python "$datasets_folder/extract_features.py" "$data_folder/$scene" --checkpoint "$datasets_folder/CVPR23_DeitS_Rerank".pth

  torchrun --standalone --nnodes 1 --nproc-per-node 1 \
    $training_exe "$data_folder/$scene" "$out_dir/$scene.pt" \
    --checkpoint_path "$check_dir/$scene.pt" \
    --mode 0 --num_head_blocks 2 --training_buffer_size 4000000 --max_iterations 30000

  python $testing_exe "$data_folder/$scene" "$out_dir/$scene.pt" 2>&1 | tee "$out_dir/log_${scene}.txt"
done

for scene in ${scenes[*]}; do
  echo "${scene}: $(cat "${out_dir}/log_${scene}.txt" | tail -2 | head -1)"
done
