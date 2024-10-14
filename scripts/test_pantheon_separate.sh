#!/usr/bin/env bash

# Find the path to the root of the repo.
SCRIPT_PATH=$(dirname $(realpath -s "$0"))
REPO_PATH=$(realpath -s "${SCRIPT_PATH}/..")

training_exe="${REPO_PATH}/train_ace.py"
testing_exe="${REPO_PATH}/test_ace.py"

datasets_folder="${REPO_PATH}/datasets"
out_dir="${REPO_PATH}/output"
check_dir="$out_dir/checkpoint"


scenes=("pantheon")
data_folder="/home/johndoe/Documents/data/GLACE"

encoders=("ace_encoder_pretrained" "fine-tuned_encoder_separate")


mkdir -p "$out_dir"
mkdir -p "$check_dir"

for scene in ${scenes[*]}; do
  for encoder in ${encoders[*]}; do

    # python $testing_exe "$data_folder/$scene" "$out_dir/$scene.pt" 2>&1 | tee "$out_dir/log_${scene}.txt"

    # Evaluate separate encoder with fake features on real training
    python $testing_exe "$data_folder/pantheon_B" "$out_dir/${scene}.pt" --encoder_path "${REPO_PATH}/${encoder}.pt" 2>&1 | tee "$out_dir/log_${scene}_${encoder}.txt"

  done
done

# for scene in ${scenes[*]}; do
#   echo "${scene}: $(cat "${out_dir}/log_${scene}.txt" | tail -2 | head -1)"
# done
