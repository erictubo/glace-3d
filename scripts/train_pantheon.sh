#!/usr/bin/env bash

# Find the path to the root of the repo.
SCRIPT_PATH=$(dirname $(realpath -s "$0"))
REPO_PATH=$(realpath -s "${SCRIPT_PATH}/..")


scenes=("pantheon_B") # "pantheon")
data_folder="/home/johndoe/Documents/data/GLACE"


training_exe="${REPO_PATH}/train_ace.py"
testing_exe="${REPO_PATH}/test_ace.py"

datasets_folder="${REPO_PATH}/datasets"
out_dir="${REPO_PATH}/output"
check_dir="$out_dir/checkpoint"

# encoders=("val_separate_w0.8_0.2_pantheon_e1")
encoders=("val_separate_w0.8_0.2_notre_dame_e1")


mkdir -p "$out_dir"
mkdir -p "$check_dir"

# torchrun --standalone --nnodes 1 --nproc-per-node 1 \
#   $training_exe "$data_folder/pantheon" "$out_dir/pantheon_mixed_loss.pt" \
#   --checkpoint_path "$check_dir/pantheon_mixed_loss.pt" \
#   --mode 1 --sparse True --num_head_blocks 2 --training_buffer_size 4000000 --max_iterations 30000

# torchrun --standalone --nnodes 1 --nproc-per-node 1 \
#   $training_exe "$data_folder/pantheon_B" "$out_dir/pantheon_B_mixed_loss.pt" \
#   --checkpoint_path "$check_dir/pantheon_B_mixed_loss.pt" \
#   --mode 1 --num_head_blocks 2 --training_buffer_size 4000000 --max_iterations 30000


for scene in ${scenes[*]}; do


  for encoder in ${encoders[*]}; do
    # python "$datasets_folder/extract_features.py" "$data_folder/$scene" --checkpoint "$datasets_folder/CVPR23_DeitS_Rerank".pth

    # python "$datasets_folder/extract_features.py" "$data_folder/pantheon" --checkpoint "$datasets_folder/CVPR23_DeitS_Rerank".pth

    torchrun --standalone --nnodes 1 --nproc-per-node 1 \
      $training_exe "$data_folder/${scene}" "$out_dir/${scene}_${encoder}.pt" \
      --checkpoint_path "$check_dir/${scene}_${encoder}.pt" \
      --encoder_path "${REPO_PATH}/${encoder}.pt" \
      --mode 0 --num_head_blocks 2 --training_buffer_size 4000000 --max_iterations 30000

    # python $testing_exe "$data_folder/$scene" "$out_dir/$scene.pt" 2>&1 | tee "$out_dir/log_${scene}.txt"

    # python $testing_exe "$data_folder/pantheon_B" "$out_dir/${scene}_${encoder}.pt" 2>&1 | tee "$out_dir/log_${scene}_${encoder}_renders.txt"

    # python $testing_exe "$data_folder/pantheon" "$out_dir/${scene}_${encoder}.pt" 2>&1 | tee "$out_dir/log_${scene}_${encoder}_real.txt"
  done
done

# for scene in ${scenes[*]}; do
#   echo "${scene}: $(cat "${out_dir}/log_${scene}.txt" | tail -2 | head -1)"
# done
