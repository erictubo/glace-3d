#!/usr/bin/env bash

# Find the path to the root of the repo.
SCRIPT_PATH=$(dirname $(realpath -s "$0"))
REPO_PATH=$(realpath -s "${SCRIPT_PATH}/..")

data_folder="/home/johndoe/Documents/data/GLACE"

testing_exe="${REPO_PATH}/test_ace_coords.py"

datasets_folder="${REPO_PATH}/datasets"
out_dir="${REPO_PATH}/output"

real_scene="pantheon"
fake_scene="pantheon_B"

real_encoder="ace_encoder_pretrained"
fake_encoders=("val_separate_w0.8_0.2_pantheon_e1" "val_separate_w0.8_0.2_notre_dame_e1")


# Testing regressor trained on real_scene:

# Real images on pre-trained encoder
python $testing_exe "$data_folder/$real_scene" "$out_dir/$real_scene.pt" \
  --encoder_path "${REPO_PATH}/${real_encoder}.pt" --sparse True


Fake images on pre-trained  encoder
python $testing_exe "$data_folder/$fake_scene" "$out_dir/$real_scene.pt" \
  --encoder_path "${REPO_PATH}/${real_encoder}.pt"


for fake_encoder in ${fake_encoders[*]}; do

  # Fake images on fine-trained encoder
  python $testing_exe "$data_folder/$fake_scene" "$out_dir/$real_scene.pt" \
    --encoder_path "${REPO_PATH}/${fake_encoder}.pt"

done

