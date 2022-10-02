#!/bin/bash
echo "Starting Run!"

# Reading arguments:
gpu_num=0
while getopts g: flag
do
    case "${flag}" in
        g) gpu_num=${OPTARG};;
    esac
done

# Setting GPU:
echo "Running on GPU: $gpu_num";
export CUDA_VISIBLE_DEVICES=$gpu_num

# Rendering function template:
train_and_render() {	
	# Render Video
	echo "Starting Rendering..."
	python render_sh_based_voxel_grid.py \
	-i logs/rf/${1}_32_sf${2}_iter${3}_lr${4}_stage${5}_samples${6}/saved_models/model_final.pth \
	-o output_renders/${1}_32_sf${2}_iter${3}_lr${4}_stage${5}_samples${6}/ \
	--num_frames 90
}

# STARTING RUN:

scene=lego
scale_factor=1.5
num_iterations_per_stage=500
learning_rate=0.03
num_stages=3
train_num_samples_per_ray=256

train_and_render $scene $scale_factor $num_iterations_per_stage $learning_rate $num_stages $train_num_samples_per_ray