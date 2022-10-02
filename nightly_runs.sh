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

# STARTING RUNS:
####### RUN 1 ####### 

train_and_render() {
	# Train:
	python train_sh_based_voxel_grid_with_posed_images.py -d data/lego/ \
	-o logs/rf/lego_64_sf${1}_iter${2}_lr${3}_stage${4}_samples${5}/ \
	--grid_dims=64 64 64 \
	--scale_factor=$1 \
	--num_iterations_per_stage=$2 \
	--learning_rate=$3 \
	--num_stages=$4 \
	--train_num_samples_per_ray=$5

	# Render Video
	python render_sh_based_voxel_grid.py \
	-i logs/rf/lego_64_sf${1}_iter${2}_lr${3}_stage${4}_samples${5}/saved_models/model_final.pth \
	-o output_renders/lego_64_sf${1}_iter${2}_lr${3}_stage${4}_samples${5}/ 
}

##### RUN 1 #####

scale_factor=2.0
num_iterations_per_stage=800
learning_rate=0.03
num_stages=3
train_num_samples_per_ray=256

train_and_render $scale_factor $num_iterations_per_stage $learning_rate $num_stages $train_num_samples_per_ray

##### RUN 2 #####

scale_factor=2.0
num_iterations_per_stage=800
learning_rate=0.02
num_stages=3
train_num_samples_per_ray=256

train_and_render $scale_factor $num_iterations_per_stage $learning_rate $num_stages $train_num_samples_per_ray

##### RUN 3 #####

scale_factor=2.0
num_iterations_per_stage=800
learning_rate=0.04
num_stages=3
train_num_samples_per_ray=256

train_and_render $scale_factor $num_iterations_per_stage $learning_rate $num_stages $train_num_samples_per_ray

##### RUN 4 #####

scale_factor=2.0
num_iterations_per_stage=2000
learning_rate=0.02
num_stages=3
train_num_samples_per_ray=256

train_and_render $scale_factor $num_iterations_per_stage $learning_rate $num_stages $train_num_samples_per_ray

##### RUN 5 #####

scale_factor=2.0
num_iterations_per_stage=2000
learning_rate=0.03
num_stages=3
train_num_samples_per_ray=256

train_and_render $scale_factor $num_iterations_per_stage $learning_rate $num_stages $train_num_samples_per_ray

##### RUN 6 #####

scale_factor=2.0
num_iterations_per_stage=2000
learning_rate=0.05
num_stages=3
train_num_samples_per_ray=256

train_and_render $scale_factor $num_iterations_per_stage $learning_rate $num_stages $train_num_samples_per_ray

##### RUN 7 #####

scale_factor=1.5
num_iterations_per_stage=2000
learning_rate=0.05
num_stages=4
train_num_samples_per_ray=256

train_and_render $scale_factor $num_iterations_per_stage $learning_rate $num_stages $train_num_samples_per_ray

