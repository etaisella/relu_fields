#!/bin/bash
echo "Starting Run!"

# Reading arguments:
gpu_num=0
while getopts g:d: flag
do
    case "${flag}" in
        g) gpu_num=${OPTARG};;
		d) scene_in=${OPTARG};;
    esac
done

# Setting GPU:
echo "Running on GPU: $gpu_num";
export CUDA_VISIBLE_DEVICES=$gpu_num

# Rendering function template:
train_and_render() {
	# Train:
	echo "Starting Training..."
	python learn_voxelArt_grid_zero_one.py -d data/${1}/ \
	-o logs/rf/${1}_${7}_sf${2}_iter${3}_lr${4}_stage${5}_samples${6}_shdeg${8}_clusters${9}_zero_one/ \
	--grid_dims=${7} ${7} ${7} \
	--learning_rate=$4 \
	--num_stages=$5 \
	--activate_zero_one_iter=1800
}

# STARTING RUN:

scene=$scene_in
scale_factor=2.0
num_iterations_per_stage=800
learning_rate=0.03
num_stages=3
train_num_samples_per_ray=512
gird_dim=32
sh_degree=0
clusters=0

train_and_render $scene $scale_factor $num_iterations_per_stage $learning_rate $num_stages \
$train_num_samples_per_ray $gird_dim $sh_degree $clusters