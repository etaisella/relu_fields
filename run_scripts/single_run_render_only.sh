#!/bin/bash
echo "Starting Run!"

while getopts g:d:r:n: flag
do
    case "${flag}" in
        g) gpu_num=${OPTARG};;
		d) scene_in=${OPTARG};;
		r) new_grid_dim=${OPTARG};;
    esac
done

# Setting GPU:
echo "Running on GPU: $gpu_num";
export CUDA_VISIBLE_DEVICES=$gpu_num,
echo "Scene: $scene_in" 

# Rendering function template:
train_and_render() {
	# Render Video
	echo "Starting Rendering..."
	python render_sh_based_voxel_grid.py \
	-i logs/rf/${1}_${7}_sf${2}_iter${3}_lr${4}_stage${5}_samples${6}_shdeg${8}_clusters0/saved_models/model_final.pth \
	-o output_renders/${1}_${7}_sf${2}_iter${3}_lr${4}_stage${5}_samples${6}_shdeg${8}/ \
	--voxelized=${9}
}

# STARTING RUN:

scene=$scene_in
scale_factor=2.0
num_iterations_per_stage=500
learning_rate=0.03
num_stages=3
train_num_samples_per_ray=512
gird_dim=32
sh_degree=2
voxelized=True

train_and_render $scene $scale_factor $num_iterations_per_stage $learning_rate $num_stages \
$train_num_samples_per_ray $gird_dim $sh_degree $voxelized