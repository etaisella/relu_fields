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
	# Render Video
	echo "Starting Rendering..."
	python render_sh_based_voxel_grid.py -d data/${1}/ \
	-i logs/rf/${1}_${7}_sf${2}_iter${3}_lr${4}_stage${5}_samples${6}_colors${8}_qc_${9}_va_nonrand/saved_models/model_best.pth \
	-o output_renders/${1}_${7}_sf${2}_iter${3}_lr${4}_stage${5}_samples${6}_colors${8}_qc_${9}_va_nonrand_voxelized/ \
	--num_colors=$8
}

# STARTING RUN:

scene=dog2
scale_factor=2.0
num_iterations_per_stage=800
learning_rate=0.03
num_stages=3
train_num_samples_per_ray=256
gird_dim=32
num_colors=5
quantize_colors=True

train_and_render $scene $scale_factor $num_iterations_per_stage $learning_rate $num_stages \
$train_num_samples_per_ray $gird_dim $num_colors $quantize_colors