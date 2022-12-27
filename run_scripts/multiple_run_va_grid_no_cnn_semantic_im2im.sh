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
	-o logs/rf/${1}_${7}_sf${2}_iter${3}_lr${4}_stage${5}_samples${6}_colors${8}_qc_${9}_sem_${10}_im2im_on_vox/ \
	--grid_dims=${7} ${7} ${7} \
	--learning_rate=$4 \
	--num_stages=$5 \
	--train_num_samples_per_ray=$6 \
	--num_iterations_per_stage=$3 \
	--num_colors=$8 \
	--semantic_weight=${10} \
	--start_semantic_iter=${11} \
	--clip_prompt=${12} \
	--accumulation_iters=32

	# Render Video
	echo "Starting Rendering Voxelized..."
	python render_sh_based_voxel_grid.py -d data/${1}/ \
	-i logs/rf/${1}_${7}_sf${2}_iter${3}_lr${4}_stage${5}_samples${6}_colors${8}_qc_${9}_sem_${10}_im2im_on_vox/saved_models/model_final.pth \
	-o output_renders/${1}_${7}_sf${2}_iter${3}_lr${4}_stage${5}_samples${6}_colors${8}_qc_${9}_sem_${10}_im2im_on_vox/ 
}

# STARTING RUN:

scene=dog2
scale_factor=2.0
num_iterations_per_stage=800
learning_rate=0.15
num_stages=3
train_num_samples_per_ray=256
gird_dim=32
num_colors=5
quantize_colors=True
semantic_weight=1.0
start_semantic_iter=2200
clip_prompt="none"

train_and_render $scene $scale_factor $num_iterations_per_stage $learning_rate $num_stages \
$train_num_samples_per_ray $gird_dim $num_colors $quantize_colors $semantic_weight $start_semantic_iter \
$clip_prompt

scene=lego
scale_factor=2.0
num_iterations_per_stage=800
learning_rate=0.15
num_stages=3
train_num_samples_per_ray=256
gird_dim=32
num_colors=5
quantize_colors=True
semantic_weight=1.0
start_semantic_iter=2200
clip_prompt="none"

train_and_render $scene $scale_factor $num_iterations_per_stage $learning_rate $num_stages \
$train_num_samples_per_ray $gird_dim $num_colors $quantize_colors $semantic_weight $start_semantic_iter \
$clip_prompt

scene=hotdog
scale_factor=2.0
num_iterations_per_stage=800
learning_rate=0.15
num_stages=3
train_num_samples_per_ray=256
gird_dim=32
num_colors=5
quantize_colors=True
semantic_weight=1.0
start_semantic_iter=2200
clip_prompt="none"

train_and_render $scene $scale_factor $num_iterations_per_stage $learning_rate $num_stages \
$train_num_samples_per_ray $gird_dim $num_colors $quantize_colors $semantic_weight $start_semantic_iter \
$clip_prompt