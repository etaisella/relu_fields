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
	-o logs/rf/${1}_${7}_sf${2}_iter${3}_lr${4}_stage${5}_samples${6}_colors${8}_qc_${9}_sa_${10}_${11}_${12}_${13}_sl${15}_sem${14}_va_on_vox_highacc/ \
	--grid_dims=${7} ${7} ${7} \
	--learning_rate=$4 \
	--num_stages=$5 \
	--train_num_samples_per_ray=$6 \
	--num_iterations_per_stage=$3 \
	--num_colors=$8 \
	--sa_init_weight=${10} \
	--sa_start_iter=${11} \
	--sa_gamma=${12} \
	--sa_interval=${13} \
	--semantic_weight=${14} \
	--sl_weight=${15} \
	--accumulation_iters=10 \
	--lr_decay_gamma_per_stage=1.0

	# Render Video
	echo "Starting Rendering Voxelized..."
	python render_sh_based_voxel_grid.py -d data/${1}/ \
	-i logs/rf/${1}_${7}_sf${2}_iter${3}_lr${4}_stage${5}_samples${6}_colors${8}_qc_${9}_sa_${10}_${11}_${12}_${13}_sl${15}_sem${14}_va_on_vox_highacc/saved_models/model_best.pth \
	-o output_renders/${1}_${7}_sf${2}_iter${3}_lr${4}_stage${5}_samples${6}_colors${8}_qc_${9}_sa_${10}_${11}_${12}_${13}_sl${15}_sem${14}_va_on_vox_highacc/ 
}

# STARTING RUN:
scene=lego
scale_factor=2.0
num_iterations_per_stage=800
learning_rate=0.09
num_stages=3
train_num_samples_per_ray=256
gird_dim=32
num_colors=5
quantize_colors=True
sa_init_weight=0.0
sa_start_iter=-1
sa_gamma=1.0
sa_interval=100
semantic_weight=0.0
sl_weight=60.0

train_and_render $scene $scale_factor $num_iterations_per_stage $learning_rate $num_stages \
$train_num_samples_per_ray $gird_dim $num_colors $quantize_colors $sa_init_weight $sa_start_iter \
$sa_gamma $sa_interval $semantic_weight $sl_weight

scene=hotdog
scale_factor=2.0
num_iterations_per_stage=800
learning_rate=0.09
num_stages=3
train_num_samples_per_ray=256
gird_dim=32
num_colors=5
quantize_colors=True
sa_init_weight=0.0
sa_start_iter=-1
sa_gamma=1.0
sa_interval=100
semantic_weight=0.0
sl_weight=60.0

train_and_render $scene $scale_factor $num_iterations_per_stage $learning_rate $num_stages \
$train_num_samples_per_ray $gird_dim $num_colors $quantize_colors $sa_init_weight $sa_start_iter \
$sa_gamma $sa_interval $semantic_weight $sl_weight


scene=drums
scale_factor=2.0
num_iterations_per_stage=800
learning_rate=0.09
num_stages=3
train_num_samples_per_ray=256
gird_dim=32
num_colors=5
quantize_colors=True
sa_init_weight=0.0
sa_start_iter=-1
sa_gamma=1.0
sa_interval=100
semantic_weight=0.0
sl_weight=60.0

train_and_render $scene $scale_factor $num_iterations_per_stage $learning_rate $num_stages \
$train_num_samples_per_ray $gird_dim $num_colors $quantize_colors $sa_init_weight $sa_start_iter \
$sa_gamma $sa_interval $semantic_weight $sl_weight