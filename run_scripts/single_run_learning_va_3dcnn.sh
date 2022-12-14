#!/bin/bash
echo "Starting Run!"

# Reading arguments:
gpu_num=0
new_grid_dim=32
naive_mode=False

while getopts g:d:r:n: flag
do
    case "${flag}" in
        g) gpu_num=${OPTARG};;
		d) scene_in=${OPTARG};;
		r) new_grid_dim=${OPTARG};;
		n) naive_mode=${OPTARG};;
    esac
done

# Setting GPU:
echo "Running on GPU: $gpu_num";
export CUDA_VISIBLE_DEVICES=$gpu_num,
echo "Scene: $scene_in" 
echo "Naive Mode: $naive_mode" 

# Rendering function template:
train_and_render() {
	# training and rendering
	echo "Starting Rendering..."
	python learn_voxelArt_grid_3dcnn.py \
	-i logs/rf/${1}_${7}_sf${2}_iter${3}_lr${4}_stage${5}_samples${6}_shdeg${8}_clusters${9}/saved_models/model_final.pth \
	-r output_renders/${1}_${7}_sf${2}_iter${3}_lr${4}_stage${5}_samples${6}_shdeg${8}_clusters${9}/ \
	-d data/${1}/ \
	-o logs/rf/voxelArt_try_${1}_3dcnn_${10}_shdeg${8}_naive_${11}_qc${12}_z1_iter_300/ \
	--new_grid_dims=${10} ${10} ${10} \
	--zero_one_density=False \
	--quantize_colors=${12} \
	--zero_one_density_iter=-1
}

# STARTING RUN:

scene=dog2
scale_factor=2.0
num_iterations_per_stage=500
learning_rate=0.03
num_stages=4
train_num_samples_per_ray=512
gird_dim=128
sh_degree=0
clusters=0
quantize_colors=True

train_and_render $scene $scale_factor $num_iterations_per_stage $learning_rate $num_stages \
$train_num_samples_per_ray $gird_dim $sh_degree $clusters $new_grid_dim $naive_mode $quantize_colors