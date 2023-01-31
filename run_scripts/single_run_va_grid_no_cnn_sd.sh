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
	-o logs/rf/${1}_qc_${2}_sdl_strt_${3}_w_sa_${4}_sa_strt_${5}_q_${6}_vas_${8}_w_reg_07_va/ \
	--quantize_colors=${2} \
	--sa_init_weight=${4} \
	--sa_start_iter=${5} \
	--start_sdl_iter=${3} \
	--prompt="$7" \
	--va_structure_mode=${8} \
	--sa_percentile=${6}


	# Render Video
	echo "Starting Rendering Final..."
	python render_sh_based_voxel_grid.py -d data/${1}/ \
	-i logs/rf/${1}_qc_${2}_sdl_strt_${3}_w_sa_${4}_sa_strt_${5}_q_${6}_vas_${8}_w_reg_07_va/saved_models/model_final.pth \
	-o output_renders/${1}_qc_${2}_sdl_strt_${3}_w_sa_${4}_sa_strt_${5}_q_${6}_vas_${8}_w_reg_07_va/final/ \
	--quantize_colors=${2} 

	# Render Video
	echo "Starting Rendering Pre - SA..."
	python render_sh_based_voxel_grid.py -d data/${1}/ \
	-i logs/rf/${1}_qc_${2}_sdl_strt_${3}_w_sa_${4}_sa_strt_${5}_q_${6}_vas_${8}_w_reg_07_va/saved_models/model_stage_3_iter_2400.pth \
	-o output_renders/${1}_qc_${2}_sdl_strt_${3}_w_sa_${4}_sa_strt_${5}_q_${6}_vas_${8}_w_reg_07_va/pre_sa/ \
	--quantize_colors=${2}
}

# STARTING RUN:

scene=robot
quantize_colors=False
start_sdl_iter=2401
sa_init_weight=0.0
sa_start_iter=-1
sa_percentile=0.1
va_structure_mode=True
prompt="a render of a happy rusty robot with an lcd screen for a face in VoxelArt style"

train_and_render $scene $quantize_colors $start_sdl_iter $sa_init_weight $sa_start_iter $sa_percentile "$prompt" $va_structure_mode