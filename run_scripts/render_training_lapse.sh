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
render_lapse() {
	# Train:
	echo "Starting Rendering..."
	python render_training_lapse.py \
	-i logs/rf/voxelArt_try_${1}_3dcnn_32_shdeg0_naive_False_qcTrue_z1_iter_300/training_logs/rendered_output/ \
	-o logs/rf/voxelArt_try_${1}_3dcnn_32_shdeg0_naive_False_qcTrue_z1_iter_300/training_logs/rendered_output/lapse_voxelized.mp4
}

# STARTING RUN:
render_lapse $scene_in
