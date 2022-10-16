from typing import Sequence, Optional
import torch

import numpy as np

from thre3d_atom.modules.volumetric_model import VolumetricModel
from thre3d_atom.utils.constants import EXTRA_ACCUMULATED_WEIGHTS, NUM_COLOUR_CHANNELS
from thre3d_atom.utils.imaging_utils import (
    CameraPose,
    CameraIntrinsics,
    scale_camera_intrinsics,
    postprocess_depth_map,
    to8b,
)
from thre3d_atom.utils.logging import log

def render_camera_path_for_volumetric_model_3_coeff_modes_gray(
    vol_mod: VolumetricModel,
    camera_path: Sequence[CameraPose],
    camera_intrinsics: CameraIntrinsics,
    render_scale_factor: Optional[float] = None,
    overridden_num_samples_per_ray: Optional[int] = None,
) -> np.array:
    if render_scale_factor is not None:
        # Render downsampled images for speed if requested
        camera_intrinsics = scale_camera_intrinsics(
            camera_intrinsics, render_scale_factor
        )

    overridden_config_dict = {}
    if overridden_num_samples_per_ray is not None:
        overridden_config_dict.update(
            {"num_samples_per_ray": overridden_num_samples_per_ray}
        )

    rendered_frames = []
    total_frames = len(camera_path) + 1

    # Make zero coeffs "gray":
    vol_mod.thre3d_repr.features.data[:,:,:,0] = 0.5 / 0.28209479177387814
    vol_mod.thre3d_repr.features.data[:,:,:,9] = 0.5 / 0.28209479177387814
    vol_mod.thre3d_repr.features.data[:,:,:,18] = 0.5 / 0.28209479177387814

    original_features = torch.zeros_like(vol_mod.thre3d_repr.features.data)
    original_features[:] = vol_mod.thre3d_repr.features.data[:]
    for frame_num, render_pose in enumerate(camera_path):
        log.info(f"rendering frame number: ({frame_num + 1}/{total_frames})")
        rendered_output_all_coeffs = vol_mod.render(
            render_pose,
            camera_intrinsics,
            gpu_render=False,
            verbose=True,
            **overridden_config_dict,
        )
        colour_frame_all_coeffs = rendered_output_all_coeffs.colour.numpy()

        # zero out 2nd order coeffs
        vol_mod.thre3d_repr.features.data[:,:,:,4:9] = 0
        vol_mod.thre3d_repr.features.data[:,:,:,13:18] = 0
        vol_mod.thre3d_repr.features.data[:,:,:,21:] = 0

        rendered_output_1st_order_coeffs = vol_mod.render(
            render_pose,
            camera_intrinsics,
            gpu_render=False,
            verbose=True,
            **overridden_config_dict,
        )
        colour_frame_1st_order_coeffs = rendered_output_1st_order_coeffs.colour.numpy()

         # zero out 2nd order coeffs
        vol_mod.thre3d_repr.features.data[:,:,:,1:9] = 0
        vol_mod.thre3d_repr.features.data[:,:,:,10:18] = 0
        vol_mod.thre3d_repr.features.data[:,:,:,19:] = 0

        rendered_output_diffuse = vol_mod.render(
            render_pose,
            camera_intrinsics,
            gpu_render=False,
            verbose=True,
            **overridden_config_dict,
        )
        colour_frame_diffuse = rendered_output_diffuse.colour.numpy()

        colour_frame_all_coeffs = to8b(colour_frame_all_coeffs)
        colour_frame_1st_order_coeffs = to8b(colour_frame_1st_order_coeffs)
        colour_frame_diffuse = to8b(colour_frame_diffuse)

        vol_mod.thre3d_repr.features.data[:] = original_features[:]

        ## create grand concatenated frame horizontally
        frame = np.concatenate([colour_frame_all_coeffs, colour_frame_1st_order_coeffs, colour_frame_diffuse], axis=1)
        rendered_frames.append(frame)

    return np.stack(rendered_frames)

def render_camera_path_for_volumetric_model_3_coeff_modes(
    vol_mod: VolumetricModel,
    camera_path: Sequence[CameraPose],
    camera_intrinsics: CameraIntrinsics,
    render_scale_factor: Optional[float] = None,
    overridden_num_samples_per_ray: Optional[int] = None,
) -> np.array:
    if render_scale_factor is not None:
        # Render downsampled images for speed if requested
        camera_intrinsics = scale_camera_intrinsics(
            camera_intrinsics, render_scale_factor
        )

    overridden_config_dict = {}
    if overridden_num_samples_per_ray is not None:
        overridden_config_dict.update(
            {"num_samples_per_ray": overridden_num_samples_per_ray}
        )

    rendered_frames = []
    total_frames = len(camera_path) + 1
    original_features = torch.zeros_like(vol_mod.thre3d_repr.features.data)
    original_features[:] = vol_mod.thre3d_repr.features.data[:]
    for frame_num, render_pose in enumerate(camera_path):
        log.info(f"rendering frame number: ({frame_num + 1}/{total_frames})")
        rendered_output_all_coeffs = vol_mod.render(
            render_pose,
            camera_intrinsics,
            gpu_render=False,
            verbose=True,
            **overridden_config_dict,
        )
        colour_frame_all_coeffs = rendered_output_all_coeffs.colour.numpy()

        # zero out 2nd order coeffs
        vol_mod.thre3d_repr.features.data[:,:,:,4:9] = 0
        vol_mod.thre3d_repr.features.data[:,:,:,13:18] = 0
        vol_mod.thre3d_repr.features.data[:,:,:,21:] = 0

        rendered_output_1st_order_coeffs = vol_mod.render(
            render_pose,
            camera_intrinsics,
            gpu_render=False,
            verbose=True,
            **overridden_config_dict,
        )
        colour_frame_1st_order_coeffs = rendered_output_1st_order_coeffs.colour.numpy()

         # zero out 2nd order coeffs
        vol_mod.thre3d_repr.features.data[:,:,:,1:9] = 0
        vol_mod.thre3d_repr.features.data[:,:,:,10:18] = 0
        vol_mod.thre3d_repr.features.data[:,:,:,19:] = 0

        rendered_output_diffuse = vol_mod.render(
            render_pose,
            camera_intrinsics,
            gpu_render=False,
            verbose=True,
            **overridden_config_dict,
        )
        colour_frame_diffuse = rendered_output_diffuse.colour.numpy()

        colour_frame_all_coeffs = to8b(colour_frame_all_coeffs)
        colour_frame_1st_order_coeffs = to8b(colour_frame_1st_order_coeffs)
        colour_frame_diffuse = to8b(colour_frame_diffuse)

        vol_mod.thre3d_repr.features.data[:] = original_features[:]

        ## create grand concatenated frame horizontally
        frame = np.concatenate([colour_frame_all_coeffs, colour_frame_1st_order_coeffs, colour_frame_diffuse], axis=1)
        rendered_frames.append(frame)

    return np.stack(rendered_frames)
