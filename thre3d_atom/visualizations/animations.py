from typing import Sequence, Optional
import torch

import numpy as np

from torch import Tensor
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

def process_rendered_id_output_for_feedback_log(
    rendered_output: Tensor,
    max_grid_dim: int,
) -> np.array:
    rendered_output_np = rendered_output.cpu().numpy()

    # force background to be whiter than the voxels:
    background_value = 1.05 * max_grid_dim 
    rendered_output_np[rendered_output_np < 0] = background_value

    # normalize to 0-1:
    rendered_output_np = rendered_output_np / background_value
    
    # move to 8b
    result = to8b(rendered_output_np)

    return result

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
        rendered_output_all_coeffs, _, _ = vol_mod.render(
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

        rendered_output_1st_order_coeffs, _, _ = vol_mod.render(
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

        rendered_output_diffuse, _, _ = vol_mod.render(
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
        rendered_output_all_coeffs, _, _ = vol_mod.render(
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

        rendered_output_1st_order_coeffs, _, _ = vol_mod.render(
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

        rendered_output_diffuse, _, _ = vol_mod.render(
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

def render_camera_path_for_volumetric_model(
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
    for frame_num, render_pose in enumerate(camera_path):
        log.info(f"rendering frame number: ({frame_num + 1}/{total_frames})")
        specular_rendered_output, specular_rendered_output_va, id_img = vol_mod.render(
            render_pose,
            camera_intrinsics,
            gpu_render=False,
            verbose=True,
            **overridden_config_dict,
        )

        grid_x, grid_y, grid_z, _ = vol_mod.thre3d_repr._densities.shape
        max_grid_dim = max([grid_x, grid_y, grid_z])

        id_feedback_image = process_rendered_id_output_for_feedback_log(
            id_img,
            max_grid_dim
        )

        specular_feedback_image = to8b(specular_rendered_output.colour.numpy())

        va_feedback_image = to8b(specular_rendered_output_va.colour.numpy())

        ## create grand concatenated frame horizontally
        frame = np.concatenate([specular_feedback_image, va_feedback_image, \
            id_feedback_image], axis=1)
        rendered_frames.append(frame)

    return np.stack(rendered_frames)