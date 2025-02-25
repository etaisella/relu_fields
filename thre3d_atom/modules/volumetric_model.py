import copy
import dataclasses
from kmeans_pytorch import kmeans
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Tuple

import torch
from torch.nn import Module
from tqdm import tqdm

from thre3d_atom.rendering.volumetric.render_interface import RenderOut, Rays
from thre3d_atom.rendering.volumetric.utils.misc import (
    cast_rays,
    flatten_rays,
    reshape_rendered_output,
    collate_rendered_output,
)
from thre3d_atom.thre3d_reprs.constants import (
    RENDER_CONFIG,
    RENDER_PROCEDURE,
    STATE_DICT,
    CONFIG_DICT,
    THRE3D_REPR,
    RENDER_CONFIG_TYPE,
)
from thre3d_atom.thre3d_reprs.voxels import create_voxel_grid_from_saved_info_dict
from thre3d_atom.thre3d_reprs.voxelArtGrid_3dcnn import VoxelArtGrid_3DCNN
from thre3d_atom.utils.constants import NUM_COLOUR_CHANNELS
from thre3d_atom.thre3d_reprs.renderers import RenderProcedure, RenderConfig
from thre3d_atom.utils.constants import EXTRA_INFO
from thre3d_atom.utils.imaging_utils import CameraIntrinsics, CameraPose


class VolumetricModel:
    def __init__(
        self,
        thre3d_repr: Module,
        render_procedure: RenderProcedure,
        render_config: RenderConfig,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ) -> None:
        # state of the object:
        self._thre3d_repr = thre3d_repr.to(device)
        self._render_procedure = render_procedure
        self._render_config = render_config
        self._device = device

    @property
    def thre3d_repr(self) -> Module:
        return self._thre3d_repr

    @thre3d_repr.setter
    def thre3d_repr(self, thre3d_repr: Module) -> None:
        self._thre3d_repr = thre3d_repr

    @property
    def render_procedure(self) -> RenderProcedure:
        return self._render_procedure

    @property
    def render_config(self) -> RenderConfig:
        return self._render_config

    @property
    def device(self) -> torch.device:
        return self._device

    @staticmethod
    def _update_render_config(
        render_config: RenderConfig, update_dict: Dict[str, Any]
    ) -> RenderConfig:
        # create a new copy for keeping the original safe
        updated_render_config = copy.deepcopy(render_config)

        # update the render configuration with the overridden kwargs:
        for field, value in update_dict.items():
            if not hasattr(updated_render_config, field):
                raise ValueError(
                    f"Unknown render configuration field {field} requested for overriding :("
                )
            setattr(updated_render_config, field, value)

        return updated_render_config

    def get_save_info(
        self, extra_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        save_info = {
            THRE3D_REPR: {
                STATE_DICT: self._thre3d_repr.state_dict(),
                CONFIG_DICT: self._thre3d_repr.get_save_config_dict(),
            },
            RENDER_PROCEDURE: self._render_procedure,
            RENDER_CONFIG_TYPE: type(self._render_config),
            RENDER_CONFIG: dataclasses.asdict(self._render_config),
        }
        if extra_info is not None:
            save_info.update({EXTRA_INFO: extra_info})
        return save_info

    def render_rays(
        self, rays: Rays, parallel_points_chunk_size: Optional[int] = None, **kwargs
    ) -> RenderOut:
        """
        renders the rays for the underlying thre3d_repr using the render
        procedure and render config ``differentiably''
        Args:
            rays: The rays to be rendered :)
            parallel_points_chunk_size: used for point-based parallelism
            **kwargs: any configuration parameters if required to be overridden
        Returns:
        """
        render_config = self._update_render_config(self._render_config, kwargs)
        return self._render_procedure(
            self._thre3d_repr, rays, render_config, parallel_points_chunk_size
        )

    def render(
        self,
        camera_pose: CameraPose,
        camera_intrinsics: CameraIntrinsics,
        parallel_rays_chunk_size: Optional[int] = 32768,
        parallel_points_chunk_size: Optional[int] = None,
        gpu_render: bool = True,
        verbose: bool = False,
        **kwargs,
    ) -> Tuple[RenderOut, RenderOut]:
        """
        renders the underlying thre3d_repr for the given camera parameters. Please
        note that this method works in pytorch's no_grad mode.
        Args:
            camera_pose: pose of the render camera
            camera_intrinsics: camera intrinsics for the render Camera
            parallel_rays_chunk_size: chunk size used for parallel ray-rendering
            parallel_points_chunk_size: chunk size used for points-based parallel processing
            gpu_render: whether to keep the rendered output on the GPU or bring to cpu. Consider turning this False
            for High resolution renders. The performance decreases quite a lot though.
            verbose: whether to show progress bar for the render.
            **kwargs: any overridden render configuration
        Returns: rendered_output :)
        """
        progress_bar = tqdm if verbose else lambda x: x

        # cast the rays for the given camera pose:
        casted_rays = cast_rays(
            camera_intrinsics=camera_intrinsics, pose=camera_pose, device=self._device
        )
        flat_rays = flatten_rays(casted_rays)

        # note that we are not using `batchify` here because of the gpu_cpu handling code
        # TODO: Improve the batchify utility to account for this following use case :)
        rendered_chunks = []
        rendered_chunks_va = []
        id_map_chunks = []
        parallel_rays_chunk_size = (
            len(flat_rays)
            if parallel_rays_chunk_size is None
            else parallel_rays_chunk_size
        )
        with torch.no_grad():
            for chunk_index in progress_bar(
                range(0, len(flat_rays), parallel_rays_chunk_size)
            ):
                rendered_chunk, rendered_chunk_va, id_map_chunk, min_distance = self.render_rays(
                    flat_rays[chunk_index : chunk_index + parallel_rays_chunk_size],
                    parallel_points_chunk_size,
                    **kwargs,
                )
                if not gpu_render:
                    rendered_chunk = rendered_chunk.to(torch.device("cpu"))
                    rendered_chunk_va = rendered_chunk_va.to(torch.device("cpu"))
                rendered_chunks.append(rendered_chunk)
                rendered_chunks_va.append(rendered_chunk_va)
                id_map_chunks.append(id_map_chunk)

        id_map_output = torch.cat(id_map_chunks, dim=0)
        id_map_output = torch.reshape(id_map_output, \
            (camera_intrinsics.height, camera_intrinsics.width, -1))

        rendered_output = reshape_rendered_output(
            collate_rendered_output(rendered_chunks),
            camera_intrinsics=camera_intrinsics,
        )
        rendered_output_va = reshape_rendered_output(
            collate_rendered_output(rendered_chunks_va),
            camera_intrinsics=camera_intrinsics,
        )

        return rendered_output, rendered_output_va, id_map_output, min_distance


def create_volumetric_model_from_saved_model(
    model_path: Path,
    thre3d_repr_creator: Callable[[Dict[str, Any]], Module],
    device: torch.device = torch.device("cpu"),
    num_clusters: int = 0,
    quantize_colors: bool=True,
) -> Tuple[VolumetricModel, Dict[str, Any]]:
    # load the saved model's data using
    model_data = torch.load(model_path, map_location=device)
    thre3d_repr = thre3d_repr_creator(model_data, quantize_colors)
    render_config = model_data[RENDER_CONFIG_TYPE](**model_data[RENDER_CONFIG])

    # ES Addition - Quantize zero order coeffs:
    if num_clusters != 0:
        coeff_grid = thre3d_repr.features.data
        x_grid, y_grid, z_grid, _ = coeff_grid.shape
        coeff_grid = coeff_grid.reshape(x_grid, y_grid, z_grid, NUM_COLOUR_CHANNELS, -1)
        coeff_grid_zero = coeff_grid[..., :1]
        grid_flat = coeff_grid_zero.view(-1, NUM_COLOUR_CHANNELS) 
    
        # run K-means:
        cluster_ids_x, cluster_centers = kmeans(
        X=grid_flat, num_clusters=num_clusters, distance='euclidean', device=device
        )
    
        # quantize coefficients:
        q_grid_flat = cluster_centers[cluster_ids_x[:],:]
        q_grid = q_grid_flat.reshape(x_grid, y_grid, z_grid, NUM_COLOUR_CHANNELS)
        coeff_grid[..., :1] = torch.unsqueeze(q_grid, -1)
        thre3d_repr.features.data = coeff_grid.reshape(x_grid, y_grid, z_grid, -1)

    # return a newly constructed VolumetricModel using the info above
    # and the additional information saved at the time of training :)
    return (
        VolumetricModel(
            thre3d_repr=thre3d_repr,
            render_procedure=model_data[RENDER_PROCEDURE],
            render_config=render_config,
            device=device,
        ),
        model_data[EXTRA_INFO],
    )

def create_voxel_grid_from_saved_info_dict_va(saved_info: Dict[str, Any], 
                                            high_res_densities, 
                                            high_res_features) -> VoxelArtGrid_3DCNN:
    voxel_grid = VoxelArtGrid_3DCNN(
        high_res_densities=high_res_densities,
        high_res_features=high_res_features,
        **saved_info[THRE3D_REPR][CONFIG_DICT]
    )

    voxel_grid.load_state_dict(saved_info[THRE3D_REPR][STATE_DICT])
    return voxel_grid

def create_volumetric_model_from_saved_model_va_3dcnn(
    model_path: Path,
    high_res_model_path: Path,
    device: torch.device = torch.device("cpu"),
    num_clusters: int = 0,
) -> Tuple[VolumetricModel, Dict[str, Any]]:
    
    # 1. Load high res model:
    vol_mod, extra_info = create_volumetric_model_from_saved_model(
            model_path=high_res_model_path,
            thre3d_repr_creator=create_voxel_grid_from_saved_info_dict,
            num_clusters=num_clusters,
        )
    high_res_densities=vol_mod.thre3d_repr.densities.data.to(device)
    high_res_features=vol_mod.thre3d_repr.features.data.to(device)

    model_data = torch.load(model_path, map_location=device)
    thre3d_repr = create_voxel_grid_from_saved_info_dict_va(model_data, \
        high_res_densities, \
        high_res_features)
    render_config = model_data[RENDER_CONFIG_TYPE](**model_data[RENDER_CONFIG])

    # ES Addition - Quantize zero order coeffs:
    if num_clusters != 0:
        coeff_grid = thre3d_repr.features.data
        x_grid, y_grid, z_grid, _ = coeff_grid.shape
        coeff_grid = coeff_grid.reshape(x_grid, y_grid, z_grid, NUM_COLOUR_CHANNELS, -1)
        coeff_grid_zero = coeff_grid[..., :1]
        grid_flat = coeff_grid_zero.view(-1, NUM_COLOUR_CHANNELS) 
    
        # run K-means:
        cluster_ids_x, cluster_centers = kmeans(
        X=grid_flat, num_clusters=num_clusters, distance='euclidean', device=device
        )
    
        # quantize coefficients:
        q_grid_flat = cluster_centers[cluster_ids_x[:],:]
        q_grid = q_grid_flat.reshape(x_grid, y_grid, z_grid, NUM_COLOUR_CHANNELS)
        coeff_grid[..., :1] = torch.unsqueeze(q_grid, -1)
        thre3d_repr.features.data = coeff_grid.reshape(x_grid, y_grid, z_grid, -1)

    # return a newly constructed VolumetricModel using the info above
    # and the additional information saved at the time of training :)
    return (
        VolumetricModel(
            thre3d_repr=thre3d_repr,
            render_procedure=model_data[RENDER_PROCEDURE],
            render_config=render_config,
            device=device,
        ),
        model_data[EXTRA_INFO],
    )
