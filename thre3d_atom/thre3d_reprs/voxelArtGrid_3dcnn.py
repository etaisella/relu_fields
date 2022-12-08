""" manually written sort-of-low-level implementation for voxel-based 3D volumetric representations """
from typing import Tuple, NamedTuple, Optional, Callable, Dict, Any
from thre3d_atom.thre3d_reprs.straightThroughSoftmax import StraightThroughSoftMax
from thre3d_atom.thre3d_reprs.voxels import VoxelSize, VoxelGridLocation, AxisAlignedBoundingBox
from thre3d_atom.thre3d_reprs.unet3d import UNet3d
from thre3d_atom.thre3d_reprs.cnn3d_naive_down import cnn3d_naive_down
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import grid_sample, normalize, interpolate

from thre3d_atom.thre3d_reprs.constants import (
    THRE3D_REPR,
    STATE_DICT,
    u_DENSITIES,
    u_FEATURES,
    CONFIG_DICT,
)
from thre3d_atom.utils.imaging_utils import adjust_dynamic_range

HIGH_DENSITY = 1000.0

class VoxelArtGrid_3DCNN(Module):
    def __init__(
        self,
        # grid values:
        high_res_densities: Tensor,
        high_res_features: Tensor,
        # grid coordinate-space properties:
        voxel_size: VoxelSize,
        # new grid dims:
        new_grid_dims: Tuple[int, int, int] = [32, 32, 32],
        grid_location: Optional[VoxelGridLocation] = VoxelGridLocation(),
        # density activations:
        density_preactivation: Callable[[Tensor], Tensor] = torch.abs,
        density_postactivation: Callable[[Tensor], Tensor] = torch.nn.Identity(),
        # feature activations:
        feature_preactivation: Callable[[Tensor], Tensor] = torch.nn.Identity(),
        feature_postactivation: Callable[[Tensor], Tensor] = torch.nn.Identity(),
        # radiance function / transfer function:
        radiance_transfer_function: Callable[[Tensor, Tensor], Tensor] = None,
        expected_density_scale: float = 1.0,
        tunable: bool = False,
        naive_down: bool = False,
        zero_one_density: bool = True,
    ):
        """
        Defines a Voxel-Grid denoting a 3D-volume. To obtain features of a particular point inside
        the volume, we obtain continuous features by doing trilinear interpolation.
        Args:
            densities: Tensor of shape [W x D x H x 1] corresponds to the volumetric density in the scene
            features: Tensor of shape [W x D x H x F] corresponds to the features on the grid-vertices
            voxel_size: Size of each voxel. (could be different in different axis (x, y, z))
            grid_location: Location of the center of the grid
            density_preactivation: the activation to be applied to the raw density values before interpolating.
            density_postactivation: the activation to be applied to the raw density values after interpolating.
            feature_preactivation: the activation to be applied to the features before interpolating.
            feature_postactivation: the activation to be applied to the features after interpolating.
            radiance_transfer_function: the function that maps (can map)
                                        the interpolated features to RGB (radiance) values
            expected_density_scale: expected scale of the raw-density values. Defaults to a nice constant=100.0
            tunable: whether to treat the densities and features Tensors as tunable (trainable) parameters
        """
        # as usual start with assertions about the inputs:
        assert (
            len(high_res_densities.shape) == 4 and high_res_densities.shape[-1] == 1
        ), f"high res densities should be of shape [W x D x H x 1] as opposed to ({high_res_densities.shape})"
        assert (
            len(high_res_features.shape) == 4
        ), f"high res features should be of shape [W x D x H x F] as opposed to ({high_res_features.shape})"
        assert (
            high_res_densities.device == high_res_features.device
        ), f"densities, features and weights are not on the same device :("

        super().__init__()

        # initialize the state of the object
        self._new_grid_dims = new_grid_dims
        self._high_res_densities = high_res_densities
        self._high_res_features = high_res_features
        self._density_preactivation = density_preactivation
        self._density_postactivation = density_postactivation
        self._feature_preactivation = feature_preactivation
        self._feature_postactivation = feature_postactivation
        self._radiance_transfer_function = radiance_transfer_function
        self._grid_location = grid_location
        self._voxel_size = voxel_size
        self._expected_density_scale = expected_density_scale
        self._tunable = tunable
        self._zero_one_density = zero_one_density

        # ES addition: adding interpolation mode as a public member
        self.interpolation_mode = "bilinear"

        # either densities or features can be used:
        self._device = high_res_densities.device

        # init high-res grid
        # Concat densities and features into high-res grid
        self._high_res_grid = torch.cat([self._high_res_densities, self._high_res_features], dim=-1)
        self._high_res_grid = torch.permute(self._high_res_grid, (3, 0, 1, 2))
        self._high_res_grid.requires_grad = False

        # note the x, y and z conventions for the width (+ve right), depth (+ve inwards) and height (+ve up)
        self.width_x, self.depth_y, self.height_z = (
            self._new_grid_dims[0],
            self._new_grid_dims[1],
            self._new_grid_dims[2],
        )

        self.hr_width_x, self.hr_depth_y, self.hr_height_z = (
            high_res_densities.shape[0],
            high_res_densities.shape[1],
            high_res_densities.shape[2],
        )

        small_mode = (self.hr_width_x / self.width_x == 4) and \
            (self.hr_depth_y / self.depth_y == 4) and (self.hr_height_z / self.height_z == 4)  

        # init 3D CNN
        grid_channels_high_res = self._high_res_densities.shape[-1] + self._high_res_features.shape[-1]

        # if we are using zero one density, we have two density channels, one for probability of "zero" density
        # and the other for the probability of "one" density
        if self._zero_one_density:
            grid_channels_low_res = grid_channels_high_res + 1
        else:
            grid_channels_low_res = grid_channels_high_res

        if naive_down:
            self._cnn3d = cnn3d_naive_down(grid_channels_high_res, small=small_mode)
        else:
            self._cnn3d = UNet3d(grid_channels_high_res, grid_channels_low_res, small=small_mode)
        
        self._cnn3d = self._cnn3d.to(self._device)

        # prepare high res features and densities
        self._high_res_densities.requires_grad = False
        self._high_res_features.requires_grad = False

        # setup the bounding box planes
        self._aabb = self._setup_bounding_box_planes()

        # setup straight through softmax
        self._st_softmax = StraightThroughSoftMax()

    @property
    def downsample_weights(self) -> Tensor:
        return self._downsample_weights

    @property
    def aabb(self) -> AxisAlignedBoundingBox:
        return self._aabb

    @property
    def grid_dims(self) -> Tuple[int, int, int]:
        return self.width_x, self.depth_y, self.height_z

    @property
    def voxel_size(self) -> VoxelSize:
        return self._voxel_size

    @voxel_size.setter
    def voxel_size(self, voxel_size: VoxelSize) -> None:
        self._voxel_size = voxel_size

    def get_save_config_dict(self) -> Dict[str, Any]:
        save_config_dict = self.get_config_dict()
        save_config_dict.update({"voxel_size": self._voxel_size})
        return save_config_dict

    def get_config_dict(self) -> Dict[str, Any]:
        return {
            "grid_location": self._grid_location,
            "density_preactivation": self._density_preactivation,
            "density_postactivation": self._density_postactivation,
            "feature_preactivation": self._feature_preactivation,
            "feature_postactivation": self._feature_postactivation,
            "radiance_transfer_function": self._radiance_transfer_function,
            "expected_density_scale": self._expected_density_scale,
            "tunable": self._tunable,
        }

    def _setup_bounding_box_planes(self) -> AxisAlignedBoundingBox:
        # compute half grid dimensions
        half_width = (self.width_x * self._voxel_size.x_size) / 2
        half_depth = (self.depth_y * self._voxel_size.y_size) / 2
        half_height = (self.height_z * self._voxel_size.z_size) / 2

        # compute the AABB (bounding_box_planes)
        width_x_range = (
            self._grid_location.x_coord - half_width,
            self._grid_location.x_coord + half_width,
        )
        depth_y_range = (
            self._grid_location.y_coord - half_depth,
            self._grid_location.y_coord + half_depth,
        )
        height_z_range = (
            self._grid_location.z_coord - half_height,
            self._grid_location.z_coord + half_height,
        )

        # return the computed planes in the packed AABB datastructure:
        return AxisAlignedBoundingBox(
            x_range=width_x_range,
            y_range=depth_y_range,
            z_range=height_z_range,
        )

    def _normalize_points(self, points: Tensor) -> Tensor:
        normalized_points = torch.empty_like(points, device=points.device)
        for coordinate_index, coordinate_range in enumerate(self._aabb):
            normalized_points[:, coordinate_index] = adjust_dynamic_range(
                points[:, coordinate_index],
                drange_in=coordinate_range,
                drange_out=(-1.0, 1.0),
                slack=True,
            )
        return normalized_points

    def extra_repr(self) -> str:
        return (
            f"grid_dims: {(self.width_x, self.depth_y, self.height_z)}, "
            f"feature_dims: {self._features.shape[-1]}, "
            f"voxel_size: {self._voxel_size}, "
            f"grid_location: {self._grid_location}, "
            f"tunable: {self._tunable}"
        )

    def get_bounding_volume_vertices(self) -> Tensor:
        x_min, x_max = self._aabb.x_range
        y_min, y_max = self._aabb.y_range
        z_min, z_max = self._aabb.z_range
        return torch.tensor(
            [
                [x_min, y_min, z_min],
                [x_min, y_min, z_max],
                [x_min, y_max, z_min],
                [x_min, y_max, z_max],
                [x_max, y_min, z_min],
                [x_max, y_min, z_max],
                [x_max, y_max, z_min],
                [x_max, y_max, z_max],
            ],
            dtype=torch.float32,
        )

    def test_inside_volume(self, points: Tensor) -> Tensor:
        """
        tests whether the points are inside the AABB or not
        Args:
            points: Tensor of shape [N x 3 (NUM_COORD_DIMENSIONS)]
        Returns: Tensor of shape [N x 1]  (boolean)
        """
        return torch.logical_and(
            torch.logical_and(
                torch.logical_and(
                    points[..., 0:1] > self._aabb.x_range[0],
                    points[..., 0:1] < self._aabb.x_range[1],
                ),
                torch.logical_and(
                    points[..., 1:2] > self._aabb.y_range[0],
                    points[..., 1:2] < self._aabb.y_range[1],
                ),
            ),
            torch.logical_and(
                points[..., 2:] > self._aabb.z_range[0],
                points[..., 2:] < self._aabb.z_range[1],
            ),
        )

    def forward(self, points: Tensor, viewdirs: Optional[Tensor] = None) -> Tensor:
        """
        computes the features/radiance at the requested 3D points
        Args:
            points: Tensor of shape [N x 3 (NUM_COORD_DIMENSIONS)]
            viewdirs: Tensor of shape [N x 3 (NUM_COORD_DIMENSIONS)]
                      this tensor represents viewing directions in world-coordinate-system
        Returns: either Tensor of shape [N x <3 + 1> (NUM_COLOUR_CHANNELS + density)]
                 or of shape [N x <features + 1> (number of features + density)], depending upon
                 whether the `self._radiance_transfer_function` is None.
        """
        # obtain the range-normalized points for interpolation
        normalized_points = self._normalize_points(points)

        # get low res-grid from high-res grid via 3D-CNN
        #torch.save(self._high_res_grid, "high_res_grid.pt")
        low_res_grid = self._cnn3d(torch.unsqueeze(self._high_res_grid, 0))
        low_res_grid = torch.squeeze(low_res_grid)
        low_res_grid = torch.permute(low_res_grid, (1, 2, 3, 0))

        if self._zero_one_density:
            low_res_densities = low_res_grid[..., :2]
            low_res_features = low_res_grid[..., 2:]
        else:
            low_res_densities = torch.unsqueeze(low_res_grid[..., 0], -1)
            low_res_features = low_res_grid[..., 1:]

        # interpolate and compute densities
        # Note the pre- and post-activations :)
        preactivated_densities = self._density_preactivation(
            low_res_densities * self._expected_density_scale
        )  # note the use of the expected density scale
        # ADDITION ES: Changed interpolation mode to nearest here
        interpolated_densities = (
            grid_sample(
                # note the weird z, y, x convention of PyTorch's grid_sample.
                # reference ->
                # https://discuss.pytorch.org/t/surprising-convention-for-grid-sample-coordinates/79997/3
                preactivated_densities[None, ...].permute(0, 4, 3, 2, 1),
                normalized_points[None, None, None, ...],
                mode=self.interpolation_mode,
                align_corners=False,
            )
            .permute(0, 2, 3, 4, 1)
            .squeeze()
        )[
            ..., None
        ]  # note this None is required because of the squeeze operation :sweat_smile:
        interpolated_densities = self._density_postactivation(interpolated_densities)
        if self._zero_one_density:
            interpolated_densities = torch.squeeze(interpolated_densities)
            interpolated_densities = self._st_softmax(interpolated_densities)
            zero_one_tensor = torch.tensor([0, HIGH_DENSITY]).to(self._device)
            interpolated_densities = torch.matmul(interpolated_densities, zero_one_tensor)
            interpolated_densities = torch.unsqueeze(interpolated_densities, dim=-1)

        # interpolate and compute features
        # ADDITION ES: Changed interpolation mode to nearest here
        preactivated_features = self._feature_preactivation(low_res_features)
        interpolated_features = (
            grid_sample(
                preactivated_features[None, ...].permute(0, 4, 3, 2, 1),
                normalized_points[None, None, None, ...],
                mode=self.interpolation_mode,
                align_corners=False,
            )
            .permute(0, 2, 3, 4, 1)
            .squeeze()
        )
        interpolated_features = self._feature_postactivation(interpolated_features)

        # apply the radiance transfer function if it is not None and if view-directions are available
        if self._radiance_transfer_function is not None and viewdirs is not None:
            interpolated_features = self._radiance_transfer_function(
                interpolated_features, viewdirs
            )

        # return a unified tensor containing interpolated features and densities
        return torch.cat([interpolated_features, interpolated_densities], dim=-1)
