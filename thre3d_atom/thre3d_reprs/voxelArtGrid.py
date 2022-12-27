""" manually written sort-of-low-level implementation for voxel-based 3D volumetric representations """
from pathlib import Path
from typing import Tuple, NamedTuple, Optional, Callable, Dict, Any
from thre3d_atom.thre3d_reprs.voxels import VoxelGrid
from thre3d_atom.thre3d_reprs.straightThroughSoftmax import StraightThroughSoftMax, ST_SoftMax

import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from torch import Tensor, device
from torch.nn import Module
from PIL import Image
from torch.nn.functional import grid_sample, interpolate, normalize
from thre3d_atom.thre3d_reprs.constants import (
    THRE3D_REPR,
    STATE_DICT,
    u_DENSITIES,
    u_FEATURES,
    CONFIG_DICT,
)
from thre3d_atom.utils.imaging_utils import adjust_dynamic_range

# TODO: Remove these includes later, they are here for debugging
#####
import imageio
import clip
import wandb
import numpy as np

def to8b(x: np.array) -> np.array:
    return (255 * np.clip(x, 0, 1)).astype(np.uint8)

HIGH_DENSITY = 1000.0
SOFTMIN_TEMP = 150.0
EMPTY_SPACE_FACTOR = 10000.0
SAVE_OUTPUTS_INTERVAL = 100
C0 = 0.28209479177387814

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

# clip preprocess for tensor
def clip_transform(n_px):
    return transforms.Compose([
        transforms.Resize(n_px, interpolation=BICUBIC),
        transforms.CenterCrop(n_px),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                             (0.26862954, 0.26130258, 0.27577711)),
    ])

clip_tensor_preprocess = clip_transform(224)


def clip_semantic_loss(output: Tensor,
                       target: Tensor,
                       image_height: int,
                       image_width: int,
                       clip_model,
                       clip_prompt):
    """Caluclates the semantic loss using CLiP"""

    # prepare out images
    out_imgs = torch.reshape(output, (-1, image_height, image_width, 3))
    out_imgs = out_imgs.permute((0, 3, 1, 2))
    out_imgs = clip_tensor_preprocess(out_imgs)
    out_features = normalize(clip_model.encode_image(out_imgs))

    if clip_prompt != "none":
        # prepare text_prompts
        text = clip.tokenize(clip_prompt).to(out_imgs.device)
        target_features = clip_model.encode_text(text).detach()
    else:    
        # prepare target images
        target_imgs = torch.reshape(target, (-1, image_height, image_width, 3)).detach()
        target_imgs = target_imgs.permute((0, 3, 1, 2))
        target_imgs = clip_tensor_preprocess(target_imgs)
        target_features = normalize(clip_model.encode_image(target_imgs))

    # return cosine distance
    distances = torch.mean(1. - torch.cosine_similarity(out_features, target_features))
    return distances

def save_sl_outputs(out_image: Tensor,
                    vox_id_image: Tensor,
                    diff_img: Tensor,
                    output_path: Path,
                    global_step: int):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 3.333))

    # Out Image:
    out_image_np = to8b(out_image.detach().cpu().numpy())
    ax1.imshow(out_image_np)
    ax1.set_title("Output Image")

    # Vox ID Image:
    vox_id_image_np = to8b(vox_id_image.detach().cpu().numpy())
    ax2.imshow(vox_id_image_np, cmap='jet')
    ax2.set_title("Voxel ID Image")

    # Diff Image:
    diff_image_np = to8b(diff_img.detach().cpu().numpy())
    ax3.imshow(diff_image_np, cmap='jet')
    ax3.set_title("Error Image")

    plt.tight_layout()
    plt.savefig(output_path / f"sl_outputs_{global_step}.png", bbox_inches='tight')
    wandb.log({"SL Outputs": wandb.Image(plt)}, step=global_step)


def structural_loss(output: Tensor,
            target: Tensor,
            voxel_ids: Tensor,
            image_height: int,
            image_width: int,
            output_path: Path,
            global_step: int,
            debug_mode: bool=True):
    """Loss whose purpose is to fill holes in the voxelart representation"""
    device = target.device
    out_imgs = torch.reshape(output, (-1, image_height, image_width, 3))
    target_imgs = torch.reshape(target, (-1, image_height, image_width, 3))
    vox_idx_imgs = torch.reshape(voxel_ids, (-1, image_height, image_width, 3))
    num_images = out_imgs.shape[0]
    vox_id_img = torch.empty((num_images, image_height, image_width, 1))
    vox_id_img[:, :, :, 0] = vox_idx_imgs[:, :, :, 0] + \
                        32 * vox_idx_imgs[:, :, :, 1] + \
                        32 * 32 * vox_idx_imgs[:, :, :, 2]

    # get mask for where no voxels appear:
    non_empty_voxel_mask = torch.squeeze(vox_id_img >= 0)

    # get background mask:
    one_channel_target = torch.matmul(target_imgs, torch.ones(3, device=device))
    background_mask = torch.squeeze(one_channel_target == 3.0)
    
    # get diff imgs:
    loss_class = torch.nn.L1Loss(reduction='none')
    diff_img = torch.squeeze(torch.mean(loss_class(out_imgs, target_imgs), dim=-1))

    # zero out the non relevant areas for this loss
    diff_img[non_empty_voxel_mask] = 0
    diff_img[background_mask] = 0
    overall_sa_loss = torch.mean(diff_img)

    if debug_mode:
            if global_step % SAVE_OUTPUTS_INTERVAL == 0:
                if num_images == 1:
                    save_sl_outputs(out_imgs[0], vox_idx_imgs[0], diff_img, output_path, global_step)
                else:
                    save_sl_outputs(out_imgs[0], vox_idx_imgs[0], diff_img[0], output_path, global_step)

    return overall_sa_loss    


def save_sa_ouputs(va_image: Tensor,
                   diff_img: Tensor,
                   min_diff_img: Tensor,
                   output_path: Path,
                   global_step: int):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 3.333))

    # VA Image:
    va_image_np = to8b(va_image.detach().cpu().numpy())
    ax1.imshow(va_image_np)
    ax1.set_title("VoxelArt Image")

    # Diff Image:
    diff_image_np = to8b(diff_img.detach().cpu().numpy())
    ax2.imshow(diff_image_np, cmap='jet')
    ax2.set_title("Error Image")

    # Min-Diff Image:
    min_diff_image_np = to8b(min_diff_img.detach().cpu().numpy())
    ax3.imshow(min_diff_image_np, cmap='jet')
    ax3.set_title("Min-Error Image")

    plt.tight_layout()
    plt.savefig(output_path / f"sa_outputs_{global_step}.png", bbox_inches='tight')
    wandb.log({"SA Outputs": wandb.Image(plt)}, step=global_step)


def sa_loss(output: Tensor,
            target: Tensor,
            voxel_ids: Tensor,
            image_height: int,
            image_width: int,
            output_path: Path,
            global_step: int,
            debug_mode: bool=True):
    """Calculates shift aware loss on an output image"""
    device = target.device
    out_imgs = torch.reshape(output, (-1, image_height, image_width, 3))
    target_imgs = torch.reshape(target, (-1, image_height, image_width, 3))
    vox_idx_imgs = torch.reshape(voxel_ids, (-1, image_height, image_width, 3))
    num_images = out_imgs.shape[0]
    vox_id_img = torch.empty((num_images, image_height, image_width, 1))
    vox_id_img[:, :, :, 0] = vox_idx_imgs[:, :, :, 0] + \
                        32 * vox_idx_imgs[:, :, :, 1] + \
                        32 * 32 * vox_idx_imgs[:, :, :, 2]

    loss_class = torch.nn.L1Loss(reduction='none')
    overall_sa_loss = torch.tensor([0.0], device=device)

    # 1. Iteratre over all frames in batch:
    for img_idx in range(num_images):
        loss_for_frame = torch.tensor([0.0], device=device)
        out_frame = out_imgs[img_idx]
        target_frame = target_imgs[img_idx]

        # 1.a. Calculate diff image:
        diff_img = torch.mean(loss_class(out_frame, target_frame), dim=-1)
        # For debugging:
        min_diff_img = torch.zeros_like(diff_img)
        min_diff_img.requires_grad = False

        # 1.b. Get unique voxel ids:
        id_frame = torch.squeeze(vox_id_img[img_idx])
        unique_ids = torch.unique(id_frame)
        one_channel_target = torch.matmul(target_frame, torch.ones(3, device=target_frame.device))
        foreground_mask = (one_channel_target != 3.0)

        # 2. Iterate over all unique ids:
        for unique_id in unique_ids.tolist():
            if unique_id < 0:
                continue
            diffs_for_voxel = diff_img[torch.logical_and((id_frame == unique_id).to(device), foreground_mask)]
            if torch.numel(diffs_for_voxel) == 0:
                continue
            # 2.a. Add minimum difference:
            #if unique_id < 0:
            #    softmin_vec = torch.nn.functional.softmax(diffs_for_voxel * SOFTMIN_TEMP, dim=-1)
            #else:
            #    softmin_vec = torch.nn.functional.softmin(diffs_for_voxel * SOFTMIN_TEMP, dim=-1)
            #softmin_vec = torch.nn.functional.softmin(diffs_for_voxel * SOFTMIN_TEMP, dim=-1)
            #min_diff = torch.matmul(softmin_vec, diffs_for_voxel)
            min_diff = torch.min(diffs_for_voxel.flatten())
            min_diff_img[torch.logical_and((id_frame == unique_id).to(device), foreground_mask)] = min_diff
            #loss_for_frame += min_diff
        
        #pixels_in_empty_voxels = torch.logical_and(foreground_mask, \
        #    (id_frame < 0.0).to(device))
        #min_diff_img[pixels_in_empty_voxels] = diff_img[pixels_in_empty_voxels]
        #loss_for_frame = torch.mean(min_diff_img)
        # 3. Normalize difference by num of unique ids:
        #loss_for_frame = loss_for_frame / unique_ids.shape[0]
        
        # 4. Add to overall loss:
        loss_for_frame = torch.mean(min_diff_img)
        overall_sa_loss = overall_sa_loss + loss_for_frame
    
        # 5. Penalize empty voxels in foreground:
        #pixels_in_empty_voxels = torch.logical_and(foreground_mask, \
        #    (id_frame < 0.0).to(device))
        #overall_sa_loss += torch.sum(pixels_in_empty_voxels) / EMPTY_SPACE_FACTOR

        # 6. Save debug outputs if debug mode is active:
        if debug_mode and img_idx == 0:
            if global_step % SAVE_OUTPUTS_INTERVAL == 0:
                save_sa_ouputs(out_frame, diff_img, min_diff_img, output_path, global_step)


    overall_sa_loss = overall_sa_loss / num_images
    return overall_sa_loss


def total_variation_loss(output: Tensor,
                         image_height: int,
                         image_width: int,
                         output_path: Path,
                         global_step: int,
                         debug_mode: bool=True):
    device = output.device
    out_imgs = torch.reshape(output, (-1, image_height, image_width, 3))
    bs_img, c_img, h_img, w_img = out_imgs.size()
    tv_h = torch.pow(out_imgs[:,:,1:,:]-out_imgs[:,:,:-1,:], 2).sum().to(device)
    tv_w = torch.pow(out_imgs[:,:,:,1:]-out_imgs[:,:,:,:-1], 2).sum().to(device)
    return (tv_h+tv_w) / (bs_img * c_img * h_img * w_img)



def fill_id_grid(empty_grid: Tensor, grid_dims: Tuple):
    """takes an empty grid and fills it with voxel locations,
    required for shift aware loss"""
    x, y, z = grid_dims
    for xi in range(x): empty_grid[xi, :, :, 0] = xi
    for yi in range(y): empty_grid[:, yi, :, 1] = yi
    for zi in range(z): empty_grid[:, :, zi, 2] = zi

class VoxelSize(NamedTuple):
    """lengths of a single voxel's edges in the x, y and z dimensions
    allows for the possibility of anisotropic voxels"""

    x_size: float = 1.0
    y_size: float = 1.0
    z_size: float = 1.0


class VoxelGridLocation(NamedTuple):
    """indicates where the Voxel-Grid is located in World Coordinate System
    i.e. indicates where the centre of the grid is located in the World
    The Grid is always assumed to be axis aligned"""

    x_coord: float = 0.0
    y_coord: float = 0.0
    z_coord: float = 0.0


class AxisAlignedBoundingBox(NamedTuple):
    """defines an axis-aligned voxel grid's spatial extent"""

    x_range: Tuple[float, float]
    y_range: Tuple[float, float]
    z_range: Tuple[float, float]


class VoxelArtGrid(Module):
    def __init__(
        self,
        # grid values:
        densities: Tensor,
        features: Tensor,
        # grid coordinate-space properties:
        voxel_size: VoxelSize,
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
        palette: Tensor = None,
        num_colors: int = 6,
        use_pure_argmax: bool = False,
        temperature: float = 1.0,
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
            len(densities.shape) == 4 and densities.shape[-1] == 2
        ), f"features should be of shape [W x D x H x 1] as opposed to ({features.shape})"
        assert (
            len(features.shape) == 4
        ), f"features should be of shape [W x D x H x F] as opposed to ({features.shape})"
        assert (
            densities.device == features.device
        ), f"densities and features are not on the same device :("

        super().__init__()

        # either densities or features can be used:
        self._device = features.device

        # initialize the state of the object
        self._densities = densities
        self._features = features
        self._density_preactivation = density_preactivation
        self._density_postactivation = density_postactivation
        self._feature_preactivation = feature_preactivation
        self._feature_postactivation = feature_postactivation
        self._radiance_transfer_function = radiance_transfer_function
        self._grid_location = grid_location
        self._voxel_size = voxel_size
        self._expected_density_scale = expected_density_scale
        self._tunable = tunable
        self._palette = palette
        # ES addition: adding interpolation mode as a public member
        self.interpolation_mode = "bilinear"

        # VoxelArt specific stuff
        self.use_pure_argmax = use_pure_argmax
        self._st_pure_argmax = StraightThroughSoftMax()
        self._num_colors = num_colors
        self.temperature = temperature
        self._softmax_density = ST_SoftMax(1.0)
        self._softmax_features = ST_SoftMax(1.0)

        if tunable:
            self._densities = torch.nn.Parameter(self._densities)
            self._features = torch.nn.Parameter(self._features)

        # note the x, y and z conventions for the width (+ve right), depth (+ve inwards) and height (+ve up)
        self.width_x, self.depth_y, self.height_z = (
            self._features.shape[0],
            self._features.shape[1],
            self._features.shape[2],
        )

        # setup the bounding box planes
        self._aabb = self._setup_bounding_box_planes()

        # set up Voxel ID grid
        self._id_grid = torch.empty((self.width_x, self.depth_y, self.height_z, 3), device=self._device)
        fill_id_grid(self._id_grid, (self.width_x, self.depth_y, self.height_z))


    @property
    def densities(self) -> Tensor:
        return self._densities

    @property
    def features(self) -> Tensor:
        return self._features

    @features.setter
    def features(self, features: Tensor) -> None:
        assert (
            features.shape == self._features.shape
        ), f"new features don't match original feature tensor's dimensions"
        if self._tunable and not isinstance(features, torch.nn.Parameter):
            self._features = torch.nn.Parameter(features)
        else:
            self._features = features

    @densities.setter
    def densities(self, densities: Tensor) -> None:
        assert (
            densities.shape == self._densities.shape
        ), f"new densities don't match original densities tensor's dimensions"
        if self._tunable and not isinstance(densities, torch.nn.Parameter):
            self._densities = torch.nn.Parameter(densities)
        else:
            self._densities = densities

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

    def update_temperature(self, new_temperature):
        self.temperature = new_temperature
        self._softmax_density = ST_SoftMax(self.temperature)

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


    def get_psuedo_voxelArt_img(self, id_batch: Tensor):
        """Gets an output ray batch with softmaxed colors but
        zero one density"""
        # 1. Get non empty voxel mask
        id_batch_1dim = id_batch[..., 0] + id_batch[..., 1] * self.width_x + \
            id_batch[..., 2] * self.depth_y * self.width_x
        non_empty_voxel_mask = id_batch_1dim >= 0

        # 2. Convert id images to indices, set 0 for empty
        feature_indices = torch.zeros_like(id_batch).long()
        feature_indices[non_empty_voxel_mask] = id_batch[non_empty_voxel_mask].long()

        # 3. Index the feature grid
        pseudo_va_batch = torch.zeros((id_batch.shape[0], \
            self._features.shape[-1]), device=self._features.device)
        pseudo_va_batch[:] = torch.squeeze(self._features[feature_indices[:,[0]], \
            feature_indices[:,[1]], feature_indices[:,[2]]])

        # 4. Convert to color and set background pixels to white
        pseudo_va_batch = torch.nn.functional.softmax(pseudo_va_batch, dim=-1)
        pseudo_va_batch_color = C0 * torch.matmul(pseudo_va_batch, self._palette)
        pseudo_va_batch_color[torch.logical_not(non_empty_voxel_mask)] = \
            torch.tensor([10000.0, 10000.0, 10000.0], device=self._features.device)
        pseudo_va_batch_color = torch.sigmoid(pseudo_va_batch_color)

        # 5. Make sure background pixels are white
        #pseudo_va_batch_color[torch.logical_not(non_empty_voxel_mask)] = \
        #    torch.tensor([1.0, 1.0, 1.0], device=self._features.device)

        return pseudo_va_batch_color


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

        if self._palette != None:
            self._palette = self._palette.to(self._device)

        # interpolate and compute densities
        # Note the pre- and post-activations :)
        preactivated_densities = self._density_preactivation(
            self._densities * self._expected_density_scale
        )  # note the use of the expected density scale
        # ADDITION ES: Changed interpolation mode to nearest here

        #----------------------------#
        #         DENSITIES          #
        #----------------------------#

        # old method
        zero_one_tensor = torch.tensor([0, HIGH_DENSITY]).to(self._device)

        # new method
        #zero_one_tensor = torch.tensor([-HIGH_DENSITY, HIGH_DENSITY]).to(self._device)

        # regular densities
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
        
        interpolated_densities = torch.squeeze(interpolated_densities)
        if self.use_pure_argmax:
            interpolated_densities = self._st_pure_argmax(interpolated_densities)
        else:
            interpolated_densities = self._softmax_density(interpolated_densities)
        interpolated_densities = torch.matmul(interpolated_densities, zero_one_tensor)
        interpolated_densities = self._density_postactivation(interpolated_densities)
        interpolated_densities = torch.unsqueeze(interpolated_densities, dim=-1)

        # va densities
        interpolated_densities_va = (
            grid_sample(
                # note the weird z, y, x convention of PyTorch's grid_sample.
                # reference ->
                # https://discuss.pytorch.org/t/surprising-convention-for-grid-sample-coordinates/79997/3
                preactivated_densities[None, ...].permute(0, 4, 3, 2, 1),
                normalized_points[None, None, None, ...],
                mode='nearest',
                align_corners=False,
            )
            .permute(0, 2, 3, 4, 1)
            .squeeze()
        )[
            ..., None
        ]  # note this None is required because of the squeeze operation :sweat_smile:

        interpolated_densities_va = torch.squeeze(interpolated_densities_va)
        interpolated_densities_va = self._st_pure_argmax(interpolated_densities_va)
        interpolated_densities_va = torch.matmul(interpolated_densities_va, zero_one_tensor)
        interpolated_densities_va = self._density_postactivation(interpolated_densities_va)
        interpolated_densities_va = torch.unsqueeze(interpolated_densities_va, dim=-1)

        #----------------------------#
        #          FEATURES          #
        #----------------------------#

        # interpolate and compute features
        # ADDITION ES: Changed interpolation mode to nearest here

        # regular features
        preactivated_features = self._feature_preactivation(self._features)
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

        if self.use_pure_argmax:
            interpolated_features = self._st_pure_argmax(interpolated_features)
        else: 
            interpolated_features = self._softmax_features(interpolated_features)
        interpolated_features = torch.matmul(interpolated_features, self._palette)

        # VA features
        interpolated_features_va = (
            grid_sample(
                preactivated_features[None, ...].permute(0, 4, 3, 2, 1),
                normalized_points[None, None, None, ...],
                mode='nearest',
                align_corners=False,
            )
            .permute(0, 2, 3, 4, 1)
            .squeeze()
        )
        interpolated_features_va = self._feature_postactivation(interpolated_features_va)
        interpolated_features_va = self._st_pure_argmax(interpolated_features_va)
        interpolated_features_va = torch.matmul(interpolated_features_va, self._palette)

        # apply the radiance transfer function if it is not None and if view-directions are available
        if self._radiance_transfer_function is not None and viewdirs is not None:
            interpolated_features = self._radiance_transfer_function(
                interpolated_features, viewdirs
            )
            interpolated_features_va = self._radiance_transfer_function(
                interpolated_features_va, viewdirs
            )

        # get "ID ray"
        id_samples = (
            grid_sample(
                self._id_grid[None, ...].permute(0, 4, 3, 2, 1),
                normalized_points[None, None, None, ...],
                mode='nearest',
                align_corners=False,
            )
            .permute(0, 2, 3, 4, 1)
            .squeeze()
        ).detach()

        # return a unified tensor containing interpolated features and densities, for both regular mode and va
        result_regular = torch.cat([interpolated_features, interpolated_densities], dim=-1)
        result_va = torch.cat([interpolated_features_va, interpolated_densities_va], dim=-1)

        return result_regular, result_va, id_samples

    def create_voxel_grid_from_z1_voxel_grid(self) -> VoxelGrid:
        # get regular densities
        softmaxed_densities = self._softmax_density(self.densities).detach()
        #if self.use_pure_argmax:
        #    softmaxed_densities = self._st_pure_argmax(self.densities).detach()
        #else:
        #    softmaxed_densities = self._softmax_density(self.densities).detach()
        zero_one_tensor = torch.tensor([0, HIGH_DENSITY]).to(self._device)
        output_densities = torch.matmul(softmaxed_densities, zero_one_tensor)
        output_densities = torch.unsqueeze(output_densities, dim=-1)

        # get features
        if self.use_pure_argmax:
            output_features = self._st_pure_argmax(self.features)
        else:
            output_features = self._softmax_features(self.features)
        output_features = torch.matmul(output_features, self._palette)
        

        # return regular grid
        new_voxel_grid = VoxelGrid(
            densities=output_densities,
            features=output_features,
            voxel_size=self.voxel_size,
            **self.get_config_dict(),
        )
        return new_voxel_grid

def scale_zero_one_voxel_grid_with_required_output_size(
    voxel_grid: VoxelArtGrid, output_size: Tuple[int, int, int], mode: str = "trilinear"
) -> VoxelArtGrid:

    # extract relevant information from the original input voxel_grid:
    og_unified_feature_tensor = torch.cat(
        [voxel_grid.features, voxel_grid.densities], dim=-1
    )
    og_voxel_size = voxel_grid.voxel_size

    # compute the new features using pytorch's interpolate function
    new_features = interpolate(
        og_unified_feature_tensor.permute(3, 0, 1, 2)[None, ...],
        size=output_size,
        mode=mode,
        align_corners=False,  # never use align_corners=True :D
        recompute_scale_factor=False,  # this needs to be set for some reason, I can't remember :D
    )[0]
    new_features = new_features.permute(1, 2, 3, 0)

    # a paranoid check that the interpolated features have the exact same output_size as required
    assert new_features.shape[:-1] == output_size

    # new voxel size is also similarly scaled
    new_voxel_size = VoxelSize(
        (og_voxel_size.x_size * voxel_grid.width_x) / output_size[0],
        (og_voxel_size.y_size * voxel_grid.depth_y) / output_size[1],
        (og_voxel_size.z_size * voxel_grid.height_z) / output_size[2],
    )

    # create a new voxel_grid by cloning the input voxel_grid and update the newly scaled properties
    new_voxel_grid = VoxelArtGrid(
        densities=new_features[..., -2:],
        features=new_features[..., :-2],
        voxel_size=new_voxel_size,
        palette=voxel_grid._palette,
        num_colors=voxel_grid._num_colors,
        use_pure_argmax=voxel_grid.use_pure_argmax,
        **voxel_grid.get_config_dict(),
    )

    # noinspection PyProtectedMember
    return new_voxel_grid

def create_voxelArt_grid_from_saved_info_dict(saved_info: Dict[str, Any]) -> VoxelArtGrid:
    densities = torch.empty_like(saved_info[THRE3D_REPR][STATE_DICT][u_DENSITIES])
    features = torch.empty_like(saved_info[THRE3D_REPR][STATE_DICT][u_FEATURES])
    voxel_grid = VoxelArtGrid(
        densities=densities, features=features, \
             **saved_info[THRE3D_REPR][CONFIG_DICT]
    )
    voxel_grid.load_state_dict(saved_info[THRE3D_REPR][STATE_DICT])
    return voxel_grid
