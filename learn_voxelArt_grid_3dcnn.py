from pathlib import Path
from xml.sax.handler import DTDHandler
from PIL import Image as im
import click
import imageio
import torch
from torch.backends import cudnn

import gc
import GPUtil
import torch


from thre3d_atom.data.datasets import PosedImagesDataset

from thre3d_atom.thre3d_reprs.voxels import VoxelSize, VoxelGridLocation
from thre3d_atom.thre3d_reprs.voxelArtGrid_3dcnn import VoxelArtGrid_3DCNN
from thre3d_atom.thre3d_reprs.unet3d import get_noise

from thre3d_atom.modules.volumetric_model import VolumetricModel

from thre3d_atom.modules.volumetric_model import (
    create_volumetric_model_from_saved_model,
)
from thre3d_atom.thre3d_reprs.voxels import (create_voxel_grid_from_saved_info_dict, 
    scale_voxel_grid_with_required_output_size
)
from thre3d_atom.modules.trainers import train_sh_vox_grid_vol_mod_with_posed_images
from thre3d_atom.rendering.volumetric.utils.misc import (
    compute_expected_density_scale_for_relu_field_grid,
)
from thre3d_atom.utils.constants import HEMISPHERICAL_RADIUS, CAMERA_INTRINSICS
from thre3d_atom.utils.imaging_utils import (
    get_thre360_animation_poses,
    get_thre360_spiral_animation_poses,
)
from thre3d_atom.visualizations.animations import (
    render_camera_path_for_volumetric_model_3_coeff_modes,
)
from thre3d_atom.thre3d_reprs.renderers import (
    render_sh_voxel_grid,
    SHVoxGridRenderConfig,
)
from thre3d_atom.utils.logging import log
from thre3d_atom.utils.misc import log_config_to_disk
from easydict import EasyDict

# Age-old custom option for fast training :)
cudnn.benchmark = True
# Also set torch's multiprocessing start method to spawn
# refer -> https://github.com/pytorch/pytorch/issues/40403
# for more information. Some stupid PyTorch stuff to take care of
torch.multiprocessing.set_start_method("spawn")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------------------------------------------------------------
#  Command line configuration for the script                                          |
# -------------------------------------------------------------------------------------
# fmt: off
# noinspection PyUnresolvedReferences
@click.command()
# Required arguments:
@click.option("-i", "--high_res_model_path", type=click.Path(file_okay=True, dir_okay=False),
              required=True, help="path to the trained High Resolution model")
@click.option("-r", "--output_render_path", type=click.Path(file_okay=False, dir_okay=True),
              required=True, help="path for saving rendered output")
@click.option("-o", "--out_model_path", type=click.Path(file_okay=True, dir_okay=True),
              required=True, help="path to the output VA model")
@click.option("-d", "--data_path", type=click.Path(file_okay=False, dir_okay=True),
              required=True, help="path to the input dataset")
@click.option("-n", "--naive_downsample_mode", type=click.BOOL, required=False, default=False,
              help="whether to naivly downsample a 128^3 grid to the given size or to learn a new one", 
              show_default=True)

# Non-required Training Configurations options:
@click.option("--white_bkgd", type=click.BOOL, required=False, default=True,
              help="whether to use white background for training with synthetic (background-less) scenes :)",
              show_default=True)  # this option is also used in pre-processing the dataset
@click.option("--data_downsample_factor", type=click.FloatRange(min=1.0), required=False,
              default=2.0, help="downscale factor for the input images if needed."
                                "Note the default, for training NeRF-based scenes", show_default=True)
@click.option("--normalize_scene_scale", type=click.BOOL, required=False, default=False,
              help="whether to normalize the scene's scale to unit radius", show_default=True)
@click.option("--train_num_samples_per_ray", type=click.INT, required=False, default=256,
              help="number of samples taken per ray during training", show_default=True)
@click.option("--render_num_samples_per_ray", type=click.INT, required=False, default=512,
              help="number of samples taken per ray during rendering", show_default=True)
@click.option("--parallel_rays_chunk_size", type=click.INT, required=False, default=32768 / 2,
              help="number of parallel rays processed on the GPU for accelerated rendering", show_default=True)
@click.option("--ray_batch_size", type=click.INT, required=False, default=16384 / 4,
              help="number of randomly sampled rays used per training iteration", show_default=True)
@click.option("--num_iterations_per_stage", type=click.INT, required=False, default=1800,
              help="number of training iterations performed per stage", show_default=True)
@click.option("--learning_rate", type=click.FLOAT, required=False, default=0.001,
              help="learning rate used at the beginning (ADAM OPTIMIZER)", show_default=True)
@click.option("--lr_decay_steps_per_stage", type=click.INT, required=False, default=500,
              help="number of iterations after which lr is exponentially decayed per stage", show_default=True)
@click.option("--lr_decay_gamma_per_stage", type=click.FLOAT, required=False, default=0.1,
              help="value of gamma for exponential lr_decay (happens per stage)", show_default=True)
@click.option("--stagewise_lr_decay_gamma", type=click.FLOAT, required=False, default=0.9,
              help="value of gamma used for reducing the learning rate after each stage", show_default=True)
@click.option("--apply_diffuse_render_regularization", type=click.BOOL, required=False, default=True,
              help="whether to apply the diffuse render regularization."
                   "this is a weird conjure of mine, where we ask the diffuse render "
                   "to match, as closely as possible, the GT-possibly-specular one :D"
                   "can be off or on, on yields stabler training :) ", show_default=True)
@click.option("--num_workers", type=click.INT, required=False, default=4,
              help="number of worker processes used for loading the data using the dataloader"
                   "note that this will be ignored if GPU-caching of the data is successful :)", show_default=True)

# Various frequencies:
@click.option("--save_frequency", type=click.INT, required=False, default=250,
              help="number of iterations after which a model is saved", show_default=True)
@click.option("--test_frequency", type=click.INT, required=False, default=250,
              help="number of iterations after which test metrics are computed", show_default=True)
@click.option("--feedback_frequency", type=click.INT, required=False, default=100,
              help="number of iterations after which rendered feedback is generated", show_default=True)
@click.option("--summary_frequency", type=click.INT, required=False, default=50,
              help="number of iterations after which training-loss/other-summaries are logged", show_default=True)

# Non-required Render configuration options:
@click.option("--overridden_num_samples_per_ray", type=click.IntRange(min=1), default=512,
              required=False, help="overridden (increased) num_samples_per_ray for beautiful renders :)")
@click.option("--render_scale_factor", type=click.FLOAT, default=2.0,
              required=False, help="overridden (increased) resolution (again :D) for beautiful renders :)")
@click.option("--camera_path", type=click.Choice(["thre360", "spiral"]), default="thre360",
              required=False, help="which camera path to use for rendering the animation")
# thre360_path options
@click.option("--camera_pitch", type=click.FLOAT, default=60.0,
              required=False, help="pitch-angle value for the camera for 360 path animation")
@click.option("--num_frames", type=click.IntRange(min=1), default=60,
              required=False, help="number of frames in the video")

# Miscellaneous modes
@click.option("--verbose_rendering", type=click.BOOL, required=False, default=False,
              help="whether to show progress while rendering feedback during training"
                   "can be turned-off when running on server-farms :D", show_default=True)
@click.option("--fast_debug_mode", type=click.BOOL, required=False, default=False,
              help="whether to use the fast debug mode while training "
                   "(skips testing and some lengthy visualizations)", show_default=True)

# spiral path options
@click.option("--vertical_camera_height", type=click.FLOAT, default=3.0,
              required=False, help="height at which the camera spiralling will happen")
@click.option("--num_spiral_rounds", type=click.IntRange(min=1), default=2,
              required=False, help="number of rounds made while transitioning between spiral radii")
@click.option("--grid_world_size", type=click.FLOAT, nargs=3, required=False, default=(3.0, 3.0, 3.0),
              help="size (extent) of the grid in world coordinate system."
                   "Please carefully note it's use in conjunction with the normalization :)", show_default=True)
@click.option("--grid_location", type=click.FLOAT, nargs=3, required=False, default=(0.0, 0.0, 0.0),
              help="dimensions (#voxels) of the grid along x, y and z axes", show_default=True)
              
# Non-required video options:
@click.option("--fps", type=click.IntRange(min=1), default=60,
              required=False, help="frames per second of the video")

# ES ADDITIONS:
@click.option("--clusters", type=click.IntRange(min=0), default=0,
              required=False, help="number of SH cluster centers")
@click.option("--init_noise", type=click.BOOL, default=False,
              required=False, help="if this is True the input of the 3D unet is constant noise")
@click.option("--zero_one_density", type=click.BOOL, default=True,
              required=False, help="whether to use zero one density :( ")
@click.option("--new_grid_dims", type=click.INT, nargs=3, required=False, default=(64, 64, 64),
              help="dimensions (#voxels) of the new grid along x, y and z axes", show_default=True)

# fmt: on
# -------------------------------------------------------------------------------------
def main(**kwargs) -> None:
    # load the requested configuration for the training
    config = EasyDict(kwargs)

# -------------------------------------------------------------------------------------
#  Learning Voxel Art Grid                                                            |
# -------------------------------------------------------------------------------------

    # parse os-checked path-strings into Pathlike Paths :)
    highres_model_path = Path(config.high_res_model_path)
    output_render_path = Path(config.output_render_path)

    # create the output path if it doesn't exist
    output_render_path.mkdir(exist_ok=True, parents=True)

    if config.init_noise:
        print("Setting U-Net input to noise")
        high_res_densities = get_noise(1, 'noise', (128, 128, 128)).to(device).detach()
        high_res_features = get_noise(27, 'noise', (128, 128, 128)).to(device).detach()
    else:
        # load volumetric_model from the model_path
        print("Setting U-Net input to high res model")
        vol_mod, extra_info = create_volumetric_model_from_saved_model(
            model_path=highres_model_path,
            thre3d_repr_creator=create_voxel_grid_from_saved_info_dict,
            num_clusters=config.clusters,
        )
        high_res_densities=vol_mod.thre3d_repr.densities.data.to(device)
        high_res_features=vol_mod.thre3d_repr.features.data.to(device)

    # ES Addition: Training new model!
    # 1. Create VA model
    # 1.a. Set up Activation dict
    vox_grid_density_activations_dict = {
            "density_preactivation": torch.nn.Identity(),
            "density_postactivation": torch.nn.ReLU(),
            # note this expected density value :)
            "expected_density_scale": compute_expected_density_scale_for_relu_field_grid(
                config.grid_world_size
            ),
        }
        
    
    # 1.b. Set up Voxel Size
    voxel_size = VoxelSize(*[dim_size / grid_dim for dim_size, grid_dim
                             in zip(config.grid_world_size, config.new_grid_dims)])
    
    # 1.c. Create VoxelArtGrid_3DCNN class
    voxel_grid = VoxelArtGrid_3DCNN(
        high_res_densities=high_res_densities,
        high_res_features=high_res_features,
        new_grid_dims=config.new_grid_dims,
        voxel_size=voxel_size,
        grid_location=VoxelGridLocation(*config.grid_location),
        **vox_grid_density_activations_dict,
        tunable=True,
        naive_down=config.naive_downsample_mode,
        zero_one_density=config.zero_one_density
    )

    # 2. Set Dataset Parameters
    # parse os-checked path-strings into Pathlike Paths :)
    data_path = Path(config.data_path)
    output_model_path = Path(config.out_model_path)

    # save a copy of the configuration for reference
    log.info("logging configuration file ...")
    log_config_to_disk(config, output_model_path)

    # create a datasets for training and testing:
    train_dataset, test_dataset = (
        PosedImagesDataset(
            images_dir=data_path / mode,
            camera_params_json=data_path / f"{mode}_camera_params.json",
            normalize_scene_scale=config.normalize_scene_scale,
            downsample_factor=config.data_downsample_factor,
            rgba_white_bkgd=config.white_bkgd,
        )
    for mode in ("train", "test")
    )

    # 3. Set up a volumetricModel using the previously created voxel-grid
    # noinspection PyTypeChecker
    vox_grid_vol_mod = VolumetricModel(
        thre3d_repr=voxel_grid,
        render_procedure=render_sh_voxel_grid,
        render_config=SHVoxGridRenderConfig(
            num_samples_per_ray=config.train_num_samples_per_ray,
            camera_bounds=train_dataset.camera_bounds,
            white_bkgd=config.white_bkgd,
            render_num_samples_per_ray=config.render_num_samples_per_ray,
            parallel_rays_chunk_size=config.parallel_rays_chunk_size,
        ),
        device=device,
    )

    # 4. Call training function
    # train the model:
    train_sh_vox_grid_vol_mod_with_posed_images(
        vol_mod=vox_grid_vol_mod,
        train_dataset=train_dataset,
        output_dir=output_model_path,
        test_dataset=test_dataset,
        ray_batch_size=config.ray_batch_size,
        num_stages=1,
        num_iterations_per_stage=config.num_iterations_per_stage,
        scale_factor=1.0, # not really relevant as we use only one stage
        learning_rate=config.learning_rate,
        lr_decay_gamma_per_stage=config.lr_decay_gamma_per_stage,
        lr_decay_steps_per_stage=config.lr_decay_steps_per_stage,
        stagewise_lr_decay_gamma=config.stagewise_lr_decay_gamma,
        save_freq=config.save_frequency,
        test_freq=config.test_frequency,
        feedback_freq=config.feedback_frequency,
        summary_freq=config.summary_frequency,
        apply_diffuse_render_regularization=config.apply_diffuse_render_regularization,
        num_workers=config.num_workers,
        verbose_rendering=config.verbose_rendering,
        fast_debug_mode=config.fast_debug_mode,
        voxelArt_mode=True,
    )

# -------------------------------------------------------------------------------------
#  Rendering Output Video                                                             |
# -------------------------------------------------------------------------------------

    hemispherical_radius = extra_info[HEMISPHERICAL_RADIUS]
    camera_intrinsics = extra_info[CAMERA_INTRINSICS]

    # generate animation using the newly_created vol_mod :)
    if config.camera_path == "thre360":
        camera_pitch, num_frames = config.camera_pitch, config.num_frames
        animation_poses = get_thre360_animation_poses(
            hemispherical_radius=hemispherical_radius,
            camera_pitch=camera_pitch,
            num_poses=num_frames,
        )
    elif config.camera_path == "spiral":
        vertical_camera_height, num_frames = (
            config.vertical_camera_height,
            config.num_frames,
        )
        animation_poses = get_thre360_spiral_animation_poses(
            horizontal_radius_range=(hemispherical_radius / 8.0, hemispherical_radius),
            vertical_camera_height=vertical_camera_height,
            num_rounds=config.num_spiral_rounds,
            num_poses=num_frames,
        )
    else:
        raise ValueError(
            f"Unknown camera_path ``{config.camera_path}'' requested."
            f"Only available options are: ['thre360' and 'spiral']"
        )

    animation_frames = render_camera_path_for_volumetric_model_3_coeff_modes(
        vol_mod=vol_mod,
        camera_path=animation_poses,
        camera_intrinsics=camera_intrinsics,
        overridden_num_samples_per_ray=config.overridden_num_samples_per_ray,
        render_scale_factor=config.render_scale_factor,
    )

    imageio.mimwrite(
        output_render_path / "rendered_video.mp4",
        animation_frames,
        fps=config.fps,
    )


if __name__ == "__main__":
    torch.cuda.empty_cache()
    GPUtil.showUtilization()
    gc.collect()
    main()
