from pathlib import Path

import click
import torch
from easydict import EasyDict
from torch.backends import cudnn

from thre3d_atom.data.datasets import PosedImagesDataset
from thre3d_atom.modules.trainers import train_sh_vox_grid_vol_mod_with_posed_images
from thre3d_atom.modules.volumetric_model import VolumetricModel
from thre3d_atom.rendering.volumetric.utils.misc import (
    compute_expected_density_scale_for_relu_field_grid,
)
from thre3d_atom.thre3d_reprs.renderers import (
    render_sh_voxelArt_grid,
    SHVoxGridRenderConfig,
)
from thre3d_atom.thre3d_reprs.voxelArtGrid import VoxelArtGrid, VoxelSize, VoxelGridLocation
from thre3d_atom.utils.constants import NUM_COLOUR_CHANNELS
from thre3d_atom.utils.logging import log
from thre3d_atom.utils.misc import log_config_to_disk

import wandb
import clip
from datetime import datetime
from get_palette import *

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
@click.option("-d", "--data_path", type=click.Path(file_okay=False, dir_okay=True),
              required=True, help="path to the input dataset")
@click.option("-o", "--output_path", type=click.Path(file_okay=False, dir_okay=True),
              required=True, help="path for training output")

# Input dataset related arguments:
@click.option("--data_downsample_factor", type=click.FloatRange(min=1.0), required=False,
              default=4.0, help="downscale factor for the input images if needed."
                                "Note the default, for training NeRF-based scenes", show_default=True)

# Voxel-grid related arguments:
@click.option("--grid_dims", type=click.INT, nargs=3, required=False, default=(32, 32, 32),
              help="dimensions (#voxels) of the grid along x, y and z axes", show_default=True)
@click.option("--grid_location", type=click.FLOAT, nargs=3, required=False, default=(0.0, 0.0, 0.0),
              help="dimensions (#voxels) of the grid along x, y and z axes", show_default=True)
@click.option("--normalize_scene_scale", type=click.BOOL, required=False, default=False,
              help="whether to normalize the scene's scale to unit radius", show_default=True)
@click.option("--grid_world_size", type=click.FLOAT, nargs=3, required=False, default=(3.0, 3.0, 3.0),
              help="size (extent) of the grid in world coordinate system."
                   "Please carefully note it's use in conjunction with the normalization :)", show_default=True)
@click.option("--sh_degree", type=click.INT, required=False, default=0,
              help="degree of the spherical harmonics coefficients to be used. "
                   "Supported values: [0, 1, 2, 3]", show_default=True)
# -------------------------------------------------------------------------------------
#                        !!! :) MOST IMPORTANT OPTION :) !!!                          |
# -------------------------------------------------------------------------------------
@click.option("--use_relu_field", type=click.BOOL, required=False, default=True,    # |
              help="whether to use relu_fields or revert to traditional grids",     # |
              show_default=True)                                                    # |
# -------------------------------------------------------------------------------------

# Rendering related arguments:
@click.option("--render_num_samples_per_ray", type=click.INT, required=False, default=256,
              help="number of samples taken per ray during rendering", show_default=True)
@click.option("--parallel_rays_chunk_size", type=click.INT, required=False, default=32768 / 4,
              help="number of parallel rays processed on the GPU for accelerated rendering", show_default=True)
@click.option("--white_bkgd", type=click.BOOL, required=False, default=True,
              help="whether to use white background for training with synthetic (background-less) scenes :)",
              show_default=True)  # this option is also used in pre-processing the dataset

# Training related arguments:
@click.option("--ray_batch_size", type=click.INT, required=False, default=49152,
              help="number of randomly sampled rays used per training iteration", show_default=True)
@click.option("--train_num_samples_per_ray", type=click.INT, required=False, default=256,
              help="number of samples taken per ray during training", show_default=True)
@click.option("--num_stages", type=click.INT, required=False, default=3,
              help="number of progressive growing stages used in training", show_default=True)
@click.option("--num_iterations_per_stage", type=click.INT, required=False, default=800,
              help="number of training iterations performed per stage", show_default=True)
@click.option("--scale_factor", type=click.FLOAT, required=False, default=2.0,
              help="factor by which the grid is up-scaled after each stage", show_default=True)
@click.option("--learning_rate", type=click.FLOAT, required=False, default=0.03,
              help="learning rate used at the beginning (ADAM OPTIMIZER)", show_default=True)
@click.option("--lr_decay_steps_per_stage", type=click.INT, required=False, default=100,
              help="number of iterations after which lr is exponentially decayed per stage", show_default=True)
@click.option("--lr_decay_gamma_per_stage", type=click.FLOAT, required=False, default=1.0,
              help="value of gamma for exponential lr_decay (happens per stage)", show_default=True)
@click.option("--stagewise_lr_decay_gamma", type=click.FLOAT, required=False, default=1.0,
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

# Miscellaneous modes
@click.option("--verbose_rendering", type=click.BOOL, required=False, default=False,
              help="whether to show progress while rendering feedback during training"
                   "can be turned-off when running on server-farms :D", show_default=True)
@click.option("--fast_debug_mode", type=click.BOOL, required=False, default=False,
              help="whether to use the fast debug mode while training "
                   "(skips testing and some lengthy visualizations)", show_default=True)

# Etai Additions:
@click.option("--temperature_gamma", type=click.FLOAT, required=False, default=1.0,
              help="the gamma by which we increase temperature", show_default=True)
@click.option("--raise_temperature_iter", type=click.INT, required=False, default=100,
              help="iteration in which to raise temperature", show_default=True)
@click.option("--use_pure_argmax", type=click.BOOL, required=False, default=False,
              help="a flag to determine whether to use pure argmax instead of heated softmax", show_default=True)
@click.option("--num_colors", type=click.INT, required=False, default=5,
              help="number of colors in palette", show_default=True)
@click.option("--sa_start_iter", type=click.INT, required=False, default=-1,
              help="Iteration in which we start using shift aware loss", show_default=True)              
@click.option("--sa_init_weight", type=click.FLOAT, required=False, default=0.0,
              help="Initial weight of shift aware loss in total loss", show_default=True)
@click.option("--sa_gamma", type=click.FLOAT, required=False, default=1.0,
              help="Gamma which we raise the shift aware loss by at every interval", show_default=True)
@click.option("--sa_interval", type=click.INT, required=False, default=100,
              help="interval in iterations where we raise the SA weight", show_default=True)
@click.option("--sa_percentile", type=click.FLOAT, required=False, default=0.1,
              help="percentile in SA loss", show_default=True)
@click.option("--semantic_weight", type=click.FLOAT, required=False, default=0.0,
              help="Weight of the semantic loss (CLiP)", show_default=True)
@click.option("--accumulation_iters", type=click.INT, required=False, default=1,
              help="iterations before optimization step", show_default=True)
@click.option("--prompt", type=click.STRING, required=False, default="none",
              help="prompt used for semantic loss (CLIP or diffusion)", show_default=True)
@click.option("--start_semantic_iter", type=click.INT, required=False, default=-1,
              help="iteration where we start using semantic loss", show_default=True)
@click.option("--convex_hull_extraction", type=click.BOOL, required=False, default=False,
              help="Use fastLayerDecomposition for palette extraction", show_default=True)
@click.option("--directional_weight", type=click.FLOAT, required=False, default=0.0,
              help="Weight for directional loss", show_default=True)
@click.option("--quantize_colors", type=click.BOOL, required=False, default=True,
              help="A flag that determines weather to quantize colors and use a palette", \
                show_default=True)
@click.option("--palette_learning_mode", type=click.BOOL, required=False, default=False,
              help="A flag that determines weather learn the palette", \
                show_default=True)
@click.option("--min_distance_weight", type=click.FLOAT, required=False, default=0.0,
              help="Weight for minimal distance (only relevant in palette learning mode)", show_default=True)
@click.option("--start_sdl_iter", type=click.INT, required=False, default=-1,
              help="iteration to start score distillation in", show_default=True)
@click.option("--va_structure_mode", type=click.BOOL, required=False, default=False,
              help="A flag that determines weather to allow va to learn structure", \
                show_default=True)

# fmt: on
# -------------------------------------------------------------------------------------
def main(**kwargs) -> None:
    # load the requested configuration for the training

    config = EasyDict(kwargs)

    assert(not (config.quantize_colors == True and \
        config.palette_learning_mode == True)), f"Illegal for both quantize colors and palette learning mode to be true"

    if config.semantic_weight != 0:
        # CLIP Setup
        print(f"Available CLiP models - {clip.available_models()}")
        clip_model, clip_preprocess = clip.load("ViT-B/16", device=device, jit=False)
        for parameter in clip_model.parameters():
            parameter.requires_grad = False
    else:
        clip_model = None

    wandb.init(project='VoxelArtReluFields', entity="etaisella",
                   config=dict(config), name="test " + str(datetime.now()), 
                   id=wandb.util.generate_id())

    # parse os-checked path-strings into Pathlike Paths :)
    data_path = Path(config.data_path)
    output_path = Path(config.output_path)

    # save a copy of the configuration for reference
    log.info("logging configuration file ...")
    log_config_to_disk(config, output_path)

    # create training data set:
    train_dataset = PosedImagesDataset(
            images_dir=data_path / "train",
            camera_params_json=data_path / f"train_camera_params.json",
            normalize_scene_scale=config.normalize_scene_scale,
            downsample_factor=config.data_downsample_factor,
            rgba_white_bkgd=config.white_bkgd,
    )
        
    # get palette here
    palette = None
    num_colors = config.num_colors
    if config.quantize_colors or config.palette_learning_mode:
        print("Calculating Palette - be patient!")
        concat_image = concatenate_images(data_path/"train")
        if config.convex_hull_extraction:
            palette, num_colors = get_palette_convex_hull(concat_image)
        else:
            palette = get_palette(concat_image, config.num_colors)
        wandb.log({"palette": wandb.Image(torch.unsqueeze(torch.permute(palette, (1, 0)), dim=-2))}, step=0)

        # save palette
        output_folder = os.path.join(data_path, 'palette')
        os.makedirs(output_folder, exist_ok=True)
        with open(os.path.join(output_folder, 'palette.npy'), 'wb') as f:
            np.save(f, palette)

        # ES: This transform is necessary because the render process multiplies by C0
        C0 = 0.28209479177387814
        if config.palette_learning_mode:
            EPSILON = 1e-5
            palette = torch.clip(palette, min=EPSILON, max=1.0-EPSILON)
            palette = torch.logit(palette)
        palette = palette / C0

    # create a datasets for training and testing:
    test_dataset, warped_dataset = (
        PosedImagesDataset(
            images_dir=data_path / mode,
            camera_params_json=data_path / f"{mode}_camera_params.json",
            normalize_scene_scale=config.normalize_scene_scale,
            downsample_factor=config.data_downsample_factor,
            rgba_white_bkgd=config.white_bkgd,
        )
    for mode in ("test", "warped")
    )
    
    # Choose the proper activations dict based on the requested mode:
    if config.use_relu_field:
        vox_grid_density_activations_dict = {
            "density_preactivation": torch.nn.Identity(),
            "density_postactivation": torch.nn.ReLU(),
            # note this expected density value :)
            "expected_density_scale": compute_expected_density_scale_for_relu_field_grid(
                config.grid_world_size
            ),
        }
    else:
        vox_grid_density_activations_dict = {
            "density_preactivation": torch.abs,
            "density_postactivation": torch.nn.Identity(),
            "expected_density_scale": 1.0,  # Also note this expected density value :wink:
        }
    # The use of terminologies pre-activation and post-activations is inspired from the
    # amazing DVGo work -> https://sunset1995.github.io/dvgo/
    # Please feel free to check out their work for lot more detailed and exhaustive experiments
    # On 3D scene reconstructions.
    # P.S. Not a criticism :wink:, but there isn't and can never be such a thing as IN-ACTIVATION
    # IT'S NOT A FEATURE, IT'S A BUG! :D :D

    # construct the VoxelGrid thre3d_repr :)
    # fmt: off
    densities = torch.empty((*config.grid_dims, 2), dtype=torch.float32, device=device)
    torch.nn.init.uniform_(densities, -1.0, 1.0)
    if config.quantize_colors:
        features = torch.empty((*config.grid_dims, num_colors), dtype=torch.float32, device=device)
    else:
        features = torch.empty((*config.grid_dims, 3), dtype=torch.float32, device=device)


    torch.nn.init.uniform_(features, -1.0, 1.0)
    voxel_size = VoxelSize(*[dim_size / grid_dim for dim_size, grid_dim
                             in zip(config.grid_world_size, config.grid_dims)])
    voxel_grid = VoxelArtGrid(
        densities=densities,
        features=features,
        voxel_size=voxel_size,
        grid_location=VoxelGridLocation(*config.grid_location),
        **vox_grid_density_activations_dict,
        tunable=True,
        palette=palette,
        num_colors=num_colors,
        use_pure_argmax=config.use_pure_argmax,
        quantize_colors=config.quantize_colors,
        palette_learning_mode=config.palette_learning_mode,
        va_structure_mode=config.va_structure_mode,
    )
    # fmt: on

    # set up a volumetricModel using the previously created voxel-grid
    # noinspection PyTypeChecker
    if config.quantize_colors:
        radiance_hdr_tone_map = torch.nn.Identity()
    else:
        radiance_hdr_tone_map = torch.sigmoid

    vox_grid_vol_mod = VolumetricModel(
        thre3d_repr=voxel_grid,
        render_procedure=render_sh_voxelArt_grid,
        render_config=SHVoxGridRenderConfig(
            num_samples_per_ray=config.train_num_samples_per_ray,
            camera_bounds=train_dataset.camera_bounds,
            white_bkgd=config.white_bkgd,
            render_num_samples_per_ray=config.render_num_samples_per_ray,
            parallel_rays_chunk_size=config.parallel_rays_chunk_size,
            radiance_hdr_tone_map=radiance_hdr_tone_map
        ),
        device=device,
    )

    # train the model:
    train_sh_vox_grid_vol_mod_with_posed_images(
        vol_mod=vox_grid_vol_mod,
        train_dataset=train_dataset,
        output_dir=output_path,
        test_dataset=test_dataset,
        ray_batch_size=config.ray_batch_size,
        num_stages=config.num_stages,
        num_iterations_per_stage=config.num_iterations_per_stage,
        scale_factor=config.scale_factor,
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
        va_no_cnn_mode=True, # this mode tells it that is a voxelArt grid with no cnn
        temperature_gamma=config.temperature_gamma,
        raise_temperature_iter=config.raise_temperature_iter,
        sa_init_weight=config.sa_init_weight,
        sa_gamma=config.sa_gamma,
        sa_interval=config.sa_interval,
        sa_percentile=config.sa_percentile,
        sa_start_iter=config.sa_start_iter,
        clip_model=clip_model,
        semantic_weight=config.semantic_weight,
        accumulation_iters=config.accumulation_iters,
        prompt=config.prompt,
        start_semantic_iter=config.start_semantic_iter,
        directional_weight=config.directional_weight,
        palette_learning_mode=config.palette_learning_mode,
        min_distance_weight=config.min_distance_weight,
        warped_dataset=warped_dataset,
        quantize_colors=config.quantize_colors,
        start_sdl_iter=config.start_sdl_iter,
        va_structure_mode=config.va_structure_mode,
    )


if __name__ == "__main__":
    main()
