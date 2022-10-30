from pathlib import Path
from PIL import Image as im
import click
import imageio
import torch

from thre3d_atom.modules.volumetric_model import (
    create_volumetric_model_from_saved_model,
)
from thre3d_atom.thre3d_reprs.voxels import (create_voxel_grid_from_saved_info_dict, 
    scale_voxel_grid_with_required_output_size
)
from thre3d_atom.utils.constants import HEMISPHERICAL_RADIUS, CAMERA_INTRINSICS
from thre3d_atom.utils.imaging_utils import (
    get_thre360_animation_poses,
    get_thre360_spiral_animation_poses,
)
from thre3d_atom.visualizations.animations import (
    render_camera_path_for_volumetric_model_3_coeff_modes,
    render_camera_path_for_volumetric_model_3_coeff_modes_gray,
)
from easydict import EasyDict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------------------------------------------------------------
#  Command line configuration for the script                                          |
# -------------------------------------------------------------------------------------
# fmt: off
# noinspection PyUnresolvedReferences
@click.command()
# Required arguments:
@click.option("-i", "--model_path", type=click.Path(file_okay=True, dir_okay=False),
              required=True, help="path to the trained (reconstructed) model")
@click.option("-o", "--output_path", type=click.Path(file_okay=False, dir_okay=True),
              required=True, help="path for saving rendered output")

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
@click.option("--num_frames", type=click.IntRange(min=1), default=180,
              required=False, help="number of frames in the video")
# spiral path options
@click.option("--vertical_camera_height", type=click.FLOAT, default=3.0,
              required=False, help="height at which the camera spiralling will happen")
@click.option("--num_spiral_rounds", type=click.IntRange(min=1), default=2,
              required=False, help="number of rounds made while transitioning between spiral radii")

# Non-required video options:
@click.option("--fps", type=click.IntRange(min=1), default=50,
              required=False, help="frames per second of the video")

# ES ADDITIONS:
@click.option("--clusters", type=click.IntRange(min=0), default=0,
              required=False, help="Number of clusters for SH coeff Quantization")
@click.option("--gray_mode", type=click.BOOL, default=False,
              required=False, help="Make zero coeffs gray to view the effects of higher order coeffs")

# fmt: on
# -------------------------------------------------------------------------------------
def main(**kwargs) -> None:
    # load the requested configuration for the training
    config = EasyDict(kwargs)

    # parse os-checked path-strings into Pathlike Paths :)
    model_path = Path(config.model_path)
    output_path = Path(config.output_path)

    # create the output path if it doesn't exist
    output_path.mkdir(exist_ok=True, parents=True)

    # load volumetric_model from the model_path
    vol_mod, extra_info = create_volumetric_model_from_saved_model(
        model_path=model_path,
        thre3d_repr_creator=create_voxel_grid_from_saved_info_dict,
        device=device,
        num_clusters=config.clusters,
    )

    # 1. Get Maximal Red feature
    all_features = torch.reshape(vol_mod.thre3d_repr.features.data, (-1, 27))
    red_coeffs = vol_mod.thre3d_repr.features.data[..., 0]
    red_coeffs = torch.reshape(red_coeffs, (-1, 1))
    all_densities = torch.flatten(vol_mod.thre3d_repr.densities.data)
    idx = torch.argmax(all_densities)
    selected_feature = all_features[idx]
    all_densities = torch.flatten(vol_mod.thre3d_repr.densities.data)
    print(f"reddest feature density: {all_densities[idx]}")
    print(f"Selected feature: {selected_feature}")

    # ES Addition: initial experiment - naive trilinear downsampling
    vol_mod.thre3d_repr = scale_voxel_grid_with_required_output_size(
                    vol_mod.thre3d_repr,
                    output_size=tuple([3, 3, 3]),
                    mode="trilinear",
                )
    vol_mod.thre3d_repr.interpolation_mode = 'nearest'
    
    # 2. Make only middle voxel visible
    vol_mod.thre3d_repr.densities.data[:] = 0
    vol_mod.thre3d_repr.densities.data[1, 1, 1] = 100
    vol_mod.thre3d_repr.features.data[1, 1, 1] = selected_feature

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

    if config.gray_mode:
        render_function = render_camera_path_for_volumetric_model_3_coeff_modes_gray
    else:
        render_function = render_camera_path_for_volumetric_model_3_coeff_modes

    animation_frames = render_function(
        vol_mod=vol_mod,
        camera_path=animation_poses,
        camera_intrinsics=camera_intrinsics,
        overridden_num_samples_per_ray=config.overridden_num_samples_per_ray,
        render_scale_factor=config.render_scale_factor,
    )

    imageio.mimwrite(
        output_path / "rendered_video.mp4",
        animation_frames,
        fps=config.fps,
    )


if __name__ == "__main__":
    main()
