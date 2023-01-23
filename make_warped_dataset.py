from get_palette import quantize_images
from easydict import EasyDict
import numpy as np
import os
import click

# -------------------------------------------------------------------------------------
#  Command line configuration for the script                                          |
# -------------------------------------------------------------------------------------
# fmt: off
# noinspection PyUnresolvedReferences
@click.command()

# Required arguments:
@click.option("-d", "--data_path", type=click.Path(file_okay=False, dir_okay=True),
              required=True, help="path to the input dataset")

# fmt: on
# -------------------------------------------------------------------------------------

def main(**kwargs) -> None:
    config = EasyDict(kwargs)
    with open(os.path.join(config.data_path, 'palette', 'palette.npy'), 'rb') as f:
        palette = np.load(f)
    quantize_images(os.path.join(config.data_path, 'train'), palette)

if __name__ == "__main__":
    main()