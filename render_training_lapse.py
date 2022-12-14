import cv2
import glob
import numpy as np
import click
from pathlib import Path
from easydict import EasyDict
from natsort import natsorted
import os

# -------------------------------------------------------------------------------------
#  Command line configuration for the script                                          |
# -------------------------------------------------------------------------------------
# fmt: off
# noinspection PyUnresolvedReferences
@click.command()
# Required arguments:
@click.option("-i", "--log_path", type=click.Path(file_okay=False, dir_okay=True),
              required=True, help="path training output logs")
@click.option("-o", "--output_path", type=click.Path(file_okay=True, dir_okay=False),
              required=True, help="path for saving output lapse")


# fmt: on
# -------------------------------------------------------------------------------------
def main(**kwargs) -> None:
    # load the requested configuration for the training
    config = EasyDict(kwargs)

    # parse os-checked path-strings into Pathlike Paths :)
    log_path = Path(config.log_path)
    output_path = Path(config.output_path)

    frameSize = (400, 400)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(str(output_path), fourcc, 2.0, frameSize)

    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # org
    org = (50, 50)
    
    # fontScale
    fontScale = 1
    
    # Blue color in BGR
    color = (0, 0, 0)
    
    # Line thickness of 2 px
    thickness = 2

    for filename in natsorted(glob.glob(str(log_path / "voxelized_*.png"))):
        img = cv2.imread(filename)
        text = os.path.basename(os.path.normpath(filename))
        img = cv2.putText(img, text, org, font, 
                       fontScale, color, thickness, cv2.LINE_AA)
        out.write(img)
    
    print("Finished rendering training lapse!")
    out.release()

if __name__ == "__main__":
    main()