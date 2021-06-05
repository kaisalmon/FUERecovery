import glob
import cv2
import numpy as np
from process_image import process_image, scale_image
import imageio

fps = 7
skip_first_n = 0
imageDir = "./images/day*.*"
images = glob.glob(imageDir)[skip_first_n:]

if True:
    with imageio.get_writer("output/input.gif", mode="I", fps=fps) as writer:
        idx = skip_first_n
        for path in images:
            frame_length = 1
            if idx == 0 or idx == len(images) - 1:
                frame_length = 4
            image = scale_image(cv2.imread(path), 50)
            for i in range(0, frame_length):
                writer.append_data(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            idx += 1

with imageio.get_writer("output/output.gif", mode="I", fps=fps) as writer:
    idx = skip_first_n
    for path in images:
        print(path)
        frame_length = 1
        if idx == skip_first_n or idx == len(images) + skip_first_n - 1:
            frame_length = 4
        image = process_image(path, idx + 1)
        cv2.imwrite(path.replace("images", "output").replace('jpg', 'png'), image)
        for i in range(0, frame_length):
            writer.append_data(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        idx += 1
