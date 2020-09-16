from __future__ import division
from pylab import *
from mandelbulb import mandelbulb, pow3d
from shapes import tetrahedron, cube, merkaba
from util import generate_mesh_slices, threaded_anti_alias
from density import illuminate_and_absorb
import numpy as np
from threading import Thread, Lock


def make_picture_frame(rgb, dither=1.0/256.0):
    if dither:
        rgb = [channel + random(channel.shape)*dither for channel in rgb]
    frame = stack(rgb, axis=-1)
    frame = clip(frame, 0.0, 1.0)
    return frame


if __name__ == '__main__':
    scale = 3
    u_samples = 2**14
    theta = 0.5
    phi = 0.6
    gamma = 0.2
    zoom = -0.65
    anti_aliasing = 4

    width = 108*scale
    height = 108*scale
    depth = u_samples
    du = 1.0 / u_samples

    def source(x, y, z):
        cx = x + 0
        cy = y + 0
        cz = z + 0

        result = x*0 + 1e100

        for _ in range(32):
            x, y, z = pow3d(x, y, z, 8, 8, -8)
            x += cx
            y += cy
            z += cz

            result = minimum(x*x + y*y + z*z, result)

        field = result - 0.7
        core = field < 0
        illumination = exp(-field*3)
        illumination = array([illumination, illumination*0.5, illumination*0.2])*50
        illumination[:, core] = 0
        absorption = array([core, 2*core, core])*60
        return illumination, absorption

    def generate_subpixel_image(offset_x, offset_y):
        slices = generate_mesh_slices(width, height, depth, 0, 0, 0, zoom, theta, phi, gamma, offset_x, offset_y)
        image = illuminate_and_absorb(slices, source, array([0.5, 0.35, 0.55]), du)
        return image

    image = threaded_anti_alias(generate_subpixel_image, width, height, anti_aliasing)

    imsave("/tmp/out.png", make_picture_frame(image))
