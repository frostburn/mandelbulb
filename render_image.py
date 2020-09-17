from __future__ import division
from pylab import *
from mandelbulb import mandelbulb, pow3d, biaxial_julia
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
    scale = 10
    u_samples = 2**10
    theta = 0.5
    phi = 0.6
    gamma = 0.2
    zoom = -0.8
    max_iter = 32
    anti_aliasing = 4

    width = 108*scale
    height = 108*scale
    depth = u_samples
    du = 1.0 / u_samples

    def source(x, y, z):
        field = biaxial_julia(x, y, z, 0.67, -0.421, 0.515, -0.41, 5, 5, max_iter)
        core = field < 0
        field[core] = 0

        illumination = exp(-0.09*field)
        illumination = array([illumination, illumination*0.5, illumination*0.2]) * 80
        illumination[:, core] = 0
        absorption = array([core, 2*core, core]) * 100
        return illumination, absorption

    def generate_subpixel_image(offset_x, offset_y):
        slices = generate_mesh_slices(width, height, depth, 0, 0, 0, zoom, theta, phi, gamma, offset_x, offset_y)
        image = illuminate_and_absorb(slices, source, array([0.5, 0.35, 0.55]), du)
        return image

    image = threaded_anti_alias(generate_subpixel_image, width, height, anti_aliasing)

    imsave("/tmp/out.png", make_picture_frame(image))
