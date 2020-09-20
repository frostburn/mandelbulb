from __future__ import division
import os
import sys
sys.path.insert(0, os.path.abspath('.'))

from pylab import *
from mandelbulb import mandelbulb, pow3d, biaxial_julia, pow_quaternion_inplace
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


def quatbrot(x, y, z, cx, cy, cz, max_iter, exponent=2):
    x = x + 0
    y = y + 0
    z = z + 0
    cx = cx + 0
    cy = cy + 0
    cz = cz + 0
    for _ in range(2):
        r = y*y + z*z
        for _ in range(2):
            t = y + 0
            y = y*y - z*z
            z = 2*z*t
        y = y/r + cy
        z = z/r + cz
    escaped = -np.ones(x.shape)
    for i in range(max_iter):
        r2 = x*x + y*y + z*z
        escaped[np.logical_and(escaped < 0, r2 >= 256)] = i
        s = escaped < 0
        xs, ys, zs = pow_quaternion_inplace(x[s], y[s], z[s], exponent)
        x[s] = xs + cx[s]
        y[s] = ys + cy[s]
        z[s] = zs + cz[s]

    s = escaped > 0
    escaped[s] = np.log(np.log(x[s]**2 + y[s]**2 + z[s]**2)*0.5) / np.log(exponent) - escaped[s] + max_iter - 1 - np.log(np.log(256)*0.5) / np.log(exponent)
    escaped[~s] = 0

    return escaped


if __name__ == '__main__':
    scale = 90
    u_samples = 2**11
    theta = -0.5
    phi = 0
    gamma = 0.45
    zoom = -0.2
    max_iter = 15
    anti_aliasing = 4

    # width = 108*scale
    # height = 108*scale
    width = 12*scale
    height = 7*scale
    depth = u_samples
    du = 1.0 / u_samples

    def source(x, y, z):
        field = quatbrot(x, y, z, x, y, z, max_iter, 2)
        core = (field == 0)

        illumination = exp(-0.4*field)
        illumination = array([illumination, illumination*0.5, illumination*0.2]) * 50
        illumination[:, core] = 0
        absorption = array([core, 2*core, core]) * 30
        return illumination, absorption

    def generate_subpixel_image(offset_x, offset_y):
        slices = generate_mesh_slices(width, height, depth, -0.5, 0, 0, zoom, theta, phi, gamma, offset_x, offset_y)
        image = illuminate_and_absorb(slices, source, array([0.5, 0.35, 0.55]), du)
        return image

    image = threaded_anti_alias(generate_subpixel_image, width, height, anti_aliasing)

    imsave("/tmp/spiritual_warrior_ray.png", make_picture_frame(image))
