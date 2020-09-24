from __future__ import division
from pylab import *
from mandelbulb import mandelbulb, pow3d, biaxial_julia, pow_quaternion_inplace, buddhabulb
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
    depth = 1<<8
    theta = 0.5
    phi = 0.2
    gamma = 0.1
    zoom = 0
    anti_aliasing = 1

    width = 108*scale
    height = 108*scale
    num_samples = 1 << 32
    du = 1.0 / depth

    sample_scale = (scale*anti_aliasing)**2 * (num_samples * 2**-20)**-1

    def cgen(cx, cy, cz, w):
        def gen(n):
            x = randn(n)*w + cx
            y = randn(n)*w + cy
            z = randn(n)*w + cz
            return x, y, z
        return gen

    w = 0.2
    exposures = [(9, 10, cgen(0.5, 0.92, 0.15, w)), (10, 11, cgen(0.5, 0.95, 0.2, w)), (11, 12, cgen(0.5, 0.9, 0.1, w))]

    def color_map(exposures):
        e = exposures[0][0]*0.0
        result = array([e, e, e])
        result[0] = 0.04
        result[1] = 0.05
        result[2] = 0.045
        for k in range(depth):
            a = exposures[0][k] * sample_scale
            b = exposures[1][k] * sample_scale
            c = exposures[2][k] * sample_scale
            illumination = array([0.9*a*a, 0.5*a**1.5, 0.4*a])*0.7
            illumination += array([0.2*b, 0.6*b, 0.2*sqrt(b)])
            illumination += array([0.3*c, 0.7*sqrt(c), 0.8*c**0.75])
            result += 3*illumination*du
            absorption = array([a*0.8, a*0.3, a*0.4])
            absorption += array([b*0.7, b*0.8, b*0.2])
            absorption += array([c*0.5, c*0.4, c*0.9])
            result *= exp(-2*absorption*du)
        return result

    image = buddhabulb(width, height, depth, 0.5, 0.7, 0, zoom, theta, phi, gamma, 8, 8, 8, num_samples, exposures, color_map, anti_aliasing=anti_aliasing)

    imsave("/tmp/buddhabulb.png", make_picture_frame(image))
