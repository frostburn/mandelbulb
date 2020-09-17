from threading import Thread, Lock
import numpy as np
from _routines import ffi


def generate_mesh_slices(width, height, depth, center_x, center_y, center_z, zoom, rotation_theta, rotation_phi, rotation_gamma, offset_x, offset_y, depth_dither=1):
    ct, st = np.cos(rotation_theta), np.sin(rotation_phi)
    cp, sp = np.cos(rotation_phi), np.sin(rotation_phi)
    cg, sg = np.cos(rotation_gamma), np.sin(rotation_gamma)
    zoom = 2**-zoom / height

    x = np.arange(width, dtype='float64') + offset_x
    y = np.arange(height, dtype='float64') + offset_y
    z = np.arange(depth, dtype='float64')

    x, y = np.meshgrid(x, y)

    x = (2 * x - width) * zoom
    y = (2 * y - height) * zoom

    for z_ in z:
        z_ += np.random.rand(*x.shape) * depth_dither - 0.5*depth_dither
        z_ = (2 * z_ - depth) * zoom
        x_, z_ = x*ct + z_*st, z_*ct - x*st
        x_, y_ = x_*cp + y*sp, y*cp - x_*sp
        y_, z_ = y_*cg + z_*sg, z_*cg - y_*sg

        x_ += center_x
        y_ += center_y
        z_ += center_z

        yield x_, y_, z_


def threaded_anti_alias(generate_subpixel_image, width, height, anti_aliasing, num_channels=3):
    lock = Lock()

    result = np.zeros((num_channels, height, width))

    def accumulate_subpixels(offset_x, offset_y):
        nonlocal result

        subpixel_image = generate_subpixel_image(offset_x, offset_y)

        lock.acquire()
        result += subpixel_image
        lock.release()

    ts = []
    offsets = np.arange(anti_aliasing) / anti_aliasing
    for i in offsets:
        for j in offsets:
            ts.append(Thread(target=accumulate_subpixels, args=(i, j)))
            ts[-1].start()
    for t in ts:
        t.join()

    result /= anti_aliasing**2

    return result


def bufferize(*args, shape=None):
    if shape is None:
        for w in args:
            if hasattr(w, 'shape'):
                if shape is None:
                    shape = w.shape
                if shape == ():
                    shape = w.shape
    # Upshape and make unique
    w = np.zeros(shape)

    refs = []
    results = []
    for z in args:
        z = z + w
        z_buffer = ffi.cast("double*", z.ctypes.data)
        refs.append(z)
        results.append(z_buffer)

    return refs, results
