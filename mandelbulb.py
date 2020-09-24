import numpy as np
from _routines import ffi, lib
from util import bufferize
from threading import Thread


def pow3d(x, y, z, exponent_theta, exponent_phi, exponent_r, shape=None):
    if shape is None:
        for w in (x, y, z):
            if hasattr(w, 'shape'):
                if shape is None:
                    shape = w.shape
                if shape == ():
                    shape = w.shape
    # Upshape and make unique
    w = np.zeros(shape)
    x = x + w
    y = y + w
    z = z + w

    x_buf = ffi.cast("double*", x.ctypes.data)
    y_buf = ffi.cast("double*", y.ctypes.data)
    z_buf = ffi.cast("double*", z.ctypes.data)

    lib.pow3d(
        x_buf, y_buf, z_buf,
        exponent_theta, exponent_phi, exponent_r,
        x.size
    )
    return x, y, z


def pow_quaternion_inplace(x, y, z, exponent):
    x_buf = ffi.cast("double*", x.ctypes.data)
    y_buf = ffi.cast("double*", y.ctypes.data)
    z_buf = ffi.cast("double*", z.ctypes.data)

    lib.pow_quaternion(
        x_buf, y_buf, z_buf,
        exponent, x.size
    )

    return x, y, z


def mandelbulb(x, y, z, cx, cy, cz, exponent_theta, exponent_phi, exponent_r, max_iter, shape=None):
    if shape is None:
        for w in (x, y, z, cx, cy, cz):
            if hasattr(w, 'shape'):
                if shape is None:
                    shape = w.shape
                if shape == ():
                    shape = w.shape
    # Upshape and make unique
    w = np.zeros(shape)
    x = x + w
    y = y + w
    z = z + w
    cx = cx + w
    cy = cy + w
    cz = cz + w

    result = w
    x_buf = ffi.cast("double*", x.ctypes.data)
    y_buf = ffi.cast("double*", y.ctypes.data)
    z_buf = ffi.cast("double*", z.ctypes.data)
    cx_buf = ffi.cast("double*", cx.ctypes.data)
    cy_buf = ffi.cast("double*", cy.ctypes.data)
    cz_buf = ffi.cast("double*", cz.ctypes.data)
    result_buf = ffi.cast("double*", result.ctypes.data)

    lib.smooth_mandelbulb(
        x_buf, y_buf, z_buf,
        cx_buf, cy_buf, cz_buf,
        result_buf, exponent_theta, exponent_phi, exponent_r, max_iter,
        result.size
    )

    return result


def biaxial_julia(x, y, z, c0x, c0y, c1y, c1z, exponent0, exponent1, max_iter, shape=None):
    refs, bufs = bufferize(x, y, z, c0x, c0y, c1y, c1z)
    result = np.zeros(refs[0].shape)
    result_buf = ffi.cast("double*", result.ctypes.data)
    args = bufs + [result_buf, exponent0, exponent1, max_iter, result.size]
    lib.biaxial_julia(*args)
    return result


def buddhabulb_exposure(width, height, depth, center_x, center_y, center_z, zoom, rotation_theta, rotation_phi, rotation_gamma, exponent_theta, exponent_phi, exponent_r, num_samples, min_iter, max_iter, generator, num_threads=16, bailout=16, chunk_size=8192, julia=None):
    num_color_channels = 3
    result = np.zeros((depth, height, width), dtype="uint64")
    result_buf = ffi.cast("unsigned long long*", result.ctypes.data)

    num_samples = int(round(num_samples / num_threads))

    ct, st = np.cos(rotation_theta), np.sin(rotation_theta)
    cp, sp = np.cos(rotation_phi), np.sin(rotation_phi)
    cg, sg = np.cos(rotation_gamma), np.sin(rotation_gamma)
    dx = np.identity(3, dtype="double")
    dx = np.matmul(dx, [[ct, 0, st], [0, 1, 0], [-st, 0, ct]])
    dx = np.matmul(dx, [[cp, sp, 0], [-sp, cp, 0], [0, 0, 1]])
    dx = np.matmul(dx, [[1, 0, 0], [0, cg, sg], [0, -sg, cg]])
    dx[0] *= 2**zoom * height
    dx[1] *= 2**zoom * height
    dx[2] *= 2**zoom * depth
    dx_buf = ffi.cast("double*", dx.ctypes.data)

    offset = np.matmul(dx, [center_x, center_y, center_z])
    x0 = width*0.5 - offset[0]
    y0 = height*0.5 - offset[1]
    z0 = depth*0.5 - offset[2]
    def accumulate_samples():
        remaining = num_samples
        while remaining:
            chunk = min(remaining, chunk_size)
            x_samples, y_samples, z_samples = generator(chunk)
            x_samples_buf = ffi.cast("double*", x_samples.ctypes.data)
            y_samples_buf = ffi.cast("double*", y_samples.ctypes.data)
            z_samples_buf = ffi.cast("double*", z_samples.ctypes.data)
            if julia is not None:
                cx_samples = 0*x_samples + julia[0]
                cy_samples = 0*y_samples + julia[1]
                cz_samples = 0*z_samples + julia[2]
            else:
                cx_samples = x_samples
                cy_samples = y_samples
                cz_samples = z_samples
            cx_samples_buf = ffi.cast("double*", cx_samples.ctypes.data)
            cy_samples_buf = ffi.cast("double*", cy_samples.ctypes.data)
            cz_samples_buf = ffi.cast("double*", cz_samples.ctypes.data)
            lib.buddhabulb(x_samples_buf, y_samples_buf, z_samples_buf, cx_samples_buf, cy_samples_buf, cz_samples_buf, chunk, result_buf, width, height, depth, x0, y0, z0, dx_buf, exponent_theta, exponent_phi, exponent_r, max_iter, min_iter, bailout)
            remaining -= chunk

    ts = []
    for _ in range(num_threads):
        ts.append(Thread(target=accumulate_samples))
        ts[-1].start()
    for t in ts:
        t.join()

    return result

def buddhabulb(width, height, depth, center_x, center_y, center_z, zoom, rotation_theta, rotation_phi, rotation_gamma, exponent_theta, exponent_phi, exponent_r, num_samples, exposures, color_map, anti_aliasing=2, num_threads=16, bailout=16, chunk_size=32768, julia=None):
    exposed = []
    for min_iter, max_iter, generator in exposures:
        exposed.append(buddhabulb_exposure(
            width*anti_aliasing, height*anti_aliasing, depth, center_x, center_y, center_z, zoom, rotation_theta, rotation_phi, rotation_gamma, exponent_theta, exponent_phi, exponent_r, num_samples,
            min_iter, max_iter, generator, num_threads, bailout, chunk_size, julia))
    result = color_map(exposed)

    image = result[:, ::anti_aliasing, ::anti_aliasing]*0
    for i in range(anti_aliasing):
        for j in range(anti_aliasing):
            image += result[:, i::anti_aliasing, j::anti_aliasing]
    image /= anti_aliasing**2

    return image
