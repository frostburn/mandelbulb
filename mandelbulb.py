import numpy as np
from _routines import ffi, lib

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
