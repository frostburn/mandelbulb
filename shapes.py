import numpy as np

def tetrahedron(x, y, z):
    """
    Uniform distance field with tetrahedral symmetry.
    """
    u = np.maximum(x + y + z, x - y - z)
    u = np.maximum(u, y - x - z)
    return np.maximum(u, z - x - y) / np.sqrt(3)


def merkaba(x, y, z):
    return np.minimum(tetrahedron(x, y, z), tetrahedron(x, y, -z))


def cube(x, y, z):
    return np.maximum(abs(x), np.maximum(abs(y), abs(z)))


def sphere(x, y, z):
    return np.sqrt(x*x + y*y + z*z)
