import numpy as np

def illuminate_and_absorb(slices, source, background, du):
    result = None
    for s in slices:
        if result is None:
            result = np.array([s[0] * 0 + b for b in background])
        illumination, absorption = source(*s)
        result += illumination*du
        result *= np.exp(-absorption*du)
    return result
