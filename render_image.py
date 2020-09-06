from pylab import *
from mandelbulb import mandelbulb
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
    u_samples = 2**12
    theta = 0.75
    phi = 0.5
    anti_aliasing = 4
    exponent = 3
    max_iter = 40

    grid_x = linspace(-1.2, 1.2, scale*108)
    grid_y = linspace(-1.2, 1.1, scale*108)
    u = linspace(-1.5, 1.5, u_samples)
    dx = grid_x[1] - grid_x[0]
    dy = grid_y[1] - grid_y[0]
    du = u[1] - u[0]

    grid_x, grid_y = meshgrid(grid_x, grid_y)

    result = array([0*grid_x, 0*grid_x, 0*grid_x])

    lock = Lock()

    def accumulate_subpixels(offset_x, offset_y):
        global result
        x = grid_x + offset_x
        y = grid_y + offset_y

        red = ones(x.shape) * 4.0
        green = ones(x.shape) * 3.2
        blue = ones(x.shape) * 2.0
        for z in u:
            x_ = cos(theta) * x + sin(theta) * z
            z_ = cos(theta) * z - sin(theta) * x
            y_ = cos(phi) * y + sin(phi) * z_
            z_ = cos(phi) * z_ - sin(phi) * y
            val = mandelbulb(x_, y_, z_, x_, y_, z_, exponent, max_iter)
            core = (val == 0)
            absorption = 5*exp(-0.125*val)
            absorption[core] = 5
            red += 40*core * du
            red *= exp(-du * absorption)

            absorption = 10*exp(-0.13*val)
            absorption[core] = 4.2
            green += 40*core * du
            green *= exp(-du * absorption)

            absorption = 10*exp(-0.135*val)
            absorption[core] = 2.5
            blue += 20*core * du
            blue *= exp(-du * absorption)

        lock.acquire()
        result += array([red, green, blue])*0.1
        lock.release()


    ts = []
    offsets_x = np.arange(anti_aliasing) / anti_aliasing * dx
    offsets_y = np.arange(anti_aliasing) / anti_aliasing * dy
    for i in offsets_x:
        for j in offsets_y:
            ts.append(Thread(target=accumulate_subpixels, args=(i, j)))
            ts[-1].start()
    for t in ts:
        t.join()

    result /= anti_aliasing**2

    image = result
    imsave("/tmp/out.png", make_picture_frame(image))
