from cffi import FFI

ffibuilder = FFI()

ffibuilder.cdef(
    """
    double smooth_mandelbulb_eval(
        double x, double y, double z,
        double cx, double cy, double cz,
        int exponent, int max_iterations
    );
    void smooth_mandelbulb(
        double *x, double *y, double *z,
        double *cx, double *cy, double *cz,
        double *out,
        int exponent, int max_iterations, int num_samples
    );
    """
)

ffibuilder.set_source(
    "_routines",
    """

    double EPSILON = 1e-12;

    double smooth_mandelbulb_eval(
        double x, double y, double z,
        double cx, double cy, double cz,
        int exponent, int max_iterations
    ) {
        int i;
        for (i = 0; i < max_iterations; ++i) {
            double l = x*x + y*y;
            double r = l + z*z;
            if (r > 256) {
                return (log(log(r)*0.5) - 1.0197814405382262)/log(exponent)- i + max_iterations + 1;
            }

            double t = sqrt(l);
            r = 1.0 / (t + (t < EPSILON));
            x *= r;
            y *= r;

            if (exponent == 2 || exponent == 4 || exponent == 8) {
                l -= z*z;
                z *= 2*t;

                t = x;
                x = x*x - y*y;
                y *= 2*t;
            }
            if (exponent == 3) {
                l = t*(l - 3*z*z);
                z = z*(3*t*t - z*z);

                t = x;
                x = x*(x*x - 3*y*y);
                y = y*(3*t*t - y*y);
            }
            if (exponent == 4 || exponent == 8) {
                t = l;
                l = l*l - z*z;
                z *= 2*t;

                t = x;
                x = x*x - y*y;
                y *= 2*t;
            }
            if (exponent == 8) {
                t = l;
                l = l*l - z*z;
                z *= 2*t;

                t = x;
                x = x*x - y*y;
                y *= 2*t;
            }

            x = x*l + cx;
            y = y*l + cy;
            z += cz;
        }
        return 0;
    }

    void smooth_mandelbulb(
        double *x, double *y, double *z,
        double *cx, double *cy, double *cz,
        double *out,
        int exponent, int max_iterations, int num_samples
    ) {
        for (int i = 0; i < num_samples; ++i) {
            out[i] = smooth_mandelbulb_eval(
                x[i], y[i], z[i],
                cx[i], cy[i], cz[i],
                exponent, max_iterations
            );
        }
    }
    """
)

if __name__ == '__main__':
    ffibuilder.compile(verbose=True)
