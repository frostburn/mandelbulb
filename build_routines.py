from cffi import FFI

ffibuilder = FFI()

ffibuilder.cdef(
    """
    void pow3d(double *x, double *y, double *z, int exponent_theta, int exponent_phi, int exponent_r, int num_samples);

    void smooth_mandelbulb(
        double *x, double *y, double *z,
        double *cx, double *cy, double *cz,
        double *out,
        int exponent_theta, int exponent_phi, int exponent_r, int max_iterations, int num_samples
    );

    void biaxial_julia(
        double *x, double *y, double *z,
        double *c0x, double *c0y,
        double *c1y, double *c1z,
        double *out,
        int exponent0, int exponent1, int max_iterations, int num_samples
    );
    """
)

ffibuilder.set_source(
    "_routines",
    """

    double EPSILON = 1e-12;

    inline double r_pow(double r, int exponent) {
        double result = 1;
        while (exponent > 0) {
            if (exponent & 1) {
                result *= r;
            }
            r *= r;
            exponent >>= 1;
        }
        return result;
    }

    inline void c_pow(double *x, double *y, int exponent) {
        double cx = *x;
        double cy = *y;
        double t;
        *x = 1;
        *y = 0;
        while (exponent > 0) {
            if (exponent & 1) {
                t = *x;
                *x = cx*(*x) - cy*(*y);
                *y = cx*(*y) + cy*t;
            }
            t = cx;
            cx = cx*cx - cy*cy;
            cy *= 2*t;
            exponent >>= 1;
        }
    }

    void pow3d(double *x, double *y, double *z, int exponent_theta, int exponent_phi, int exponent_r, int num_samples) {
        for (int i = 0; i < num_samples; ++i) {
            double l = x[i]*x[i] + y[i]*y[i];
            double r = l + z[i]*z[i];
            l = sqrt(l);
            double t = 1.0 / (l + (l < EPSILON));
            x[i] *= t;
            y[i] *= t;

            r = sqrt(r);
            t = 1.0 / (r + (r < EPSILON));
            l *= t;
            z[i] *= t;

            r = r_pow(r, abs(exponent_r));
            c_pow(x + i, y + i, abs(exponent_theta));
            c_pow(&l, z + i, abs(exponent_phi));

            if (exponent_theta < 0) {
                y[i] = -y[i];
            }
            if (exponent_phi < 0) {
                z[i] = -z[i];
            }
            if (exponent_r < 0) {
                r = 1.0/r;
            }
            l *= r;
            x[i] *= l;
            y[i] *= l;
            x[i] *= r;
        }
    }

    double smooth_mandelbulb_eval(
        double x, double y, double z,
        double cx, double cy, double cz,
        int exponent_theta, int exponent_phi, int exponent_r, int max_iterations
    ) {
        for (int i = 0; i < max_iterations; ++i) {
            double l = x*x + y*y;
            double r = l + z*z;
            if (r > 256) {
                return (log(log(r)*0.5) - 1.0197814405382262)/log(exponent_r)- i + max_iterations + 1;
            }

            l = sqrt(l);
            double t = 1.0 / (l + (l < EPSILON));
            x *= t;
            y *= t;

            r = sqrt(r);
            t = 1.0 / (r + (r < EPSILON));
            l *= t;
            z *= t;

            r = r_pow(r, exponent_r);
            c_pow(&x, &y, exponent_theta);
            c_pow(&l, &z, exponent_phi);

            l *= r;

            x = x*l + cx;
            y = y*l + cy;
            z = z*r + cz;

            t = cy;
            cy = 0.95*cy + 0.1*cz;
            cz = 0.95*cz - 0.1*t;
        }
        return 0;
    }

    void smooth_mandelbulb(
        double *x, double *y, double *z,
        double *cx, double *cy, double *cz,
        double *out,
        int exponent_theta, int exponent_phi, int exponent_r, int max_iterations, int num_samples
    ) {
        for (int i = 0; i < num_samples; ++i) {
            out[i] = smooth_mandelbulb_eval(
                x[i], y[i], z[i],
                cx[i], cy[i], cz[i],
                exponent_theta, exponent_phi, exponent_r, max_iterations
            );
        }
    }

    double biaxial_julia_eval(
        double x, double y, double z,
        double c0x, double c0y,
        double c1y, double c1z,
        int exponent0, int exponent1, int max_iterations,
        double log_exponent1,
        double log_exponent0
    ) {
        int i;
        double r = 256.0;
        for (i = 0; i < max_iterations; ++i) {
            r = x*x + y*y + z*z;
            if (r > 256) {
                break;
            }
            if (i & 1) {
                c_pow(&y, &z, exponent1);
                y += c1y;
                z += c1z;
            } else {
                c_pow(&x, &y, exponent0);
                x += c0x;
                y += c0y;
            }
        }
        if (r < 256) {
            return -r;
        }
        double log_log_r = log(log(r));
        for (;i < max_iterations; ++i) {
            if (i & 1) {
                log_log_r += log_exponent1;
            } else {
                log_log_r += log_exponent0;
            }
        }
        return log_log_r - 1.7129286210981716;
    }

    void biaxial_julia(
        double *x, double *y, double *z,
        double *c0x, double *c0y,
        double *c1y, double *c1z,
        double *out,
        int exponent0, int exponent1, int max_iterations, int num_samples
    ) {
        double log_exponent0 = log(exponent0);
        double log_exponent1 = log(exponent1);
        for (int i = 0; i < num_samples; ++i) {
            out[i] = biaxial_julia_eval(
                x[i], y[i], z[i],
                c0x[i], c0y[i],
                c1y[i], c1z[i],
                exponent0, exponent1, max_iterations,
                log_exponent0, log_exponent1
            );
        }
    }
    """
)

if __name__ == '__main__':
    ffibuilder.compile(verbose=True)
