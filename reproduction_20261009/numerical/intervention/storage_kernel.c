#include <math.h>
#include <stddef.h>

/* Exact accelerated form of build_wp6_pv_storage.simulate_storage. */
void dispatch(const double *pv, size_t n, double ratio, double hours,
              double rte, size_t warmup, double *residual) {
    double soc = 0.0;
    const double eta = sqrt(rte);
    for (size_t t = 0; t < n + warmup; ++t) {
        size_t index = t < warmup ? n - warmup + t : t - warmup;
        double resource = fmax(0.0, pv[index]) * ratio;
        double direct = fmin(1.0, resource);
        double deficit = 1.0 - direct;
        double surplus = fmax(0.0, resource - 1.0);
        double discharged = 0.0;
        if (hours > 0.0) {
            double charge = fmin(surplus, fmin(1.0, fmax(0.0, hours - soc) / eta));
            soc += charge * eta;
            discharged = fmin(deficit, fmin(1.0, soc * eta));
            soc -= discharged / eta;
        }
        if (t >= warmup) residual[t - warmup] = fmax(0.0, deficit - discharged);
    }
}
