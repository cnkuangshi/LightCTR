//
//  significance.h
//  LightCTR
//
//  Created by SongKuangshi on 2018/5/4.
//  Copyright © 2018年 SongKuangshi. All rights reserved.
//

#ifndef significance_h
#define significance_h

#include <cmath>

// error function
inline double Erf(double x) {
    // handle either positive or negative x. because error function is negatively symmetric of x
    double a = 0.140012;
    double b = x * x;
    double item = -b * (4 / M_PI + a * b) / (1 + a * b);
    double result = sqrt(1 - exp(item));
    if (x >= 0)
        return result;
    return -result;
}

inline double LogCDF(double x, double alpha = 10) {
    const double scaler = (alpha == 10) ? 1 : log(alpha);
    return (x * log(fabs(x)) - x) / scaler;
}

// calculate the standard cumulative distribution function F(x) = P(Z less or equal than x),
// where Z follows a standard normal distribution.
inline double StandardCDF(double x) {
    const double SquareRootOfTwo = 1.414213562373095;
    return (1.0 + Erf(x / SquareRootOfTwo)) / 2;
}

inline double CustomCDF(double x, double u, double sigma) {
    x = x - u;
    return 0.5 + 0.5 * Erf(x / sigma / 1.414213562373095);
}

inline double ReverseCDF(double p, double mu, double sigma) {
    double lower = -5.0, upper = 5.0, middle;
    while(1) {
        middle = (lower + upper) / 2;
        double estimate = CustomCDF(middle, mu, sigma);
        if (fabs(estimate - p) < 1e-7)
            break;
        // because standard CDF is monotonic, thus we use binary search
        if (estimate > p) {
            upper = middle;
        } else {
            lower = middle;
        }
    }
    return middle;
}

// given a confidence level we calculate the Z such that P(Z greater than alpha) = alpha
inline double ReverseAlpha(double alpha) {
    assert(alpha > 0 && alpha < 1);
    return ReverseCDF(1.0f - alpha, 0, 1);
}

// calculate the statistical significance for a gaussian distribution
// the observed x value, its mean value and standard deviation
inline double GaussianSignificance(double x, double u, double sigma) {
    double cdf = CustomCDF(x, u, sigma);
    return 2 * cdf - 1;
}

#endif /* significance_h */
