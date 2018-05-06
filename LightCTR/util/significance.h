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

// calculate the standard cumulative distribution function F(x) = P(Z less or equal than x),
// where Z follows a standard normal distribution.
inline double StandardCDF(double x) {
    const double SquareRootOfTwo = 1.414213562373095;
    return (1.0 + Erf(x / SquareRootOfTwo)) / 2;
}

// given a confidence level we calculate the Z such that P(Z greater than alpha) = alpha
inline double ReverseAlpha(double alpha) {
    double p = 1.0 - alpha;
    // for a standard normal distribution, the probability that x is smaller than lower or x is larger than upper is almost zero.
    // we can set a larger value but already has no gain.
    double lower = -5.0, upper = 5.0, middle;
    while(1) {
        middle = (lower + upper) / 2;
        double estimate = StandardCDF(middle);
        if (abs(estimate - p) < 0.00000001)
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

// calculate the statistical significance for a gaussian distribution
// the observed x value, its mean value and standard deviation
inline double GaussianSignificance(double x, double u, double sigma) {
    double x1 = abs(x - u);
    // 1.414213562373095 is sqrt(2)
    double cdf = 0.5 + 0.5 * Erf(x1 / sigma / 1.414213562373095);
    return 2 * cdf - 1;
}

#endif /* significance_h */
