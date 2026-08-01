/**
 * @file      chi2.hpp
 * @brief     Chi-squared distribution functions
 * @author    @vojtagart
 * @date      17/02/2026
 * @see       https://github.com/Vojtagart/mtt
 * * @par Credits & Acknowledgments
 * - Documentation formatted and refined with the assistance of Google Gemini
 * * @copyright Copyright (c) 2026 @vojtagart. Released under the MIT License.
 */

#pragma once
#include <cmath>
#include <cassert>
#include <numbers>
#include <limits>
#include <type_traits>


namespace mtt {

/**
 * @brief Computes the Probability Density Function (PDF) of the Chi-squared distribution
 * 
 * @param x The value at which to evaluate the PDF
 * @param k The degrees of freedom
 * 
 * @return The probability density
 */
template <typename Scalar>
requires (std::is_floating_point_v<Scalar>)
[[nodiscard]] Scalar chi2_pdf(Scalar x, Scalar k) {
    assert(k > 0);
    if (x <= 0) return Scalar{0};
    Scalar half_k = k / 2;
    // ln(f(x)) = (k/2 - 1)*ln(x) - x/2 - (k/2)*ln(2) - ln(gamma(k/2))
    Scalar logpdf = (half_k - 1) * std::log(x) - x / 2 - half_k * std::numbers::ln2_v<Scalar> - std::lgamma(half_k);
    return std::exp(logpdf);
}

/**
 * @brief Computes the Inverse Cumulative Distribution Function
 * 
 * @note Currently restricted to k=2
 * 
 * @param p The probability value [0, 1]
 * @param k The degrees of freedom
 * 
 * @return The value x such that P(X <= x) = p
 */
template <typename Scalar>
requires (std::is_floating_point_v<Scalar>)
[[nodiscard]] Scalar chi2_inv(Scalar p, Scalar k) {
    assert(k == 2 && "Only k = 2 is supported at this moment");
    assert(0 <= p && p <= 1 && "Invalid value for chi2inv");
    (void)k;
    if (p >= 1) return std::numeric_limits<Scalar>::max();
    // Using analytic solution for k = 2
    // x = -2 * ln(1 - p)
    return -Scalar{2} * std::log1p(-p);
}

} // namesapce mtt