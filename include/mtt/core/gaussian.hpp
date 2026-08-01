/**
 * @file      gaussian.hpp
 * @brief     Gaussian distribution and related functions
 * @author    @vojtagart
 * @date      17/02/2026
 * @see       https://github.com/Vojtagart/mtt
 * * @par Credits & Acknowledgments
 * - Documentation formatted and refined with the assistance of Google Gemini
 * * @copyright Copyright (c) 2026 @vojtagart. Released under the MIT License.
 */

#pragma once
#include <cmath>
#include <numbers>
#include <utility>
#include <type_traits>
#include <cassert>
#include <Eigen/Core>
#include <Eigen/Cholesky>

#include "eigen_concepts.hpp"


namespace mtt {

namespace internal {

    template <typename Scalar>
    static inline const Scalar LOG_2PI = std::log(2 * std::numbers::pi_v<Scalar>);

    /**
     * @brief Calculates n/2 * log(2*pi)
     * 
     * @tparam Scalar Value type
     * @param dim Dimension of the Gaussian (n)
     * @return Scalar n/2 * log(2*pi)
     */
    template <typename Scalar>
    [[nodiscard]] inline Scalar get_half_n_log_2pi(int dim) {
        return Scalar(0.5) * static_cast<Scalar>(dim) * LOG_2PI<Scalar>;
    }
    /**
     * @brief Compute log of normalization constant c from the Cholesky lower-triangular factor L
     *
     * Given L such that cov = L * L^T, returns: -(n/2)*log(2*pi) - 0.5*log(det(cov))
     * 
     * @param L Cholesky lower-triangular factor L of cov
     * @return log of normalization constant c
     */
    template <typename Derived>
    [[nodiscard]] typename Derived::Scalar logc_from_L(const Eigen::TriangularBase<Derived>& L) {
        using Scalar = typename Derived::Scalar;
        const auto dim = L.rows();
        assert(dim == L.cols());
        Scalar logdet = 0;
        for (Eigen::Index i = 0; i < dim; i++) {
            logdet += std::log(L.derived()(i, i));
        }
        return -internal::get_half_n_log_2pi<Scalar>(dim) - logdet;
    }
    /**
     * @brief Compute the inverse of the lower-triangular matrix L
     * 
     * @param L Cholesky lower-triangular factor L of cov
     * @return inversion of L
     */
    template <typename Derived>
    [[nodiscard]] auto get_Linv(const Eigen::TriangularBase<Derived>& L) {
        using Scalar = typename Derived::Scalar;
        constexpr Eigen::Index crows = Derived::RowsAtCompileTime;
        constexpr Eigen::Index ccols = Derived::ColsAtCompileTime;
        const Eigen::Index dim = L.rows();
        return L.derived().solve(Eigen::Matrix<Scalar, crows, ccols>::Identity(dim, dim)).eval();
    }
    /**
     * @return Whether the given matrix is square
     */
    template <typename Derived>
    bool is_square(const Eigen::MatrixBase<Derived>& m) {
        if constexpr (Derived::RowsAtCompileTime != Eigen::Dynamic && Derived::ColsAtCompileTime != Eigen::Dynamic)
            return Derived::RowsAtCompileTime == Derived::ColsAtCompileTime;
        else
            return m.rows() == m.cols();
    }

} // namespace internal

/**
 * @brief Container representing N(mu, cov)
 */
template <typename Scalar, int DIM=Eigen::Dynamic>
struct Gaussian {
    using MeanT = Eigen::Matrix<Scalar, DIM, 1>;
    using CovT = Eigen::Matrix<Scalar, DIM, DIM>;
    MeanT mu;       ///< Mean og the Gaussian
    CovT cov;       ///< Covariance of the Gaussian

    /**
     * @brief Construct a new Gaussian object
     * 
     * @param dim Dimension of the Gaussian (required if DIM is Dynamic)
     */
    explicit Gaussian(int dim = DIM) {
        if constexpr (DIM == Eigen::Dynamic) {
            assert(dim > 0 && dim != Eigen::Dynamic && "Runtime dim must be positive and fixed");
            mu.setZero(dim, 1);
            cov.setIdentity(dim, dim);
        } else {
            assert(dim == DIM && "Compile-time size must equal runtime size");
            (void)dim;
        }
    }

    /**
     * @brief Constructs new Gaussian from Eigen expressions
     * 
     * @param mu Mean of the underlying Gaussian
     * @param cov Covariance of the underlying Gaussian
     */
    template <typename M, typename C>
    requires is_col_vector<M> && can_be_square<C>
    Gaussian(M&& mu, C&& cov) : mu(std::forward<M>(mu)), cov(std::forward<C>(cov)) {
        assert(this->mu.rows() == this->cov.rows() && "Mean/Cov dimension mismatch");
        assert(internal::is_square(this->cov) && "Covariance must be a square");
        if constexpr (DIM != Eigen::Dynamic)
            assert(this->mu.rows() == DIM && "Input dimensions do not match compile-time DIM");
    }

    /**
     * @return Dimension of the Gaussian
     */
    [[nodiscard]] int get_dim() const noexcept {
        if constexpr (DIM == Eigen::Dynamic)
            return static_cast<int>(mu.rows());
        else
            return DIM;
    }

    /**
     * @brief Swap contents with another Gaussian
     * 
     * @param other Other Gaussian
     */
    void swap(Gaussian& other) noexcept {
        mu.swap(other.mu);
        cov.swap(other.cov);
    }
    /**
     * @brief Exchange the contents of two Gaussians
     */
    friend void swap(Gaussian& lhs, Gaussian& rhs) noexcept {
        lhs.swap(rhs);
    }
};

//------------------------------------------------------------------------------------

/**
 * @brief Compute squared Mahalanobis distance using the inverse Cholesky factor L^-1
 *
 * Computes ||L_inv * (x-mu)||^2
 * 
 * @param x Vector to measure distance to
 * @param mu Mean vector of the Gaussian
 * @param L_inv Inverse of the lower-triangular matrix L of covariance
 * @return squared mahalanobis distance
 */
template <typename DerivedA, typename DerivedB, typename DerivedC>
requires is_col_vector<DerivedA> && is_col_vector<DerivedB> && can_be_square<DerivedC>
[[nodiscard]] typename DerivedA::Scalar mahalanobis_distance_Linv(
        const Eigen::MatrixBase<DerivedA>& x, const Eigen::MatrixBase<DerivedB>& mu, const Eigen::MatrixBase<DerivedC>& L_inv) {
    assert(x.rows() == mu.rows() && x.rows() == L_inv.rows() && "Dimension mismatch");
    assert(internal::is_square(L_inv) && "L_inv must be a square");
    return (L_inv * (x - mu)).squaredNorm();
}

/**
 * @brief Compute squared Mahalanobis distance using the Cholesky factor L
 * 
 * @param x Vector to measure distance to
 * @param mu Mean vector of the Gaussian
 * @param L Lower-triangular matrix L of covariance
 * @return squared mahalanobis distance
 */
template <typename DerivedA, typename DerivedB, typename DerivedC>
requires is_col_vector<DerivedA> && is_col_vector<DerivedB>
[[nodiscard]] typename DerivedA::Scalar mahalanobis_distance_L(
        const Eigen::MatrixBase<DerivedA>& x, const Eigen::MatrixBase<DerivedB>& mu, const Eigen::TriangularBase<DerivedC>& L) {
    assert(x.rows() == mu.rows() && x.rows() == L.rows() && "Dimension mismatch");
    return L.derived().solve(x - mu).squaredNorm();
}

/**
 * @brief Compute squared Mahalanobis distance
 * 
 * @param x Vector to measure distance to
 * @param mu Mean vector of the Gaussian
 * @param cov Covariance of the Gaussian
 * @return squared mahalanobis distance
 */
template <typename DerivedA, typename DerivedB, typename DerivedC>
requires is_col_vector<DerivedA> && is_col_vector<DerivedB> && can_be_square<DerivedC>
[[nodiscard]] typename DerivedA::Scalar mahalanobis_distance(
        const Eigen::MatrixBase<DerivedA>& x, const Eigen::MatrixBase<DerivedB>& mu, const Eigen::MatrixBase<DerivedC>& cov) {
    auto llt = cov.llt();
    if (llt.info() != Eigen::Success)
        throw std::runtime_error("LLT failed");
    return mahalanobis_distance_L(x, mu, llt.matrixL());
}

//------------------------------------------------------------------------------------

/**
 * @brief Calculates the logartihm of multivariate normal probability density at x
 *
 * Uneffective for batched calculations (recalculation of L)
 * 
 * @param x Vector to enumerate pdf at
 * @param mu Mean vector of the Gaussian
 * @param cov Covariance of the Gaussian
 * @return log of pdf at x
 */
template <typename DerivedA, typename DerivedB, typename DerivedC>
requires is_col_vector<DerivedA> && is_col_vector<DerivedB> && can_be_square<DerivedC>
[[nodiscard]] typename DerivedA::Scalar mvn_logpdf(
        const Eigen::MatrixBase<DerivedA>& x, const Eigen::MatrixBase<DerivedB>& mu, const Eigen::MatrixBase<DerivedC>& cov) {
    using Scalar = DerivedA::Scalar;
    assert(x.rows() == mu.rows() && x.rows() == cov.rows() && "Dimension mismatch");
    assert(internal::is_square(cov) && "cov must be a square");

    auto llt = cov.derived().llt();
    if (llt.info() != Eigen::Success)
        throw std::runtime_error("cov isn't PD, try adding jitter");
    const auto& L = llt.matrixL();
    Scalar logc = internal::logc_from_L(L);
    Scalar qform = -Scalar(0.5) * mahalanobis_distance_L(x, mu, L);
    return logc + qform;
}

//------------------------------------------------------------------------------------

/**
 * @brief Calculates the multivariate normal probability density at x
 *
 * Uneffective for batched calculations (recalculation of L)
 * 
 * @param x Vector to enumerate pdf at
 * @param mu Mean vector of the Gaussian
 * @param cov Covariance of the Gaussian
 * @return pdf at x
 */
template <typename DerivedA, typename DerivedB, typename DerivedC>
requires is_col_vector<DerivedA> && is_col_vector<DerivedB> && can_be_square<DerivedC>
[[nodiscard]] typename DerivedA::Scalar mvn_pdf(
        const Eigen::MatrixBase<DerivedA>& x, const Eigen::MatrixBase<DerivedB>& mu, const Eigen::MatrixBase<DerivedC>& cov) {
    return std::exp(mvn_logpdf(x, mu, cov));
}

//------------------------------------------------------------------------------------

/**
 * @brief Struct for repeated calculation of logpdf
 * 
 * @tparam Scalar Scalar type used
 * @tparam DIM dimension of the gaussian
 */
template <typename Scalar, int DIM=Eigen::Dynamic>
struct MvnLogPdf {
    using MeanT = Eigen::Matrix<Scalar, DIM, 1>;
    using CovT = Eigen::Matrix<Scalar, DIM, DIM>;
    using GaussianT = Gaussian<Scalar, DIM>;

    Scalar logc;        ///< Constant term (normalization factor)
    MeanT mu;           ///< Mean of the Gaussian
    CovT L_inv;         ///< Inverse of the Lower Cholesky factor of covariance

    /**
     * @brief Constructs pdf from mean and cov
     * 
     * @param mu Mean vector of the Gaussian
     * @param cov Covariance of the Gaussian
     */
    template <typename DerivedA, typename DerivedB>
    requires is_col_vector<DerivedA> && can_be_square<DerivedB>
    MvnLogPdf(const Eigen::MatrixBase<DerivedA>& mu, const Eigen::MatrixBase<DerivedB>& cov)
        : MvnLogPdf(mu, cov.derived().llt()) {
        assert(mu.rows() == cov.rows() && "Dimension mismatch");
        assert(internal::is_square(cov) && "cov must be a square");
    }
    /**
     * @brief Constructs pdf from mean and LLT of cov
     * 
     * @param mu Mean vector of the Gaussian
     * @param llt LLT of covariance of the Gaussian
     */
    template <typename DerivedA, typename DerivedB>
    requires is_col_vector<DerivedA>
    MvnLogPdf(const Eigen::MatrixBase<DerivedA>& mu, const Eigen::LLT<DerivedB>& llt) : mu(mu) {
        assert(mu.rows() == llt.rows() && "Dimension mismatch");
        if (llt.info() != Eigen::Success)
            throw std::runtime_error("Passed failed llt");
        if constexpr (DIM != Eigen::Dynamic)
            assert(mu.rows() == DIM && "Input dimensions do not match compile-time DIM");
        const auto& L = llt.matrixL();
        logc = internal::logc_from_L(L);
        L_inv = internal::get_Linv(L).eval();
    }

    /**
     * @brief Evaluate pdf at x
     * 
     * @param x Vector to evalute pdf at
     * @return log of pdf at x
     */
    template <typename Derived>
    requires is_col_vector<Derived>
    [[nodiscard]] Scalar operator() (const Eigen::MatrixBase<Derived>& x) const {
        assert(mu.rows() == x.rows() && "Dimension mismatch");
        Scalar qform = -Scalar(0.5) * mahalanobis_distance_Linv(x, mu, L_inv);
        return logc + qform;
    }
};

//------------------------------------------------------------------------------------

/**
 * @brief Struct for repeated calculation of pdf
 * 
 * @tparam Scalar Scalar type used
 * @tparam DIM dimension of the gaussian
 */
template <typename Scalar, int DIM=Eigen::Dynamic>
struct MvnPdf {
    using GaussianT = Gaussian<Scalar, DIM>;

    MvnLogPdf<Scalar, DIM> logpdf;

    /**
     * @brief Constructs pdf from mean and cov
     * 
     * @param mu Mean vector of the Gaussian
     * @param cov Covariance of the Gaussian
     */
    template <typename DerivedA, typename DerivedB>
    requires is_col_vector<DerivedA> && can_be_square<DerivedB>
    MvnPdf(const Eigen::MatrixBase<DerivedA>& mu, const Eigen::MatrixBase<DerivedB>& cov) : logpdf(mu, cov) {}
    /**
     * @brief Constructs pdf from mean and LLT of cov
     * 
     * @param mu Mean vector of the Gaussian
     * @param llt LLT of covariance of the Gaussian
     */
    template <typename DerivedA, typename DerivedB>
    requires is_col_vector<DerivedA>
    MvnPdf(const Eigen::MatrixBase<DerivedA>& mu, const Eigen::LLT<DerivedB>& llt) : logpdf(mu, llt) {}

    /**
     * @brief Evaluate pdf at x
     * 
     * @param x Vector to evalute pdf at
     * @return pdf at x
     */
    template <typename Derived>
    requires is_col_vector<Derived>
    [[nodiscard]] Scalar operator() (const Eigen::MatrixBase<Derived>& x) const {
        return std::exp(logpdf(x));
    }
};

} // namespace mtt