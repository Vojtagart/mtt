/**
 * @file      kalman_filter.hpp
 * @brief     Kalman Filter implementation
 * @author    @vojtagart
 * @date      17/02/2026
 * @see       https://github.com/Vojtagart/mtt
 * * @par Credits & Acknowledgments
 * - Documentation formatted and refined with the assistance of Google Gemini
 * * @copyright Copyright (c) 2026 @vojtagart. Released under the MIT License.
 */

#pragma once
#include <tuple>
#include <utility>
#include <Eigen/Core>
#include <Eigen/Cholesky>

#include "gaussian.hpp"
#include "eigen_concepts.hpp"


namespace mtt {

/**
 * @brief Ensures symmetry and PF of given matrix
 * 
 * !!! WARNING !!! this function do not evaluate m, so it may
 * bloat the expression tree and force unnecasarlly calculations
 * 
 * @param m Matrix to be corrected
 * @param jitter Jitter to be added to diagonal
 * @return Corrected matrix AS EXPRESSION
 */
template <typename Derived>
requires can_be_square<Derived>
[[nodiscard]] auto correct_cov(const Eigen::MatrixBase<Derived>& m, typename Derived::Scalar jitter = 1e-8) {
    assert(m.cols() == m.rows());
    using Scalar = typename Derived::Scalar;
    constexpr int crows = Derived::RowsAtCompileTime;
    constexpr int ccols = Derived::ColsAtCompileTime;
    int dim = static_cast<int>(m.rows());
    assert(dim == m.cols());
    return Scalar(0.5) * (m + m.transpose()) + jitter * Eigen::Matrix<Scalar, crows, ccols>::Identity(dim, dim);
}

/**
 * @brief Ensures symmetry and PF of given matrix
 * 
 * If m doesnt need to be evaluated, prefer correct_cov
 * 
 * @param m Matrix to be corrected
 * @param jitter Jitter to be added to diagonal
 * @return Corrected matrix
 */
template <typename Derived>
requires can_be_square<Derived>
[[nodiscard]] auto correct_cov_eval(const Eigen::MatrixBase<Derived>& m, typename Derived::Scalar jitter = 1e-8) {
    assert(m.cols() == m.rows());
    auto m_eval = m.eval();
    return correct_cov(m_eval, jitter).eval();
}

//------------------------------------------------------------------------------------

/**
 * @brief Performs inplace Kalman prediction
 * 
 * @param mu Mean at previous time step
 * @param F Transition matrix
 */
template <typename DerivedA, typename DerivedB>
requires is_col_vector<DerivedA> && can_be_square<DerivedB>
void kf_predict_mu(Eigen::MatrixBase<DerivedA>& mu, const Eigen::MatrixBase<DerivedB>& F) {
    assert(F.rows() == F.cols() && "F must be square");
    assert(mu.rows() == F.rows() && "F and mu must have the same dim");
    mu = F * mu;
}

/**
 * @brief Performs inplace Kalman prediction
 * 
 * @param cov Covariance at previous time step
 * @param F Transition matrix
 * @param Q Process noise
 */
template <typename DerivedA, typename DerivedB, typename DerivedC>
requires can_be_square<DerivedA> && can_be_square<DerivedB> && can_be_square<DerivedC>
void kf_predict_cov(
        Eigen::MatrixBase<DerivedA>& cov, const Eigen::MatrixBase<DerivedB>& F, const Eigen::MatrixBase<DerivedC>& Q) {
    assert(cov.rows() == cov.cols() && "Cov must be square");
    assert(F.rows() == F.cols() && F.rows() == cov.rows() && "F must be square with the same dim as cov");
    assert(Q.rows() == Q.cols() && Q.rows() == cov.rows() && "Q must be square with the same dim as cov");
    cov = correct_cov_eval(F * cov * F.transpose() + Q);
}

//------------------------------------------------------------------------------------

/**
 * @brief Performs inplace Kalman prediction
 * 
 * @param mu Mean at previous time step
 * @param cov Covariance at previous time step
 * @param F Transition matrix
 * @param Q Process noise
 */
template <typename DerivedA, typename DerivedB, typename DerivedC, typename DerivedD>
requires is_col_vector<DerivedA> && can_be_square<DerivedB> && can_be_square<DerivedC> && can_be_square<DerivedD>
void kf_predict(
        Eigen::MatrixBase<DerivedA>& mu, Eigen::MatrixBase<DerivedB>& cov,
        const Eigen::MatrixBase<DerivedC>& F, const Eigen::MatrixBase<DerivedD>& Q) {
    kf_predict_mu(mu, F);
    kf_predict_cov(cov, F, Q);
}

//------------------------------------------------------------------------------------

/**
 * @brief Computes Eta for Kalman update
 *
 * @param mu Predicted Mean
 * @param H Measurement matrix
 * @return Eta for Kalman update AS EXPRESSION
 */
template <typename DerivedA, typename DerivedB>
requires is_col_vector<DerivedA>
[[nodiscard]] auto kf_eta(const Eigen::MatrixBase<DerivedA>& mu, const Eigen::MatrixBase<DerivedB>& H) {
    assert(mu.rows() == H.cols());
    return H * mu;
}

//------------------------------------------------------------------------------------

/**
 * @brief Computes Innovation matrix
 * 
 * @param cov Predicted covariance
 * @param H Measurement matrix
 * @param R Measurement noise
 * @return Innovation matrix
 */
template <typename DerivedA, typename DerivedB, typename DerivedC>
requires can_be_square<DerivedA> && can_be_square<DerivedC>
[[nodiscard]] auto kf_S(
        const Eigen::MatrixBase<DerivedA>& cov, const Eigen::MatrixBase<DerivedB>& H, const Eigen::MatrixBase<DerivedC>& R) {
    assert(cov.rows() == cov.cols() && "cov must be a square");
    assert(R.rows() == R.cols() && "R must be a square");
    assert(H.rows() == R.rows() && H.cols() == cov.rows() && "H must have shape (MDIM, SDIM)");
    return correct_cov_eval(H * cov * H.transpose() + R);
}

//------------------------------------------------------------------------------------

/**
 * @brief Computes inverse of the Innovation matrix
 * 
 * @param cov Predicted covariance
 * @param H Measurement matrix
 * @param R Measurement noise
 * @return inverse of the Innovation matrix
 */
template <typename DerivedA, typename DerivedB, typename DerivedC>
requires can_be_square<DerivedA> && can_be_square<DerivedC>
[[nodiscard]] auto kf_S_inv(
        const Eigen::MatrixBase<DerivedA>& cov, const Eigen::MatrixBase<DerivedB>& H, const Eigen::MatrixBase<DerivedC>& R) {
    using MMT = Eigen::Matrix<typename DerivedA::Scalar, DerivedA::RowsAtCompileTime, DerivedA::ColsAtCompileTime>;

    assert(cov.rows() == cov.cols() && "Cov must be a sqaure");
    assert(R.rows() == R.cols() && "R must be a sqaure");
    assert(H.rows() == R.rows() && H.cols() == cov.rows() && "H must have shape (MDIM, SDIM)");
    auto S = kf_S(cov, H, R);
    Eigen::LLT<MMT> llt(S);
    if (llt.info() != Eigen::Success)
        throw std::runtime_error("S isn't PD");
    int dim = cov.rows();
    return llt.solve(MMT::Identity(dim, dim)).eval();
}

//------------------------------------------------------------------------------------

/**
 * @brief Computes Eigen::LLT of the Innovation matrix
 * 
 * @param cov Predicted covariance
 * @param H Measurement matrix
 * @param R Measurement noise
 * @return Eigen::LLT of the Innovation matrix
 */
template <typename DerivedA, typename DerivedB, typename DerivedC>
requires can_be_square<DerivedA> && can_be_square<DerivedC>
[[nodiscard]] auto kf_S_LLT(
        const Eigen::MatrixBase<DerivedA>& cov, const Eigen::MatrixBase<DerivedB>& H, const Eigen::MatrixBase<DerivedC>& R) {
    auto S = kf_S(cov, H, R);
    auto llt = S.llt();
    if (llt.info() != Eigen::Success)
        throw std::runtime_error("LLT failed");
    return llt;
}

//------------------------------------------------------------------------------------

/**
 * @brief Computes Updated covariance and Kalman gain
 * 
 * @param cov Predicted covariance
 * @param S_inv Inverse of the innovation matrix
 * @param H Measurement matrix
 * @return pair (Updated covariance, Kalman gain)
 */
template <typename DerivedA, typename DerivedB, typename DerivedC>
requires can_be_square<DerivedA> && can_be_square<DerivedB>
[[nodiscard]] auto kf_cov_K(
        const Eigen::MatrixBase<DerivedA>& cov, const Eigen::MatrixBase<DerivedB>& S_inv, const Eigen::MatrixBase<DerivedC>& H) {
    assert(cov.rows() == cov.cols() && "Cov must be a sqaure");
    assert(S_inv.rows() == S_inv.cols() && "S_inv must be a sqaure");
    assert(H.rows() == S_inv.rows() && H.cols() == cov.rows() && "H must have shape (MDIM, SDIM)");
    auto A = (cov * H.transpose()).eval();
    auto K = (A * S_inv).eval();
    auto P = correct_cov_eval(cov - K * A.transpose());
    return std::make_pair(P, K);
}

/**
 * @brief Computes Updated covariance and Kalman gain
 * 
 * @param cov Predicted covariance
 * @param S_LLT Eigen::LLT of the innovation matrix
 * @param H Measurement matrix
 * @return pair (Updated covariance, Kalman gain)
 */
template <typename DerivedA, typename DerivedB, typename DerivedC>
requires can_be_square<DerivedA> && can_be_square<DerivedB>
[[nodiscard]] auto kf_cov_K(
        const Eigen::MatrixBase<DerivedA>& cov, const Eigen::LLT<DerivedB>& S_LLT, const Eigen::MatrixBase<DerivedC>& H) {
    assert(cov.rows() == cov.cols() && "Cov must be a sqaure");
    assert(H.cols() == cov.rows() && "H must have shape (MDIM, SDIM)");
    if (S_LLT.info() != Eigen::Success)
        throw std::runtime_error("Passed failed LLT");
    auto A = (cov * H.transpose()).eval();
    auto K = (S_LLT.solve(A.transpose()).transpose()).eval();
    auto P = correct_cov_eval(cov - K * A.transpose());
    return std::make_pair(P, K);
}

//------------------------------------------------------------------------------------

/**
 * @brief Computes updated mean
 * 
 * @param mu Predicted mean
 * @param z Measurement to update based on
 * @param K Kalman gain
 * @param eta Eta
 * @return Updated mean
 */
template <typename DerivedA, typename DerivedB, typename DerivedC, typename DerivedD>
requires is_col_vector<DerivedA> && is_col_vector<DerivedB> && is_col_vector<DerivedD>
[[nodiscard]] auto kf_mean(
        const Eigen::MatrixBase<DerivedA>& mu, const Eigen::MatrixBase<DerivedB>& z,
        const Eigen::MatrixBase<DerivedC>& K, const Eigen::MatrixBase<DerivedD>& eta) {
    assert(mu.rows() == K.rows() && "mu and K shapes must match");
    assert(K.cols() == z.rows() && "z and K shapes must match");
    assert(z.rows() == eta.rows() && "z and eta must have the same shape");
    return mu + K * (z - eta);
}

//------------------------------------------------------------------------------------

/**
 * @brief Performs inplace Kalman update
 * 
 * @param mu Predicted mean
 * @param cov Predicted covariance
 * @param z Measurement to update based on
 * @param S_LLT Eigen::LLT of the innovation matrix
 * @param H Measurement matrix
 * @param R Measurement noise
 */
template <typename DerivedA, typename DerivedB, typename DerivedC, typename DerivedD, typename DerivedE>
requires is_col_vector<DerivedA> && can_be_square<DerivedB> && is_col_vector<DerivedC> && can_be_square<DerivedD>
void kf_update(
        Eigen::MatrixBase<DerivedA>& mu, Eigen::MatrixBase<DerivedB>& cov, const Eigen::MatrixBase<DerivedC>& z,
        const Eigen::LLT<DerivedD>& S_LLT, const Eigen::MatrixBase<DerivedE>& H) {
    assert(cov.rows() == cov.cols() && "Cov must be a sqaure");
    assert(H.cols() == cov.rows() && H.rows() == z.rows() && "H must have shape (MDIM, SDIM)");
    if (S_LLT.info() != Eigen::Success)
        throw std::runtime_error("Passed failed LLT");
    auto [P, K] = kf_cov_K(cov, S_LLT, H);
    auto innov = (z - H * mu).eval();
    mu += K * innov;
    cov = P;
}

/**
 * @brief Performs inplace Kalman update
 * 
 * @param mu Predicted mean
 * @param cov Predicted covariance
 * @param z Measurement to update based on
 * @param H Measurement matrix
 * @param R Measurement noise
 */
template <typename DerivedA, typename DerivedB, typename DerivedC, typename DerivedD, typename DerivedE>
requires is_col_vector<DerivedA> && can_be_square<DerivedB> && is_col_vector<DerivedC> && can_be_square<DerivedE>
void kf_update(
        Eigen::MatrixBase<DerivedA>& mu, Eigen::MatrixBase<DerivedB>& cov, const Eigen::MatrixBase<DerivedC>& z,
        const Eigen::MatrixBase<DerivedD>& H, const Eigen::MatrixBase<DerivedE>& R) {
    auto llt = kf_S_LLT(cov, H, R);
    if (llt.info() != Eigen::Success)
        throw std::runtime_error("S isn't PD");
    kf_update(mu, cov, z, llt, H);
}

} // namespace mtt