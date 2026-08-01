/**
 * @file      phd.hpp
 * @brief     GM-PHD Filter implementation
 * @author    @vojtagart
 * @date      17/02/2026
 * @see       https://github.com/Vojtagart/mtt
 * * @par Credits & Acknowledgments
 * - Algorithm inspired by: B. . -N. Vo and W. . -K. Ma, "The Gaussian Mixture Probability Hypothesis Density Filter," in IEEE Transactions on Signal Processing, vol. 54, no. 11, pp. 4091-4104, Nov. 2006
 * - Documentation formatted and refined with the assistance of Google Gemini
 * * @copyright Copyright (c) 2026 @vojtagart. Released under the MIT License.
 */

#pragma once
#include <vector>
#include <numeric>
#include <algorithm>
#include <limits>
#include <utility>
#include <Eigen/Core>
#include <Eigen/Cholesky>

#include "../core/core.hpp"


namespace mtt::phd {

/**
 * @brief Reusable workers for PHD update and merge
 * 
 * @tparam Scalar Value type
 * @tparam MDIM Dimension of the measurement (number or Eigen::Dynamic)
 */
template <typename Scalar, int MDIM=Eigen::Dynamic>
struct UpdateWorkers {
    UpdateWorkers(int mdim = MDIM) : gater(mdim) {}

    std::vector<int> from_z;                        ///< From which measurement this component originated
    std::vector<Scalar> w_z;                        ///< Total weight of components created from this measruement (for normalization)
    mtt::Gater<Scalar, MDIM> gater;                 ///< Gater over the measurement set

    /**
     * @brief Reserves space for the workers
     * 
     * @param n_from_z what from_z initialize to
     * @param n_meas expected number of measurements
     */
    void reserve(size_t n_from_z, size_t n_meas) {
        from_z.reserve(n_from_z);
        w_z.reserve(n_meas);
    }
};

/**
 * @brief Reusable workers for PHD update and merge
 * 
 * @tparam Scalar Value type
 * @tparam DIM Dimension of the mixutre (number or Eigen::Dynamic)
 */
template <typename Scalar, int DIM=Eigen::Dynamic>
struct MergeWorkers {
    MergeWorkers(int sdim = DIM) : mix(sdim) {}

    mtt::GaussianMixture<Scalar, DIM> mix;                      ///< Helper mixture for merging
    std::vector<int> all_idxs;                                  ///< Indices for merge sort
    std::vector<int> cur_idxs;                                  ///< Indices for current merge step
    std::vector<Eigen::Matrix<Scalar, DIM, DIM>> L_invs;        ///< Precomputed inverses for merging
    std::vector<int> order;                                     ///< Indices for pruning

    /**
     * @brief Reserves space for the workers
     * 
     * @param n_comps expected number of components
     */
    void reserve(size_t n_comps) {
        mix.reserve(n_comps);
        all_idxs.reserve(n_comps);
        cur_idxs.reserve(n_comps);
        L_invs.reserve(n_comps);
        order.reserve(n_comps);
    }
};


//------------------------------------------------------------------------------------

/**
 * @brief Performs a PHD prediction
 * 
 * @param post Posterior mixture
 * @param births Birth mixture
 * @param PS probability of survival
 * @param F Transirtion matrix
 * @param Q Process noise matrix
 */
template <typename DerivedA, typename DerivedB, typename Scalar, int SDIM>
requires (mtt::can_be_square<DerivedA> && mtt::can_be_square<DerivedB>)
void prediction(
        mtt::GaussianMixture<Scalar, SDIM>& post, const mtt::GaussianMixture<Scalar, SDIM>& births,
        Scalar PS, const Eigen::MatrixBase<DerivedA>& F, const Eigen::MatrixBase<DerivedB>& Q) {
    
    const auto sdim = post.get_dim();
    assert(F.rows() == F.cols() && F.rows() == sdim && "F must have shape (SDIM, SDIM)");
    assert(Q.rows() == Q.cols() && Q.rows() == sdim && "Q must have shape (SDIM, SDIM)");
    assert(births.get_dim() == sdim && "Mixture dims must match");

    // weight prediction
    post.scale_weight(PS);

    // Gaussian KF prediction
    for (size_t i = 0; i < post.size(); i++) {
        auto mu = post.mu(i);
        mtt::kf_predict_mu(mu, F);
    }
    for (size_t i = 0; i < post.size(); i++) {
        auto cov = post.cov(i);
        mtt::kf_predict_cov(cov, F, Q);
    }

    // Inserting new born targets
    for (size_t i = 0; i < births.size(); i++) {
        post.push(births.w(i), births.mu(i), births.cov(i));
    }
}

//------------------------------------------------------------------------------------

/**
 * @brief Performs a PHD update
 * 
 * @param prior prior mixture
 * @param Z Matrix of measurements
 * @param PD Probability of detection
 * @param PG Probability of being inside the gate
 * @param lambda clutter lambda (from PPP)
 * @param H Measurement matrix
 * @param R Measurement noise
 * @param W Workers
 */
template <typename DerivedA, typename DerivedB, typename DerivedC, typename Scalar, int SDIM, int MDIM>
requires (mtt::can_be_square<DerivedC>)
void update(
        mtt::GaussianMixture<Scalar, SDIM>& prior, const Eigen::MatrixBase<DerivedA>& Z, Scalar PD, Scalar PG,
        Scalar lambda, const Eigen::MatrixBase<DerivedB>& H, const Eigen::MatrixBase<DerivedC>& R, UpdateWorkers<Scalar, MDIM>& W) {

    using StateT = Eigen::Matrix<Scalar, SDIM, 1>;
    using MeasT = Eigen::Matrix<Scalar, MDIM, 1>;
    
    const auto sdim = prior.get_dim();
    const auto mdim = Z.rows();
    assert(R.rows() == R.cols() && R.rows() == mdim && "R must have shape (MDIM, MDIM)");
    assert(H.rows() == mdim && H.cols() == sdim && "RH must have shape (MDIM, SDIM)");

    size_t n_p = prior.size();
    size_t n_z = static_cast<size_t>(Z.cols());
    
    // Initialized to lambda - Kappa from the paper
    W.w_z.assign(n_z, lambda);
    W.gater.set_measurements(Z);
    W.from_z.clear();
    
    Scalar thr = mtt::chi2_inv<Scalar>(PG, static_cast<Scalar>(mdim));

    // Perform gating
    for (size_t i = 0; i < n_p; i++) {
        auto S_LLT = mtt::kf_S_LLT(prior.cov(i), H, R);
        if (S_LLT.info() != Eigen::Success)
            throw std::runtime_error("LLT failed");
        Eigen::Matrix<Scalar, MDIM, MDIM> L_inv = mtt::internal::get_Linv(S_LLT.matrixL()).eval();
        MeasT eta = mtt::kf_eta(prior.mu(i), H);
        W.gater.gate_Linv(eta, L_inv, thr);

        if (W.gater.gated_size() == 0) continue;

        // Compute updated components that doesnt depend on z
        auto [P, K] = mtt::kf_cov_K(prior.cov(i), S_LLT, H);

        // constant used in gaussian pdf
        Scalar logc = mtt::internal::logc_from_L(S_LLT.matrixL());
        Scalar wiPD = prior.w(i) * PD;

        // Create new Gaussian components
        for (size_t j = 0; j < W.gater.gated_size(); j++) {
            auto z_idx = W.gater.idxs[j];

            // reusing already computed Mahal dist for likelihood
            Scalar lkl = std::exp(logc - Scalar(0.5) * W.gater.distances[j]);
            Scalar nw = wiPD * lkl;

            StateT mean = mtt::kf_mean(prior.mu(i), Z.col(z_idx), K, eta);
            // push new component into the mixture
            prior.push(nw, std::move(mean), P);

            W.from_z.push_back(z_idx);
            W.w_z[z_idx] += nw;
        }
    }
    
    size_t added = prior.size() - n_p;

    // Normalize the weights
    for (size_t i = 0; i < added; i++) {
        prior.w(i + n_p) /= W.w_z[W.from_z[i]];
    }
    
    // Update the original components
    for (size_t i = 0; i < n_p; i++) {
        prior.w(i) *= Scalar(1) - PD * PG;
    }
}

/**
 * @brief Performs a PHD update
 * 
 * @param prior prior mixture
 * @param Z Matrix of measurements
 * @param PD Probability of detection
 * @param PG Probability of being inside the gate
 * @param lambda clutter lambda (from PPP)
 * @param H Measurement matrix
 * @param R Measurement noise
 */
template <typename DerivedA, typename DerivedB, typename DerivedC, typename Scalar, int SDIM>
requires (mtt::can_be_square<DerivedC>)
void update(
        mtt::GaussianMixture<Scalar, SDIM>& prior, const Eigen::MatrixBase<DerivedA>& Z, Scalar PD, Scalar PG,
        Scalar lambda, const Eigen::MatrixBase<DerivedB>& H, const Eigen::MatrixBase<DerivedC>& R) {

    constexpr int MDIM = DerivedA::RowsAtCompileTime;
    const int mdim = H.rows();
    UpdateWorkers<Scalar, MDIM> workers(mdim);
    size_t n_p = prior.size();
    size_t n_z = static_cast<size_t>(Z.derived().cols());
    size_t exp_comp = n_p * std::min(n_z, size_t(5));
    prior.reserve(exp_comp + n_p);
    workers.reserve(exp_comp, n_z);
    update(prior, Z, PD, PG, lambda, H, R, workers);
}

//------------------------------------------------------------------------------------

/**
 * @brief Performs pruning by merging
 * 
 * @param mix Mixture to be prunned
 * @param min_w Threshold for weight - comps below threshold are removed before merging
 * @param merge_thr Maximum squared mahal. distance for components to be merged
 * @param max_comps Maximum number of components after merging
 * @param W Workers
 */
template <typename Scalar, int SDIM>
void merge(mtt::GaussianMixture<Scalar, SDIM>& mix, Scalar min_w, Scalar merge_thr, size_t max_comps, MergeWorkers<Scalar, SDIM>& W) {

    assert(W.mix.get_dim() == mix.get_dim());

    mix.filter_out(min_w);
    if (mix.size() == 0) return;

    W.all_idxs.resize(mix.size());
    std::iota(W.all_idxs.begin(), W.all_idxs.end(), 0);
    std::sort(W.all_idxs.begin(), W.all_idxs.end(), [&](uint32_t x, uint32_t y){
        return mix.w(x) < mix.w(y);
    });
    W.L_invs.clear();
    W.mix.clear();

    // precompute L inverses for each component
    for (size_t i = 0; i < mix.size(); i++) {
        auto cov = mix.cov(i);
        auto llt = cov.llt();
        if (llt.info() != Eigen::Success)
            throw std::runtime_error("LLT failed");
        Eigen::Matrix<Scalar, SDIM, SDIM> x = mtt::internal::get_Linv(llt.matrixL());
        W.L_invs.push_back(x);
    }
    while (!W.all_idxs.empty()) {
        // Indexes are sorted as per the weight and the order isnt changed
        size_t idx = W.all_idxs.back();

        W.cur_idxs.clear();
        const auto& mu = mix.mu(idx);
        // const auto& L_inv = W.L_invs[idx];               // <----- used for distance from the dominant comp

        mtt::MixtureToGaussian<Scalar, SDIM> mtg(mix.get_dim());

        for (auto i : W.all_idxs) {
            if (mtt::mahalanobis_distance_Linv(mu, mix.mu(i), W.L_invs[i]) <= merge_thr)        // <----- distance to the dominant comp
                mtg.add_gauss(mix.w(i), mix.mu(i), mix.cov(i));
            // if (mtt::mahalanobis_distance_Linv(mix.mu(i), mu, L_inv) <= merge_thr)           // <----- distance from the dominant comp
            //     mtg.add_gauss(mix.w(i), mix.mu(i), mix.cov(i));
            else
                W.cur_idxs.push_back(i);
        }
        // perform moment matching to obtain single Gaussian
        auto gauss = mtg.get_gauss();
        W.mix.push(mtg._w, gauss.mu, gauss.cov);
        // set all_idxs to those that weren't used
        W.cur_idxs.swap(W.all_idxs);
    }
    mix.swap(W.mix);
    
    if (mix.size() <= max_comps) return;

    W.order.resize(mix.size());
    std::iota(W.order.begin(), W.order.end(), 0);
    std::nth_element(W.order.begin(), W.order.begin() + max_comps - 1, W.order.end(), [&](uint32_t x, uint32_t y){
        return mix.w(x) > mix.w(y);
    });
    W.mix.clear();
    for (size_t i = 0; i < max_comps; i++) {
        auto idx = W.order[i];
        W.mix.push(mix.w(idx), mix.mu(idx), mix.cov(idx));
    }
    mix.swap(W.mix);
}

/**
 * @brief Performs pruning by merging
 * 
 * @param mix Mixture to be prunned
 * @param min_w Threshold for weight - comps below threshold are removed before merging
 * @param merge_thr Maximum squared mahal. distance for components to be merged
 * @param max_comps Maximum number of components after merging
 */
template <typename Scalar, int SDIM>
void merge(mtt::GaussianMixture<Scalar, SDIM>& mix, Scalar min_w, Scalar merge_thr, size_t max_comps) {
    MergeWorkers<Scalar, SDIM> workers(mix.get_dim());
    workers.reserve(mix.size());
    merge(mix, min_w, merge_thr, max_comps, workers);
}

//------------------------------------------------------------------------------------

/**
 * @brief Estimates the state based on the given mixture
 * 
 * @param mix Mixture
 * @param conf_thr Minimum weight of a component
 * @return (means, covariances) of the estimated targets
 */
template <typename Scalar, int SDIM>
auto estimator(const mtt::GaussianMixture<Scalar, SDIM>& mix, Scalar conf_thr) {
    size_t cnt = 0;
    for (size_t i = 0; i < mix.size(); i++) {
        if (mix.w(i) >= conf_thr)
            cnt++;
    }
    int sdim = mix.get_dim();
    Eigen::Matrix<Scalar, SDIM, Eigen::Dynamic> mus(sdim, cnt);
    constexpr int COV_ROWS = (SDIM == Eigen::Dynamic ? Eigen::Dynamic : SDIM * SDIM);
    Eigen::Matrix<Scalar, COV_ROWS, Eigen::Dynamic> covs(sdim*sdim, cnt);

    size_t idx = 0;
    for (size_t i = 0; i < mix.size(); i++) {
        if (mix.w(i) >= conf_thr) {
            mus.col(idx) = mix.mu(i);
            Eigen::Map<Eigen::Matrix<Scalar, SDIM, SDIM>>(covs.col(idx).data(), sdim, sdim) = mix.cov(i);
            idx++;
        }
    }
    return std::make_pair(mus, covs);
}

} // namespace mtt::phd