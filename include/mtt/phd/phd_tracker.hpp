/**
 * @file      phd_tracker.hpp
 * @brief     Wrapper class for the GM-PHD Tracker
 * @author    @vojtagart
 * @date      17/02/2026
 * @see       https://github.com/Vojtagart/mtt
 * * @par Credits & Acknowledgments
 * - Documentation formatted and refined with the assistance of Google Gemini
 * * @copyright Copyright (c) 2026 @vojtagart. Released under the MIT License.
 */

#pragma once
#include <vector>
#include <tuple>
#include <utility>
#include <algorithm>
#include <cassert>
#include <optional>
#include <Eigen/Core>

#include "phd.hpp"
#include "../core/core.hpp"


namespace mtt {
namespace phd {

/**
 * @brief Wrapper for the PHD recursion
 * 
 * @tparam Scalar Value type
 * @tparam SDIM Dimension of the state (int or Eigen::Dynamic)
 * @tparam MDIM Dimension of the measurement (int or Eigen::Dynamic)
 */
template <typename Scalar, int SDIM=Eigen::Dynamic, int MDIM=Eigen::Dynamic>
struct Tracker {

    using StateT = Eigen::Matrix<Scalar, SDIM, 1>;
    using MeasT = Eigen::Matrix<Scalar, MDIM, 1>;
    using StateCovT = Eigen::Matrix<Scalar, SDIM, SDIM>;
    using MeasCovT = Eigen::Matrix<Scalar, MDIM, MDIM>;
    using HT = Eigen::Matrix<Scalar, MDIM, SDIM>;
    using GMT = mtt::GaussianMixture<Scalar, SDIM>;

    GMT mix;
    GMT births;
    std::optional<GMT> births0;

    StateCovT F;
    StateCovT Q;
    HT H;
    MeasCovT R;

    Scalar PD;
    Scalar PS;
    Scalar lambda;
    Scalar PG;

    Scalar trunc_thr;
    Scalar merge_thr;
    size_t max_comps;
    Scalar conf_thr;

    phd::UpdateWorkers<Scalar, MDIM> update_workers;
    phd::MergeWorkers<Scalar, SDIM> merge_workers;

    bool t0 = true;

    /**
     * @brief Initalizes Tracker with given parameters
     * 
     * @param births Birth Gaussian components
     * @param F Transition matrix
     * @param Q Process noise
     * @param H Measurement matrix
     * @param R Measurement noise
     * @param PD Probability of detection
     * @param PS Probability of survival
     * @param lambda Clutter intensity
     * @param PG Probability of being inside a gate
     * @param trunc_thr truncation threshold (minimal component weight)
     * @param merge_thr Merge threshold (minimal squared Mahal. dist to merge components)
     * @param max_comps Maximum number of components after performing all steps
     * @param conf_thr Confirmation threshold - used for estimation of the state
     * @param _births0 Override of the initial birth mixture
     */
    Tracker(const GMT& births, const StateCovT& F, const StateCovT& Q, const HT& H, const MeasCovT& R,
            Scalar PD, Scalar PS, Scalar lambda, Scalar PG = 0.99, Scalar trunc_thr = 1e-5,
            Scalar merge_thr = 4, size_t max_comps = 250, Scalar conf_thr = 0.5, std::optional<GMT> _births0 = {})
            : mix(F.rows()), births(births), births0(std::move(_births0)), F(F), Q(Q), H(H), R(R), PD(PD), PS(PS), lambda(lambda), PG(PG), 
              trunc_thr(trunc_thr), merge_thr(merge_thr),
              max_comps(max_comps), conf_thr(conf_thr), update_workers(H.rows()), merge_workers(F.rows()) {
        const auto sdim = F.rows();
        const auto mdim = H.rows();
        assert(F.rows() == F.cols() && "F must be square");
        assert(Q.rows() == Q.cols() && Q.rows() == sdim && "Q must match state dim");
        assert(births.get_dim() == sdim && "Births dim must match state dim");
        assert(H.cols() == sdim && H.rows() == mdim && "H must be (MDIM, SDIM)");
        assert(R.rows() == R.cols() && R.rows() == mdim && "R must be square (MDIM, MDIM)");
        assert(0 <= PD && PD <= 1);
        assert(0 <= PS && PS <= 1);
        assert(0 <= PG && PG <= 1);
    }

    /**
     * @brief Performs the whole PHD recursion
     * 
     * Performs prediction, update and merge step
     * 
     * @param Z Matrix of measurements
     */
    template <typename Derived>
    void step(const Eigen::MatrixBase<Derived>& Z) {
        prediction();
        update(Z);
    }

    /**
     * @brief Performs PHD Prediction
     */
    void prediction() {
        if (t0 && births0.has_value()) {
            phd::prediction(mix, births0.value(), PS, F, Q);
        } else {
            phd::prediction(mix, births, PS, F, Q);
        }
        t0 = false;
    }

    /**
     * @brief Performs PHD Update
     */
    template <typename Derived>
    void update(const Eigen::MatrixBase<Derived>& Z) {
        size_t n_m = mix.size();
        size_t n_z = static_cast<size_t>(Z.cols());
        size_t n_comp = n_m * std::min(n_z, size_t(5));
        mix.reserve(n_m + n_comp);
        update_workers.reserve(n_comp, n_z);
        merge_workers.reserve(n_m + n_comp);

        phd::update(mix, Z, PD, PG, lambda, H, R, update_workers);
        phd::merge(mix, trunc_thr, merge_thr, max_comps, merge_workers);
    }

    /**
     * @brief Returns estimated state
     * 
     * @param type Type of the estimator used
     * 
     * @return (means, covariances) of estimated targets
     */
    auto confirmed_tracks(int type = 1) const {
        if (type != 1)
            throw std::invalid_argument("Unknown estimator type");
        return phd::estimator(mix, conf_thr);
    }

    auto unconfirmed_tracks(int type = 1) const {
        throw std::runtime_error("Not implemented yet");
        return phd::estimator(mix, conf_thr);
    }
    
    /**
     * @brief Resets the tracker
     */
    void reset() {
        mix.clear();
        t0 = true;
    }
};

} // namespace phd

template <typename Scalar, int SDIM=Eigen::Dynamic, int MDIM=Eigen::Dynamic>
using PhdTracker = phd::Tracker<Scalar, SDIM, MDIM>;

} // namespace mtt