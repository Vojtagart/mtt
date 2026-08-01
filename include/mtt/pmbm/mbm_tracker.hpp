/**
 * @file      mbm_tracker.hpp
 * @brief     Wrapper class for the GM-MBM filter
 * @author    @vojtagart
 * @date      9/03/2026
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
#include <stdexcept>
#include <Eigen/Core>

#include "multi_bernoulli_mixture.hpp"
#include "prediction.hpp"
#include "update.hpp"
#include "prune.hpp"
#include "estimate.hpp"
#include "pmbm_tracker.hpp"
#include "../core/core.hpp"


namespace mtt {
namespace pmbm {

template <typename Scalar>
using MbmConfig = Config<Scalar>;

/**
 * @brief Wrapper for the MBM recursion
 * 
 * @tparam Scalar Value type
 * @tparam SDIM Dimension of the state (int or Eigen::Dynamic)
 * @tparam MDIM Dimension of the measurement (int or Eigen::Dynamic)
 */
template <typename Scalar, int SDIM=Eigen::Dynamic, int MDIM=Eigen::Dynamic>
struct MbmTracker {

    using StateT = Eigen::Matrix<Scalar, SDIM, 1>;
    using MeasT = Eigen::Matrix<Scalar, MDIM, 1>;
    using StateCovT = Eigen::Matrix<Scalar, SDIM, SDIM>;
    using MeasCovT = Eigen::Matrix<Scalar, MDIM, MDIM>;
    using HT = Eigen::Matrix<Scalar, MDIM, SDIM>;
    using GMT = mtt::GaussianMixture<Scalar, SDIM>;
    using MBT = mtt::MultiBernoulli<Scalar, SDIM>;
    using MBMT = mtt::MultiBernoulliMixture<Scalar, SDIM>;

    GMT dummy_gm;
    MBT births;
    std::optional<MBT> births0;
    GMT dummy_poiss;
    MBMT mbm;

    StateCovT F;
    StateCovT Q;
    HT H;
    MeasCovT R;

    Scalar PD;
    Scalar PS;
    Scalar lambda;
    MbmConfig<Scalar> config;

    UpdateWorkers<Scalar, SDIM, MDIM> update_workers;
    PruneWorkers<Scalar, SDIM> prune_workers;
    mutable EstimatorWorkers<Scalar> estimator_workers;

    bool t0 = true;

    /**
     * @brief Initalizes MbmTracker with given parameters
     * 
     * @param births Birth Multi-Bernoulli
     * @param F Transition matrix
     * @param Q Process noise
     * @param H Measurement matrix
     * @param R Measurement noise
     * @param PD Probability of detection
     * @param PS Probability of survival
     * @param lambda Clutter intensity
     * @param config Config for the filter
     * @param _births0 Override of the initial birth mixture
     */
    MbmTracker(const MBT& births, const StateCovT& F, const StateCovT& Q, const HT& H, const MeasCovT& R,
            Scalar PD, Scalar PS, Scalar lambda, MbmConfig<Scalar> config, std::optional<MBT> _births0 = {})
            : dummy_gm(F.rows()), births(births), births0(std::move(_births0)), dummy_poiss(F.rows()), mbm(F.rows()), F(F), Q(Q),
              H(H), R(R), PD(PD), PS(PS), lambda(lambda), config(std::move(config)), update_workers(H.rows()) {
        const auto sdim = F.rows();
        const auto mdim = H.rows();
        assert(F.rows() == F.cols() && "F must be square");
        assert(Q.rows() == Q.cols() && Q.rows() == sdim && "Q must match state dim");
        assert(births.get_dim() == sdim && "Births dim must match state dim");
        assert(H.cols() == sdim && H.rows() == mdim && "H must be (MDIM, SDIM)");
        assert(R.rows() == R.cols() && R.rows() == mdim && "R must be square (MDIM, MDIM)");
        assert(0 <= PD && PD <= 1);
        assert(0 <= PS && PS <= 1);
        assert(0 <= config.PG && config.PG <= 1);
    }

    /**
     * @brief Performs the whole MBM recursion
     * 
     * Performs prune -> prediction -> update
     * 
     * @param Z Matrix of measurements
     */
    template <typename Derived>
    void step(const Eigen::MatrixBase<Derived>& Z) {
        prune();
        prediction();
        update(Z);
    }

    /**
     * @brief Performs MBM Prediction
     */
    void prediction() {
        if (t0 && births0.has_value()) {
            pmbm::prediction(dummy_poiss, mbm, dummy_gm, births0.value(), PS, F, Q);
        } else {
            pmbm::prediction(dummy_poiss, mbm, dummy_gm, births, PS, F, Q);
        }
        t0 = false;
    }

    /**
     * @brief Performs MBM Update
     * 
     * @param Z Matrix of measurements
     */
    template <typename Derived>
    void update(const Eigen::MatrixBase<Derived>& Z) {
        assert(Z.rows() == H.rows() && "Measurement dimension mismatch");
        pmbm::update(dummy_poiss, mbm, Z, PD, config.PG, lambda, config.max_hypots, H, R, update_workers, config.sparsify, config.max_per_row);
    }

    /**
     * @brief Performs MBM Pruning
     */
    void prune() {
        pmbm::prune(dummy_poiss, mbm, config.min_hypot_w, config.min_bern_r, config.min_poiss_w, prune_workers,
                    false, config.merge_comps, config.merge_thr);
    }

    /**
     * @brief Returns estimated state
     * 
     * @param type Type of the estimator used
     * 
     * @return (means, covariances, ids) of estimated targets
     */
    auto confirmed_tracks(int type = 1) const {
        if (type == 1)
            return pmbm::estimator_map<Scalar, SDIM>(mbm, config.conf_thr);
        else if (type == 2)
            return pmbm::estimator_eap<Scalar, SDIM>(mbm, config.conf_thr, estimator_workers);
        else
            throw std::invalid_argument("Unknown estimator type");
    }

    auto unconfirmed_tracks(int type = 1) const {
        if (type == 1)
            return pmbm::estimator_map<Scalar, SDIM>(mbm, config.conf_thr, true);
        else if (type == 2)
            return pmbm::estimator_eap<Scalar, SDIM>(mbm, config.conf_thr, estimator_workers, true);
        else
            throw std::invalid_argument("Unknown estimator type");
    }

    void reset() {
        mbm.clear();
        t0 = true;
    }
};

} // namespace pmbm

template <typename Scalar, int SDIM=Eigen::Dynamic, int MDIM=Eigen::Dynamic>
using MbmTracker = pmbm::MbmTracker<Scalar, SDIM, MDIM>;

} // namespace mtt
