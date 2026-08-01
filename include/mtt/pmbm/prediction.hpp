/**
 * @file      prediction.hpp
 * @brief     GM-PMBM Filter prediction
 * @author    @vojtagart
 * @date      23/03/2026
 * @see       https://github.com/Vojtagart/mtt
 * * @par Credits & Acknowledgments
 * - Algorithm inspired by: Á. F. García-Fernández, J. L. Williams, K. Granström and L. Svensson, "Poisson Multi-Bernoulli Mixture Filter: Direct Derivation and Implementation," in IEEE Transactions on Aerospace and Electronic Systems, vol. 54, no. 4, pp. 1883-1901, Aug. 2018, doi: 10.1109/TAES.2018.2805153
 * - Documentation formatted and refined with the assistance of Google Gemini
 * * @copyright Copyright (c) 2026 @vojtagart. Released under the MIT License.
 */

#pragma once
#include <vector>
#include <algorithm>
#include <utility>
#include <Eigen/Core>

#include "multi_bernoulli_mixture.hpp"
#include "../core/core.hpp"
#include "../phd/phd.hpp"


namespace mtt::pmbm {

using Idx = int;
    
/**
 * @brief Performs a PMBM prediction
 * 
 * @param post_poiss Posterior Poisson component
 * @param post_mbm Posterior Multi-Bernoulli mixture componet
 * @param births_gm Birth Gaussian mixture
 * @param births_mb Birth Multi-Bernoulli
 * @param PS probability of survival
 * @param F Transirtion matrix
 * @param Q Process noise matrix
 */
template <typename DerivedA, typename DerivedB, typename Scalar, int SDIM>
requires mtt::can_be_square<DerivedA> && mtt::can_be_square<DerivedB>
void prediction(mtt::GaussianMixture<Scalar, SDIM>& post_poiss, mtt::MultiBernoulliMixture<Scalar, SDIM, Idx>& post_mbm,
                const mtt::GaussianMixture<Scalar, SDIM>& births_gm, const mtt::MultiBernoulli<Scalar, SDIM>& births_mb,
                Scalar PS, const Eigen::MatrixBase<DerivedA>& F, const Eigen::MatrixBase<DerivedB>& Q) {

    const auto sdim = post_mbm.get_dim();
    assert(F.rows() == F.cols() && F.rows() == sdim && "F must have shape (SDIM, SDIM)");
    assert(Q.rows() == Q.cols() && Q.rows() == sdim && "Q must have shape (SDIM, SDIM)");
    assert(post_poiss.get_dim() == sdim && births_gm.get_dim() == sdim && births_mb.get_dim() == sdim && "Mixtures dims must match");
    
    assert(post_mbm.locals.size() == post_mbm.ids.size());
    assert(post_mbm.locals.size() == post_mbm.last_meas.size());

    #ifndef NDEBUG
    for (const auto& g : post_mbm.globals) {
        for (auto x : g.idxs) assert(x < static_cast<Idx>(post_mbm.locals.size()));
    }
    #endif
    
    // Poisson prediction
    phd::prediction(post_poiss, births_gm, PS, F, Q);
    
    // MBM prediction

    auto& locals = post_mbm.locals;
    size_t orig_comps = locals.size();
    locals.scale_exist_prob(PS);

    if (post_mbm.globals.empty())
        post_mbm.add_global(1., {});
    
    if (births_mb.size() > 0) {
        // Update global hypothesis - now they also consider birth components
        for (auto& g : post_mbm.globals) {
            g.idxs.reserve(g.idxs.size() + births_mb.size());
            // the global hypothesis remains sorted - useful for cache hits
            for (size_t i = 0; i < births_mb.size(); i++) {
                g.idxs.push_back(static_cast<Idx>(i + post_mbm.locals.size()));
            }
        }
        // Adding the birth components if any
        for (size_t i = 0; i < births_mb.size(); i++) {
            post_mbm.add_comp(births_mb.r(i), births_mb.mu(i), births_mb.cov(i));
        }
    }

    for (size_t i = 0; i < orig_comps; i++) {
        auto mu = locals.mu(i);
        mtt::kf_predict_mu(mu, F);
    }
    for (size_t i = 0; i < orig_comps; i++) {
        auto cov = locals.cov(i);
        mtt::kf_predict_cov(cov, F, Q);
    }

    assert(post_mbm.locals.size() == post_mbm.ids.size());
    assert(post_mbm.locals.size() == post_mbm.last_meas.size());
}

} // namespace mtt::pmbm