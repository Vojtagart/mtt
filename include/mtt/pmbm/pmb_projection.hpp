/**
 * @file      pmb_projection.hpp
 * @brief     MBM to MB by minimizing KLD
 * @author    @vojtagart
 * @date      10/03/2026
 * @see       https://github.com/Vojtagart/mtt
 * * @par Credits & Acknowledgments
 * - Algorithm inspired by: J. L. Williams, "An Efficient, Variational Approximation of the Best Fitting Multi-Bernoulli Filter," in IEEE Transactions on Signal Processing, vol. 63, no. 1, pp. 258-273, Jan.1, 2015, doi: 10.1109/TSP.2014.2370946
 * - Documentation formatted and refined with the assistance of Google Gemini
 * * @copyright Copyright (c) 2026 @vojtagart. Released under the MIT License.
 */

#pragma once
#include <vector>
#include <algorithm>
#include <utility>
#include <numeric>

#include "../core/core.hpp"
#include "multi_bernoulli_mixture.hpp"


namespace mtt::pmbm {

using Idx = int;


template <typename Scalar, int DIM=Eigen::Dynamic>
struct PmbProjectionWorkers {

    void resize(size_t locals, size_t max_id) {
        if (order.size() < locals + 1) {
            order.resize(locals + 1);
            cum_w.resize(locals);
        }
        if (norm_fact.size() < max_id) {
            norm_fact.resize(max_id);
        }
    }

    std::vector<Idx> order;
    std::vector<Scalar> cum_w, norm_fact;
    std::vector<uint8_t> is_max;
};


template <typename Scalar, int SDIM>
size_t project_to_tracks(mtt::MultiBernoulliMixture<Scalar, SDIM, Idx>& mbm, PmbProjectionWorkers<Scalar, SDIM>& W) {

    size_t locals = mbm.locals.size();

    // Grouping the same track local hypothesis next to each other
    std::sort(W.order.begin(), W.order.begin() + locals, [&](Idx x, Idx y){
        return mbm.ids[x] < mbm.ids[y];
    });

    mtt::MixtureToBernoulli<Scalar, SDIM> mtb(mbm.get_dim());
    size_t ptr = 0, ptr_i = 0, ptr_idx = W.order[0];
    for (size_t i = 1; i <= locals; i++) {
        size_t idx = (i < locals) ? W.order[i] : 0;
        if (i == locals || mbm.ids[idx] != mbm.ids[ptr_idx]) {
            // Merging only if there is more than 1 component
            if (i - ptr_i > 1) {
                mtb.add_bern(W.cum_w[ptr_idx], mbm.locals.r(ptr_idx), mbm.locals.mu(ptr_idx), mbm.locals.cov(ptr_idx));
                auto bern = mtb.get_bern(false);
                mbm.locals.set(ptr_idx, bern.r, bern.mu, bern.cov);
                mtb.clear();
            } else {
                mbm.locals.r(ptr_idx) *= W.cum_w[ptr_idx];
            }
            W.order[ptr] = ptr_idx;
            ptr_i = i;
            ptr_idx = idx;
            ++ptr;
        // Cumulating the same track
        } else {
            mtb.add_bern(W.cum_w[idx], mbm.locals.r(idx), mbm.locals.mu(idx), mbm.locals.cov(idx));
        }
    }
    return ptr;
}

template <typename Scalar, int SDIM>
size_t project_to_meas(mtt::MultiBernoulliMixture<Scalar, SDIM, Idx>& mbm, PmbProjectionWorkers<Scalar, SDIM>& W, Scalar alpha = 0) {

    size_t locals = mbm.locals.size();
    if (locals == 0) return 0;

    if (alpha > Scalar(0)) {
        std::sort(W.order.begin(), W.order.begin() + locals, [&](Idx x, Idx y){
            return mbm.ids[x] < mbm.ids[y];
        });
        W.is_max.assign(locals, false);
        
        size_t ptr_idx = W.order[0], max_idx = ptr_idx;
        for (size_t i = 1; i <= locals; i++) {
            size_t idx = (i < locals) ? W.order[i] : 0;
            if (i == locals || mbm.ids[idx] != mbm.ids[ptr_idx]) {
                W.is_max[max_idx] = true;
                ptr_idx = max_idx = idx;
            } else {
                if (W.cum_w[idx] > W.cum_w[max_idx])
                    max_idx = idx;
            }
        }
    }

    std::fill_n(W.norm_fact.begin(), mbm.cur_id, Scalar(1));
    // Grouping the same track local hypothesis next to each other
    std::sort(W.order.begin(), W.order.begin() + locals, [&](Idx x, Idx y){
        if (mbm.last_meas[y] == mbm.UNDEF) return false;
        if (mbm.last_meas[x] == mbm.UNDEF) return true;
        return mbm.last_meas[x] < mbm.last_meas[y];
    });

    size_t ptr = 0;
    // Skipping the miss detections
    while (ptr < locals && mbm.last_meas[W.order[ptr]] == mbm.UNDEF) {
        Idx idx = W.order[ptr];
        Scalar p = W.cum_w[idx];

        if (alpha > Scalar(0)) {
            // sum-product missed det prob/ weight
            Scalar p_sum = W.cum_w[idx];
            // max-product missed der prob/ weight
            Scalar p_max = (W.is_max[idx] ? 1 : 0);
            p = alpha * p_max + (1 - alpha) * p_sum;
            Scalar norm_fact = (p_sum < 1.0 - 1e-6) ? (1 - p) / (1 - p_sum) : 1;
            W.norm_fact[mbm.ids[idx]] = norm_fact;
        }
        mbm.locals.r(idx) *= p;
        ptr++;
    }

    // All misdetection hypotheses
    if (ptr == locals)
        return ptr;

    mtt::MixtureToBernoulli<Scalar, SDIM> mtb(mbm.get_dim());
    size_t ptr_i = ptr, ptr_idx = W.order[ptr];
    for (size_t i = ptr + 1; i <= locals; i++) {
        size_t idx = W.order[i];
        if (i == locals || mbm.last_meas[idx] != mbm.last_meas[ptr_idx]) {
            Scalar w = W.cum_w[ptr_idx] * W.norm_fact[mbm.ids[ptr_idx]];
            // Merging only if there is more than 1 component
            if (i - ptr_i > 1) {
                mtb.add_bern(w, mbm.locals.r(ptr_idx), mbm.locals.mu(ptr_idx), mbm.locals.cov(ptr_idx));
                auto bern = mtb.get_bern(false);
                mbm.locals.set(ptr_idx, bern.r, bern.mu, bern.cov);
                mtb.clear();
            } else {
                mbm.locals.r(ptr_idx) *= w;
            }
            W.order[ptr] = ptr_idx;
            ptr_i = i;
            ptr_idx = idx;
            ++ptr;
        // Cumulating the same measurement
        } else {
            Scalar w = W.cum_w[idx] * W.norm_fact[mbm.ids[idx]];
            mtb.add_bern(w, mbm.locals.r(idx), mbm.locals.mu(idx), mbm.locals.cov(idx));
        }
    }
    return ptr;
}

/**
 * @brief Approximates MBM is MB
 * 
 * @param mbm Multi-Bernoulli mixture
 * @param W PmbProjectionWorkers
 */
template <typename Scalar, int SDIM>
void pmb_projection(mtt::MultiBernoulliMixture<Scalar, SDIM, Idx>& mbm, bool use_id, PmbProjectionWorkers<Scalar, SDIM>& W, Scalar alpha = 0) {

    assert(alpha == Scalar(0) || !use_id);

    size_t locals = mbm.locals.size();

    if (locals == 0) {
        mbm.clear_globals();
        mbm.add_global(1., {});
        return;
    }
    assert(!mbm.globals.empty());

    W.resize(locals, (use_id ? 0 : mbm.cur_id));

    std::fill_n(W.cum_w.begin(), locals, 0);
    for (const auto& g : mbm.globals) {
        Scalar w = g.w;
        for (auto x : g.idxs) {
            W.cum_w[x] += w;
        }
    }
    // Needs to be sorted inside the specialized project functions
    std::iota(W.order.begin(), W.order.begin() + locals, 0);

    size_t ptr;
    if (use_id)
        ptr = project_to_tracks(mbm, W);
    else
        ptr = project_to_meas(mbm, W, alpha);

    // Sorting the stored components so we can easily move them to the front
    // without overwritting other stored components
    std::sort(W.order.begin(), W.order.begin() + ptr);
    for (size_t i = 0; i < ptr; i++) {
        size_t idx = W.order[i];
        if (i != idx)
            mbm.locals.set(i, mbm.locals.r(idx), mbm.locals.mu(idx), mbm.locals.cov(idx));
        if (use_id) {
            mbm.ids[i] = mbm.ids[idx];
            // for TOMB, last_meas becomes invalidated
            mbm.last_meas[i] = mbm.UNDEF;
        } else {
            // for MOMB, reset the ids
            mbm.ids[i] = i;
            mbm.last_meas[i] = mbm.last_meas[idx];
        }
    }
    mbm.resize_locals(ptr);
    // fix for the MOMB that alters the ids
    // without this, W.norm_factor will grow infinitelly. Since the ids no longer have any
    // prior information as we re-labeled the hypothesis, we can also change the cur_id and the
    // id invariant will be kept as it was before
    if (!use_id)
        mbm.cur_id = ptr;

    mbm.clear_globals();
    std::vector<Idx> tmp(ptr);
    std::iota(tmp.begin(), tmp.end(), 0);
    mbm.add_global(1., std::move(tmp));
}

} // namespace mtt::pmbm