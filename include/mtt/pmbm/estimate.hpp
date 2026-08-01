/**
 * @file      estimate.hpp
 * @brief     GM-PMBM Filter estimation
 * @author    @vojtagart
 * @date      1/03/2026
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
#include <tuple>
#include <limits>
#include <Eigen/Core>

#include "../core/core.hpp"
#include "multi_bernoulli_mixture.hpp"


namespace mtt::pmbm {

using Idx = int;

template <typename Scalar>
struct EstimatorWorkers {
    void resize(size_t locals) {
        if (order.size() < locals + 1) {
            order.resize(locals + 1);
            cum_w.resize(locals);
        }
    }
    std::vector<Idx> order;
    std::vector<Scalar> cum_w;
};


template <typename Scalar, int SDIM>
auto estimator_map(const mtt::MultiBernoulliMixture<Scalar, SDIM, Idx>& post, Scalar conf_thr, bool reverse = false) {
    constexpr int COV_ROWS = (SDIM == Eigen::Dynamic ? Eigen::Dynamic : SDIM * SDIM);
    int sdim = post.get_dim();

    auto cmp = [conf_thr, reverse](Scalar r) {
        return reverse ? (r < conf_thr) : (r >= conf_thr);
    };

    if (post.globals.empty()) {
        Eigen::Matrix<Scalar, SDIM, Eigen::Dynamic> mus(sdim, 0);
        Eigen::Matrix<Scalar, COV_ROWS, Eigen::Dynamic> covs(sdim * sdim, 0);
        std::vector<Idx> ids;
        return std::make_tuple(mus, covs, ids);
    }

    size_t mx = 0;
    for (size_t i = 1; i < post.globals.size(); i++) {
        if (post.globals[i].w > post.globals[mx].w)
            mx = i;
    }
    const auto& g = post.globals[mx];

    size_t cnt = 0;
    for (auto x : g.idxs) {
        if (cmp(post.locals.r(x)))
            cnt++;
    }
    Eigen::Matrix<Scalar, SDIM, Eigen::Dynamic> mus(sdim, cnt);
    Eigen::Matrix<Scalar, COV_ROWS, Eigen::Dynamic> covs(sdim*sdim, cnt);
    std::vector<Idx> ids; ids.reserve(cnt);

    size_t idx = 0;
    for (auto x : g.idxs) {
        if (cmp(post.locals.r(x))) {
            mus.col(idx) = post.locals.mu(x);
            Eigen::Map<Eigen::Matrix<Scalar, SDIM, SDIM>>(covs.col(idx).data(), sdim, sdim) = post.locals.cov(x);
            ids.push_back(post.ids[x]);
            idx++;
        }
    }
    return std::make_tuple(std::move(mus), std::move(covs), std::move(ids));
}

template <typename Scalar, int SDIM>
auto estimator_eap(
        const mtt::MultiBernoulliMixture<Scalar, SDIM, Idx>& post, Scalar conf_thr,
        EstimatorWorkers<Scalar>& W, bool reverse = false) {
    constexpr int COV_ROWS = (SDIM == Eigen::Dynamic ? Eigen::Dynamic : SDIM * SDIM);
    int sdim = post.get_dim();

    auto cmp = [conf_thr, reverse](Scalar r) {
        return reverse ? (r < conf_thr) : (r >= conf_thr);
    };

    size_t locals = post.locals.size();
    if (post.globals.empty() || locals == 0) {
        Eigen::Matrix<Scalar, SDIM, Eigen::Dynamic> mus(sdim, 0);
        Eigen::Matrix<Scalar, COV_ROWS, Eigen::Dynamic> covs(sdim * sdim, 0);
        std::vector<Idx> ids;
        return std::make_tuple(mus, covs, ids);
    }

    W.resize(locals);

    std::fill_n(W.cum_w.begin(), locals, 0);
    for (const auto& g : post.globals) {
        Scalar w = g.w;
        for (auto x : g.idxs) {
            W.cum_w[x] += w;
        }
    }
    std::iota(W.order.begin(), W.order.begin() + locals, 0);
    std::sort(W.order.begin(), W.order.begin() + locals, [&](Idx x, Idx y){
        return post.ids[x] < post.ids[y];
    });

    size_t max_tracks = (locals > 0);
    for (size_t i = 1; i < locals; i++) {
        if (post.ids[W.order[i - 1]] != post.ids[W.order[i]])
            max_tracks++;
    }

    size_t cnt = 0;
    Eigen::Matrix<Scalar, SDIM, Eigen::Dynamic> mus(sdim, max_tracks);
    Eigen::Matrix<Scalar, COV_ROWS, Eigen::Dynamic> covs(sdim*sdim, max_tracks);
    std::vector<Idx> ids; ids.reserve(max_tracks);
    auto add_comp = [&](Scalar r, const auto& mu, const auto& cov, Idx id){
        if (!cmp(r)) return;
        mus.col(cnt) = mu;
        Eigen::Map<Eigen::Matrix<Scalar, SDIM, SDIM>>(covs.col(cnt).data(), sdim, sdim) = cov;
        ids.push_back(id);
        cnt++;
    };

    mtt::MixtureToBernoulli<Scalar, SDIM> mtb(post.get_dim());
    size_t ptr = 0, ptr_i = 0, ptr_idx = W.order[0];
    for (size_t i = 1; i <= locals; i++) {
        size_t idx = W.order[i];
        if (i == locals || post.ids[idx] != post.ids[ptr_idx]) {
            // Merging only if there is more than 1 component
            if (i - ptr_i > 1) {
                mtb.add_bern(W.cum_w[ptr_idx], post.locals.r(ptr_idx), post.locals.mu(ptr_idx), post.locals.cov(ptr_idx));
                auto bern = mtb.get_bern(false);
                add_comp(bern.r, bern.mu, bern.cov, post.ids[ptr_idx]);
                mtb.clear();
            } else {
                add_comp(post.locals.r(ptr_idx) * W.cum_w[ptr_idx], post.locals.mu(ptr_idx), post.locals.cov(ptr_idx), post.ids[ptr_idx]);
            }
            W.order[ptr] = ptr_idx;
            ptr_i = i;
            ptr_idx = idx;
            ++ptr;
        // Cumulating the same track
        } else {
            mtb.add_bern(W.cum_w[idx], post.locals.r(idx), post.locals.mu(idx), post.locals.cov(idx));
        }
    }
    mus.conservativeResize(sdim, cnt);
    covs.conservativeResize(sdim*sdim, cnt);
    return std::make_tuple(std::move(mus), std::move(covs), std::move(ids));
}

} // namespace mtt::pmbm