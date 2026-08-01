/**
 * @file      prune.hpp
 * @brief     GM-PMBM Filter prune implementation
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
#include <span>
#include <limits>
#include <Eigen/Core>
#include <murty/binary_heap.hpp>

#include "multi_bernoulli_mixture.hpp"
#include "../core/core.hpp"


namespace mtt::pmbm {

using Idx = int;

template <typename Scalar, int SDIM>
struct PruneWorkers {
    PruneWorkers() = default;

    std::vector<uint8_t> used;
    std::vector<Idx> psm;
    std::vector<Scalar> cum_w;

    std::vector<Idx> order;
    std::vector<Idx> mapping;

    std::vector<std::pair<Idx, Idx>> vals;
    std::vector<Idx> last_updt;
    murty::BinaryHeap<std::pair<Scalar, Idx>> q;
    std::vector<Eigen::LLT<Eigen::Matrix<Scalar, SDIM, SDIM>>> llts;
    std::vector<Scalar> logdets;

    void resize(size_t comps) {
        if (psm.size() < comps + 1) {
            used.resize(comps);
            psm.resize(comps + 1, 0);
        }
    }
    void prepare(size_t comps) {
        resize(comps);
        std::fill_n(used.begin(), comps, false);
    }

    void resize_merging(size_t comps) {
        if (last_updt.size() < comps) {
            last_updt.resize(comps);
            llts.resize(comps);
            logdets.resize(comps);
        }
    }
};

template <typename Scalar, int SDIM>
size_t merge_close_locals(mtt::MultiBernoulliMixture<Scalar, SDIM, Idx>& mbm, size_t locals, PruneWorkers<Scalar, SDIM>& W, Scalar merge_thr) {
    constexpr Idx MAX = std::numeric_limits<Idx>::max();

    auto get_llt_det = [&](size_t i, size_t idx){
        W.llts[i] = mbm.locals.cov(idx).llt();
        if (W.llts[i].info() != Eigen::Success)
            throw std::runtime_error("LLT failed");
        Scalar logdet = 0;
        // Determinnant can be calculated from lower cholesky as product
        // over diagonal to the power of 2
        const auto& L = W.llts[i].matrixL();
        for (int j = 0; j < mbm.get_dim(); j++) {
            logdet += std::log(L(j, j));
        }
        W.logdets[i] = 2 * logdet;
    };
    auto recalc_dist = [&](size_t i, size_t st, size_t n, size_t from){
        Idx p1 = W.order[st + i];
        for (size_t j = from; j < n; j++) {
            if (j == i || W.last_updt[j] == MAX) continue;
            Idx p2 = W.order[st + j];
            Scalar dist1 = kld_dist(mbm.locals.r(p1), mbm.locals.mu(p1), mbm.locals.cov(p1), W.logdets[i],
                                    mbm.locals.r(p2), mbm.locals.mu(p2), W.llts[j], W.logdets[j]);
            Scalar dist2 = kld_dist(mbm.locals.r(p2), mbm.locals.mu(p2), mbm.locals.cov(p2), W.logdets[j],
                                    mbm.locals.r(p1), mbm.locals.mu(p1), W.llts[i], W.logdets[i]);
            Scalar dist = Scalar(0.5) * (dist1 + dist2);
            if (dist < merge_thr) {
                W.q.emplace(dist, static_cast<Idx>(W.vals.size()));
                W.vals.emplace_back(i, j);
            }
        }
    };

    // Merging similar components in each track
    mtt::MixtureToBernoulli<Scalar, SDIM> mtb(mbm.get_dim());
    size_t ptr = 0;
    size_t st = 0, ed = 1;
    while (st < locals) {
        // extracting start (inclusive) and end (exclusive) index of current track
        while (ed < locals && mbm.ids[W.order[ed]] == mbm.ids[W.order[st]]) {
            ed++;
        }
        size_t n = ed - st;
        W.resize_merging(n);
        W.q.clear();
        W.vals.clear();

        // Precalculating LLT's for faster calculation
        for (size_t i = 0; i < n; i++) {
            get_llt_det(i, W.order[i + st]);
        }

        // Precalculating all the distances, storing them inside a binary heap
        for (size_t i = 0; i < n; i++) {
            W.last_updt[i] = 0;
            recalc_dist(i, st, n, i + 1);
        }

        // Greedily merging two closest components
        while (!W.q.empty()) {
            Idx idx = W.q.top().second;
            W.q.pop();
            auto [x, y] = W.vals[idx];
            if (W.last_updt[x] > idx || W.last_updt[y] > idx) continue;

            // Moment match them
            mtb.clear();
            Idx p1 = W.order[x + st], p2 = W.order[y + st];
            mtb.add_bern(W.cum_w[p1], mbm.locals.r(p1), mbm.locals.mu(p1), mbm.locals.cov(p1));
            mtb.add_bern(W.cum_w[p2], mbm.locals.r(p2), mbm.locals.mu(p2), mbm.locals.cov(p2));

            // Store them inside p1
            W.cum_w[p1] += W.cum_w[p2];
            W.mapping[p2] = p1;
            auto bern = mtb.get_bern(true);
            mbm.locals.set(p1, bern.r, bern.mu, bern.cov);

            // banning all the distances to the original component
            W.last_updt[x] = W.vals.size();
            // effectively banning this component, marking it as deleted
            W.last_updt[y] = MAX;

            get_llt_det(x, p1);
            recalc_dist(x, st, n, 0);
        }

        // Moving the left hypothesis to the beginning
        for (size_t i = 0; i < n; i++) {
            if (W.last_updt[i] != MAX) {
                W.order[ptr] = W.order[st + i];
                ptr++;
            }
        }

        // One now could decide if this track is divergent or not
        // Current distances are stored in W.vals + W.q, where the valid
        // distances index is >= W.last_updt for both of its edges

        st = ed;
        ed++;
    }
    return ptr;
}

template <typename Scalar, int SDIM>
void merge_locals(mtt::MultiBernoulliMixture<Scalar, SDIM, Idx>& mbm, PruneWorkers<Scalar, SDIM>& W, Scalar merge_thr = 0) {

    size_t locals = mbm.locals.size();
    if (locals == 0) return;

    // resize workers
    if (W.order.size() < locals + 1) {
        W.cum_w.resize(locals);
        W.order.resize(locals + 1);
        W.mapping.resize(locals);
    }

    std::fill_n(W.cum_w.begin(), locals, Scalar(0));
    for (const auto& g : mbm.globals) {
        Scalar w = g.w;
        for (auto x : g.idxs) {
            W.cum_w[x] += w;
        }
    }

    std::iota(W.order.begin(), W.order.begin() + locals, 0);
    std::sort(W.order.begin(), W.order.begin() + locals, [&](int x, int y){
        return std::tie(mbm.ids[x], mbm.last_meas[x]) < std::tie(mbm.ids[y], mbm.last_meas[y]);
    });
    // Initialize mapping as identity
    std::iota(W.mapping.begin(), W.mapping.begin() + locals, 0);

    // Merging the same track-measurement associations into one
    // There will be ptr of them, their indices stored in W.order[0], ..., W.order[ptr-1]
    mtt::MixtureToBernoulli<Scalar, SDIM> mtb(mbm.get_dim());
    size_t ptr = 0, ptr_i = 0, ptr_idx = W.order[0];
    for (size_t i = 1; i <= locals; i++) {
        size_t idx = W.order[i];
        // Cumulating the same track-measurement association
        // We have to treat two miss-detection hypothesis as different ones
        if (i == locals || mbm.ids[idx] != mbm.ids[ptr_idx] || mbm.last_meas[idx] != mbm.last_meas[ptr_idx] || mbm.last_meas[idx] == mbm.UNDEF) {
            // Merging only if there is more then 1 component
            if (i - ptr_i > 1) {
                mtb.add_bern(W.cum_w[ptr_idx], mbm.locals.r(ptr_idx), mbm.locals.mu(ptr_idx), mbm.locals.cov(ptr_idx));
                auto bern = mtb.get_bern(true);
                mbm.locals.set(ptr_idx, bern.r, bern.mu, bern.cov);
                W.cum_w[ptr_idx] = mtb._w;
                mtb.clear();
            }
            W.order[ptr] = ptr_idx;
            ptr_i = i;
            ptr_idx = idx;
            ++ptr;
        } else {
            mtb.add_bern(W.cum_w[idx], mbm.locals.r(idx), mbm.locals.mu(idx), mbm.locals.cov(idx));
            W.mapping[idx] = ptr_idx;
            // Just to be sure
            W.cum_w[idx] = 0;
        }
    }
    locals = ptr;

    if (merge_thr > Scalar(0))
        locals = merge_close_locals(mbm, locals, W, merge_thr);

    // Fix mapping orig local -> new local
    // with path compression - maybe unnecesarly?
    for (size_t i = 0; i < mbm.locals.size(); i++) {
        Idx idx = i;
        while (W.mapping[idx] != idx) {
            idx = W.mapping[idx];
        }
        Idx cur = i;
        while (W.mapping[cur] != cur) {
            Idx prev = cur;
            cur = W.mapping[cur];
            W.mapping[prev] = idx;
        }
    }

    // Reindex the global hypothesis
    for (auto& gh : mbm.globals) {
        for (auto& x : gh.idxs) {
            x = W.mapping[x];
        }
    }
}

/**
 * @brief Performs pruning of PMBM
 * 
 * @param poiss Poisson component
 * @param mbm Multi-Bernoulli mixture componet
 * @param min_hypot_w  Minium hypothesis weight
 * @param min_bern_r Minium Bernoulli component (local hypothesis) eixstence probability
 * @param min_poiss_w Minium weight of Gaussian component from Poisson
 * @param W Prune Workers
 */
template <typename Scalar, int SDIM>
void prune(
        mtt::GaussianMixture<Scalar, SDIM>& post_poiss, mtt::MultiBernoulliMixture<Scalar, SDIM, Idx>& post_mbm,
        Scalar min_hypot_w, Scalar min_bern_r, Scalar min_poiss_w, PruneWorkers<Scalar, SDIM>& W, bool recyclate = false,
        bool merge_components = false, Scalar merge_thr = 0.0) {

    assert(post_mbm.locals.size() == post_mbm.ids.size());
    assert(post_mbm.locals.size() == post_mbm.last_meas.size());

    #ifndef NDEBUG
    for (const auto& g : post_mbm.globals) {
        for (auto x : g.idxs) assert(x < static_cast<Idx>(post_mbm.locals.size()));
    }
    #endif
    
    size_t comps = post_mbm.locals.size();
    W.prepare(comps);

    // Prune Poisson
    post_poiss.filter_out(min_poiss_w);
    // phd::merge(post_poisson, min_poiss_w, 6, 1000);
    
    // Prune MBM

    // Pruning low weight global hypothesis
    post_mbm.filter_globals(min_hypot_w);

    if (post_mbm.globals.empty()) {
        post_mbm.clear_locals();
        post_mbm.add_global(1., {});
        return;
    }

    if (merge_components)
        merge_locals(post_mbm, W, merge_thr);

    // Mark used components
    for (const auto& g : post_mbm.globals) {
        for (auto x : g.idxs) {
            W.used[x] = true;
        }
    }

    // Recyclate if wanted
    if (recyclate) {
        if (W.cum_w.size() < comps)
            W.cum_w.resize(comps);
        std::fill_n(W.cum_w.begin(), comps, Scalar(0));
        for (const auto& g : post_mbm.globals) {
            Scalar w = g.w;
            for (auto x : g.idxs) {
                W.cum_w[x] += w;
            }
        }
        for (size_t i = 0; i < comps; i++) {
            if (W.used[i] && post_mbm.locals.r(i) < min_bern_r) {
                W.used[i] = false;
                Scalar nw = W.cum_w[i] * post_mbm.locals.r(i);
                if (nw >= min_poiss_w)
                    post_poiss.push(nw, post_mbm.locals.mu(i), post_mbm.locals.cov(i));
            }
        }
    } else {
        // Mark low existence probability ones as unused
        for (size_t i = 0; i < comps; i++) {
            if (post_mbm.locals.r(i) < min_bern_r)
                W.used[i] = false;
        }
    }
    // Removing locals, recalculating indexes
    post_mbm.filter_locals(W.used, W.psm);

    // Merging the same hypothesis into one
    for (auto& gh : post_mbm.globals) {
        std::sort(gh.idxs.begin(), gh.idxs.end());
    }
    std::sort(post_mbm.globals.begin(), post_mbm.globals.end(), [](const auto& x, const auto& y){
        return x.idxs < y.idxs;
    });

    size_t ptr = 0;
    for (size_t i = 1; i < post_mbm.globals.size(); i++) {
        if (post_mbm.globals[i].idxs == post_mbm.globals[ptr].idxs) {
            post_mbm.globals[ptr].w += post_mbm.globals[i].w;
        } else {
            ++ptr;
            if (ptr != i)
                post_mbm.globals[ptr] = std::move(post_mbm.globals[i]);
        }
    }
    post_mbm.resize_globals(ptr + 1);
    post_mbm.normalize_globals();
    assert(post_mbm.locals.size() == post_mbm.ids.size());
    assert(post_mbm.locals.size() == post_mbm.last_meas.size());
}

} // namespace mtt::pmbm