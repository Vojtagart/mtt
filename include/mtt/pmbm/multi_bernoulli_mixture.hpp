/**
 * @file      multi_bernoulli_mixture.hpp
 * @brief     Multi-Bernoulli mixture implementation
 * @author    @vojtagart
 * @date      8/03/2026
 * @see       https://github.com/Vojtagart/mtt
 * * @par Credits & Acknowledgments
 * - Documentation formatted and refined with the assistance of Google Gemini
 * * @copyright Copyright (c) 2026 @vojtagart. Released under the MIT License.
 */

#pragma once
#include <vector>
#include <algorithm>
#include <utility>
#include <cassert>
#include <cstddef>
#include <Eigen/Core>

#include "../core/multi_bernoulli.hpp"


namespace mtt {

/**
 * @brief Container representing a Multi-Bernoulli mixture
 * 
 * @tparam Scalar Type used for scalars
 * @tparam DIM Dimension of Gaussians
 */
template <typename Scalar, int DIM=Eigen::Dynamic, typename Idx=int>
struct MultiBernoulliMixture {

    constexpr static Idx UNDEF = -Idx(1);

    struct GlobalHypot {
        Scalar w = Scalar(0);
        std::vector<Idx> idxs;
        GlobalHypot() = default;
        GlobalHypot(Scalar w, const std::vector<Idx>& idxs) : w(w), idxs(idxs) {}
        GlobalHypot(Scalar w, std::vector<Idx>&& idxs) noexcept : w(w), idxs(std::move(idxs)) {}
        [[nodiscard]] constexpr Idx& operator [] (size_t idx) {assert(idx < idxs.size()); return idxs[idx];}
        [[nodiscard]] constexpr Idx operator [] (size_t idx) const {assert(idx < idxs.size()); return idxs[idx];}
        [[nodiscard]] constexpr size_t size() const noexcept {return idxs.size();};
    };

    std::vector<GlobalHypot> globals;
    mtt::MultiBernoulli<Scalar, DIM> locals;
    std::vector<Idx> ids;
    std::vector<Idx> last_meas;

    Idx cur_id = 0;

    MultiBernoulliMixture(int dim) : locals(dim) {}

    /**
     * @return Dimension of the Bernoulli componets
     */
    [[nodiscard]] int get_dim() const noexcept {
        return locals.get_dim();
    }

    template <typename DerivedA, typename DerivedB>
    requires (mtt::is_col_vector<DerivedA> && mtt::can_be_square<DerivedB>)
    size_t add_comp(Scalar r, const Eigen::MatrixBase<DerivedA>& mu, const Eigen::MatrixBase<DerivedB>& cov, Idx meas_idx = UNDEF) {
        return add_local(r, mu, cov, cur_id++, meas_idx);
    }
    template <typename DerivedA, typename DerivedB>
    requires (mtt::is_col_vector<DerivedA> && mtt::can_be_square<DerivedB>)
    size_t add_local(Scalar r, const Eigen::MatrixBase<DerivedA>& mu, const Eigen::MatrixBase<DerivedB>& cov, Idx id, Idx meas_idx = UNDEF) {
        size_t ret = locals.size();
        locals.push(r, mu, cov);
        ids.push_back(id);
        last_meas.push_back(meas_idx);
        return ret;
    }
    void add_global(Scalar w, const std::vector<int>& idxs) {
        globals.emplace_back(w, idxs);
    }
    void add_global(Scalar w, std::vector<int>&& idxs) {
        globals.emplace_back(w, std::move(idxs));
    }

    void filter_globals(Scalar min_w) {
        size_t ptr = 0;
        for (size_t i = 0; i < globals.size(); i++) {
            if (globals[i].w < min_w) continue;
            if (i != ptr)
                globals[ptr] = std::move(globals[i]);
            ptr++;
        }
        globals.resize(ptr);
        normalize_globals();
    }
    void filter_locals(const std::vector<uint8_t>& mask, std::vector<Idx>& psm) {
        assert(locals.size() <= mask.size() && locals.size() < psm.size());
        // Removing unused locals
        psm[0] = 0;
        for (size_t i = 0; i < locals.size(); i++) {
            size_t idx = psm[i];
            psm[i + 1] = psm[i] + mask[i];
            if (mask[i] && idx != i) {
                locals.r(idx) = locals.r(i);
                locals.mu(idx) = locals.mu(i);
                locals.cov(idx) = locals.cov(i);
                ids[idx] = ids[i];
                last_meas[idx] = last_meas[i];
            }
        }
        size_t nsize = psm[locals.size()];
        resize_locals(nsize);

        for (auto& g : globals) {
            size_t ptr = 0;
            for (size_t i = 0; i < g.size(); i++) {
                if (!mask[g[i]]) continue;
                g[ptr] = psm[g[i]];
                ptr++;
            }
            g.idxs.resize(ptr);
        }
    }
    void filter_locals(const std::vector<uint8_t>& mask) {
        std::vector<Idx> psm(locals.size() + 1);
        filter_locals(mask, psm);
    }
    void normalize_globals() {
        Scalar sm = 0;
        for (const auto& g : globals) {
            sm += g.w;
        }
        if (sm <= 10 * std::numeric_limits<Scalar>::epsilon()) {
            for (auto& g : globals) {
                g.w = Scalar{1} / globals.size();
            }
        } else {
            for (auto& g : globals) {
                g.w /= sm;
            }
        }
    }
    void log_sum_exp_norm_globals() {
        if (globals.empty()) return;
        Scalar max_w = globals[0].w;
        for (const auto& g : globals) {
            max_w = std::max(max_w, g.w);
        }
        Scalar sm = 0;
        for (auto& g : globals) {
            g.w = std::exp(g.w - max_w);
            sm += g.w;
        }
        if (sm <= 10 * std::numeric_limits<Scalar>::epsilon()) {
            for (auto& g : globals) {
                g.w = Scalar{1} / globals.size();
            }
        } else {
            for (auto& g : globals) {
                g.w /= sm;
            }
        }
    }

    void clear_globals() {
        globals.clear();
    }
    void clear_locals() {
        locals.clear();
        ids.clear();
        last_meas.clear();
    }
    void clear() {
        clear_globals();
        clear_locals();
        cur_id = 0;
    }

    void resize_globals(size_t nsize) {
        globals.resize(nsize);
    }
    void resize_locals(size_t nsize) {
        locals.resize(nsize);
        ids.resize(nsize);
        last_meas.resize(nsize);
    }
};

} // namesapce mtt