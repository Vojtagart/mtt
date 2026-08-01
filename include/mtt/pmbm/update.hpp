/**
 * @file      update.hpp
 * @brief     GM-PMBM Filter update
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
#include <tuple>
#include <limits>
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <murty/solve.hpp>

#include "../pmbm/multi_bernoulli_mixture.hpp"
#include "../core/core.hpp"


namespace mtt::pmbm {

using Idx = int;

/**
 * @brief Workers used for PMBM update
 * 
 * @tparam Scalar Value type
 * @tparam SDIM State dimension
 * @tparam MDIM Measurement dimension
 */
template <typename Scalar, int SDIM=Eigen::Dynamic, int MDIM=Eigen::Dynamic>
struct UpdateWorkers {
    UpdateWorkers(int mdim = MDIM)
            : gater(mdim), matrix(0, 0) {}

    constexpr static Scalar INF = std::numeric_limits<Scalar>::max();
    constexpr static Idx UNDEF = -Idx(1);

    murty::MurtyWorkers<Scalar, Idx> MW;
    std::vector<mtt::MixtureToGaussian<Scalar, SDIM>> mtgs;
    murty::SparseMatrix<Scalar, Idx> smat;
    std::vector<std::vector<Idx>> row_subs;
    mtt::Gater<Scalar, MDIM> gater;

    std::vector<Idx> meas_poiss;
    std::vector<Scalar> log_miss;
    std::vector<uint8_t> meas_mbm;
    std::vector<Idx> meas_list;
    
    std::vector<Scalar> poiss_cost;
    std::vector<Scalar> base_costs;
    murty::DenseMatrix<Scalar> matrix;
    std::vector<Idx> mapping;

    std::vector<Idx> global;

    Idx comps = 0, meas = 0;

    void resize(size_t comps, size_t meas) {
        this->comps = static_cast<Idx>(comps);
        this->meas = static_cast<Idx>(meas);
        if (poiss_cost.size() < meas) {
            poiss_cost.resize(meas, 0);
            meas_mbm.resize(meas, false);
        }
        if (log_miss.size() < comps) {
            log_miss.resize(comps);
        }
        matrix.resize(comps, meas);
        if (mapping.size() < (comps + 1) * (meas + 1)) {
            mapping.resize((comps + 1) * (meas + 1));
        }
    }
    void prepare(size_t comps, size_t meas) {
        base_costs.clear();
        meas_poiss.clear();
        meas_list.clear();

        meas_poiss.reserve(meas);
        meas_list.reserve(meas);
        resize(comps, meas);
    
        std::fill_n(mapping.begin(), (comps + 1) * (meas + 1), UNDEF);
        std::fill_n(meas_mbm.begin(), meas, false);
    }

    Idx& map_elem(Idx comp, Idx z_idx) {
        if (comp == UNDEF) comp = comps;
        if (z_idx == UNDEF) z_idx = meas;
        return mapping[comp * (meas + 1) + z_idx];
    }
};

namespace internal {

/**
 * @brief Updates the Poisson component
 * 
 * The new potentially detected Bernoulli components are added to the
 * mbm. Also dummy components are added with existence probability 0
 */
template <typename DerivedA, typename DerivedB, typename DerivedC, typename Scalar, int SDIM, int MDIM>
requires (mtt::can_be_square<DerivedC>)
void update_poisson(
        mtt::GaussianMixture<Scalar, SDIM>& prior_poiss, mtt::MultiBernoulliMixture<Scalar, SDIM, Idx>& prior_mbm,
        const Eigen::MatrixBase<DerivedA>& Z, Scalar PD, Scalar lambda, Scalar thr,
        const Eigen::MatrixBase<DerivedB>& H, const Eigen::MatrixBase<DerivedC>& R, UpdateWorkers<Scalar, SDIM, MDIM>& W) {

    using StateT = Eigen::Matrix<Scalar, SDIM, 1>;
    using MeasT = Eigen::Matrix<Scalar, MDIM, 1>;

    size_t n_meas = Z.cols();
    int sdim = prior_mbm.get_dim();

    // Iniit the mtgs
    if (prior_poiss.size() > 0) {
        if (W.mtgs.size() < n_meas)
            W.mtgs.resize(n_meas, mtt::MixtureToGaussian<Scalar, SDIM>(sdim));
        for (size_t i = 0; i < n_meas; i++)
            W.mtgs[i].clear();
    }
    
    // Gating with all components. Will be skipped for MBM
    for (size_t i = 0; i < prior_poiss.size(); i++) {
        auto S_LLT = mtt::kf_S_LLT(prior_poiss.cov(i), H, R);
        if (S_LLT.info() != Eigen::Success)
            throw std::runtime_error("LLT failed");
        Eigen::Matrix<Scalar, MDIM, MDIM> L_inv = mtt::internal::get_Linv(S_LLT.matrixL()).eval();
        MeasT eta = mtt::kf_eta(prior_poiss.mu(i), H);

        // Gating over all the measurements
        W.gater.gate_Linv(eta, L_inv, thr);
        if (W.gater.gated_size() == 0) continue;

        // Compute updated components that doesnt depend on z
        auto [P, K] = mtt::kf_cov_K(prior_poiss.cov(i), S_LLT, H);

        // constant used in gaussian pdf
        Scalar logc = mtt::internal::logc_from_L(S_LLT.matrixL());
        Scalar wiPD = prior_poiss.w(i) * PD;

        for (size_t j = 0; j < W.gater.gated_size(); j++) {
            auto z_idx = W.gater.idxs[j];

            // reusing already computed Mahal dist for likelihood
            Scalar lkl = std::exp(logc - Scalar(0.5) * W.gater.distances[j]);
            Scalar nw = wiPD * lkl;

            StateT mean = mtt::kf_mean(prior_poiss.mu(i), Z.col(z_idx), K, eta);
            // Adding gaussian to the moment matcher - at the end, it will result in a single
            // Gaussian for each measurement
            W.mtgs[z_idx].add_gauss(nw, mean, P);
        }
    }

    Scalar log_lambda = (lambda > 0 ? std::log(lambda) : Scalar(-1e3));

    // Create mapping measurement to Bernoulli component
    for (size_t i = 0; i < n_meas; i++) {

        if (prior_poiss.size() == 0) {
            W.poiss_cost[i] = log_lambda;
            continue;
        }

        // labelling is from the paper
        Scalar e = W.mtgs[i]._w;
        Scalar rho = e + lambda;
        Scalar r = (rho < std::numeric_limits<Scalar>::epsilon() ? Scalar(0) : e / rho);
        Scalar log_rho = (rho < std::numeric_limits<Scalar>::epsilon() ? Scalar(-1e3) : std::log(rho));
        // Storing diagonal of the -log[W_nt] matrix
        W.poiss_cost[i] = log_rho;

        if (W.mtgs[i].contains_gauss()) {
            auto gauss = W.mtgs[i].get_gauss();
            W.meas_poiss.push_back(i);
            W.map_elem(W.UNDEF, i) = static_cast<Idx>(prior_mbm.add_comp(r, gauss.mu, gauss.cov, i));
        }
    }
    // Poisson update
    prior_poiss.scale_weight(1 - PD);
}

/**
 * @brief Initialize the cost matrix and updates mbm
 * 
 * In the paper, the cost matrix is designed as -ln[w_match / w_miss], and we
 * solve the association problem. In the PMBM formulation, there is also additional matrix
 * W_nt concatenated to the right and we solve the full assignment problem. Both of these can
 * be transformed into the same problem.
 * 
 * For MBM,
 * since it is an association problem, we can consider the transposed matrix
 * instead of the original one. Also, we final form of the matrix can be rewritten as
 * C = -ln(ri * PD * lkl / (1 - ri + ri * (1 - PD))) - -ln(kappa).
 * 
 * In PMBM,
 * the the cost matrix is constructed as -ln[W_ot, W_nt], where
 * W_ot are old components and W_nt are new potential components and each row (measurement)
 * must be matched to exactly one column. Subtracting a value from the whole row do not change the optimal solution.
 * We can subtract the diagonal element of -ln[W_nt] (other elements are INF) from each row, effectively zeroing W_nt.
 * Now the problem can be transformed into association problem where the cost of not assigning a row is equal to
 * assigning it with new potentially detected component. The final matrix is
 * C = -ln(ri * PD * lkl / (1 - ri + ri * (1 - PD))) - -ln(e(z) + kappa). This problem is also smaller and thus
 * faster to be solved than the original one.
 * 
 * One can see that if we set e(z) to zero (effectively ignoring the Poisson part), the MBM and PMBM
 * cost matrices become equal. We only need to provide the term -ln(e(z) + kappa) for every row
 * 
 * We can also use the transposed matrix instead of the original one to reduce the complexity
 */
template <typename DerivedA, typename DerivedB, typename DerivedC, typename Scalar, int SDIM, int MDIM>
requires (mtt::can_be_square<DerivedC>)
void update_mbm_init_C(
        mtt::MultiBernoulliMixture<Scalar, SDIM, Idx>& prior, size_t orig_comps, const Eigen::MatrixBase<DerivedA>& Z, Scalar PD,
        Scalar thr, const Eigen::MatrixBase<DerivedB>& H, const Eigen::MatrixBase<DerivedC>& R,
        UpdateWorkers<Scalar, SDIM, MDIM>& W) {

    using StateT = Eigen::Matrix<Scalar, SDIM, 1>;
    using MeasT = Eigen::Matrix<Scalar, MDIM, 1>;

    // Initialize all to INF - won't be used in murty
    std::fill_n(W.matrix.data(), W.matrix.rows() * W.matrix.cols(), W.INF);
    Scalar log_PD = std::log(PD);

    for (size_t i = 0; i < orig_comps; i++) {
        auto S_LLT = mtt::kf_S_LLT(prior.locals.cov(i), H, R);
        if (S_LLT.info() != Eigen::Success)
            throw std::runtime_error("LLT failed");
        Eigen::Matrix<Scalar, MDIM, MDIM> L_inv = mtt::internal::get_Linv(S_LLT.matrixL()).eval();
        MeasT eta = mtt::kf_eta(prior.locals.mu(i), H);

        // Making sure that miss_term > 0 for stability
        Scalar ri = std::min(prior.locals.r(i), Scalar(1.0 - 1e-6));
        Scalar miss_term = (1 - ri + ri * (1 - PD));
        Scalar log_miss = std::log(miss_term);
        W.log_miss[i] = log_miss;

        // Update undected component
        W.map_elem(i, W.UNDEF) = i;
        prior.locals.r(i) *= (1 - PD) / miss_term;
        // last measurement association is missdetection
        prior.last_meas[i] = prior.UNDEF;

        // Gating over all the measurements
        W.gater.gate_Linv(eta, L_inv, thr);
        if (W.gater.gated_size() == 0) continue;

        auto [P, K] = mtt::kf_cov_K(prior.locals.cov(i), S_LLT, H);

        // constant used in gaussian pdf
        Scalar logc = mtt::internal::logc_from_L(S_LLT.matrixL());
        Scalar log_ri = std::log(ri);
        Scalar log_riPD = log_ri + log_PD;

        for (size_t j = 0; j < W.gater.gated_size(); j++) {
            auto z_idx = W.gater.idxs[j];
            W.meas_mbm[z_idx] = true;
            Scalar log_lkl = logc - Scalar(0.5) * W.gater.distances[j];
            Scalar log_term = log_riPD + log_lkl;
            // C_ij = -log[detected weight / missed weight] - -log[e(z) + kappa]
            W.matrix(i, z_idx) = -(log_term - log_miss) + W.poiss_cost[z_idx];

            StateT mean = mtt::kf_mean(prior.locals.mu(i), Z.col(z_idx), K, eta);
            W.map_elem(i, z_idx) = prior.add_local(1., mean, P, prior.ids[i], z_idx);
        }
    }
    // NOTE
    // The structure of mbm components is now:
    // - First are the original components updated as missed
    // - Then there are new potentially detected components if PMBM
    // - in the end, the updated original components
}

template <typename Scalar, int SDIM, int MDIM>
auto get_assocs(
        mtt::MultiBernoulliMixture<Scalar, SDIM, Idx>& prior, size_t n_meas, size_t max_hypots,
        bool sparsify, size_t max_per_row, UpdateWorkers<Scalar, SDIM, MDIM>& W) {

    Scalar MAX_VAL = 1e100;
    
    size_t globals = prior.globals.size();
    assert(globals > 0);

    for (size_t i = 0; i < n_meas; i++) {
        if (W.meas_mbm[i]) 
            W.meas_list.push_back(i);
    }

    // Calculating base costs for murty - negative log weight of each global hypothesis
    W.base_costs.clear();
    W.base_costs.reserve(prior.globals.size());
    for (size_t i = 0; i < prior.globals.size(); i++) {
        W.base_costs.push_back(-std::log(prior.globals[i].w));
        // Add also the default term for each g. hypot - all components were missed
        // If component is associted with measurement, its cost in -log[C] is log_w - log_w_miss,
        // so it will cancel this added cost out and result just in log_w for this component
        for (auto& comp : prior.globals[i].idxs) {
            W.base_costs.back() -= W.log_miss[comp];
        }
    }

    // Create row subsets for each global hypothesis
    W.row_subs.clear();
    for (size_t i = 0; i < globals; i++) {
        auto& g = prior.globals[i];
        W.row_subs.emplace_back(std::move(g.idxs));
    }

    // submatrix using only measurement inside some Bernoulli component gate
    std::vector<Idx> tmp(W.matrix.rows());
    std::iota(tmp.begin(), tmp.end(), 0);
    murty::MatrixView<Scalar, Idx> C(W.matrix.data(), W.matrix.cols(), std::move(tmp), W.meas_list);

    if (sparsify) {
        W.smat.fill_from(C, max_per_row, MAX_VAL);
        assert(static_cast<Idx>(W.smat.rows()) == W.comps);
        return murty::solve_subsets<Scalar, Idx, murty::SparseMatrix<Scalar, Idx>>(W.smat, max_hypots, W.MW, W.row_subs, {}, W.base_costs);
    } else {
        return murty::solve_subsets<Scalar, Idx, murty::MatrixView<Scalar, Idx>>(C, max_hypots, W.MW, W.row_subs, {}, W.base_costs);
    }
}

} // namespace internal

/**
 * @brief Performs the PMBM update
 * 
 * @param poiss Poisson component
 * @param mbm Multi-Bernoulli mixture componet
 * @param Z Measurement set as matrix
 * @param PD Probability of detection
 * @param PG Probability of being inside a gate
 * @param lambda clutter intensity
 * @param max_hypots Maximum number of hypothesis
 * @param H Measurement matrix
 * @param R Measurement noise
 * @param W Update workers
 * @param sparsify Whether to sparsify cost matrix
 * @param max_per_row Maximum number of elements per row in the sparsified cost matrix
 */
template <typename DerivedA, typename DerivedB, typename DerivedC, typename Scalar, int SDIM, int MDIM>
requires mtt::can_be_square<DerivedC>
void update(
        mtt::GaussianMixture<Scalar, SDIM>& prior_poiss,  mtt::MultiBernoulliMixture<Scalar, SDIM, Idx>& prior_mbm,
        const Eigen::MatrixBase<DerivedA>& Z, Scalar PD, Scalar PG, Scalar lambda, size_t max_hypots,
        const Eigen::MatrixBase<DerivedB>& H, const Eigen::MatrixBase<DerivedC>& R, UpdateWorkers<Scalar, SDIM, MDIM>& W,
        bool sparsify = true, size_t max_per_row = 25) {

    const auto sdim = prior_mbm.get_dim();
    const auto mdim = Z.rows();
    assert(R.rows() == R.cols() && R.rows() == mdim && "R must have shape (MDIM, MDIM)");
    assert(H.rows() == mdim && H.cols() == sdim && "RH must have shape (MDIM, SDIM)");
    assert(prior_poiss.get_dim() == sdim && "Mixtures dims must match");
    
    assert(prior_mbm.locals.size() == prior_mbm.ids.size());
    assert(prior_mbm.locals.size() == prior_mbm.last_meas.size());

    #ifndef NDEBUG
    for (const auto& g : prior_mbm.globals) {
        for (auto x : g.idxs) assert(x < static_cast<Idx>(prior_mbm.locals.size()));
    }
    #endif

    size_t orig_comps = prior_mbm.locals.size();
    size_t n_meas = Z.cols();
    
    Scalar thr = mtt::chi2_inv<Scalar>(PG, static_cast<Scalar>(mdim));
    W.prepare(orig_comps, n_meas);
    W.gater.set_measurements(Z);

    internal::update_poisson(prior_poiss, prior_mbm, Z, PD, lambda, thr, H, R, W);
    internal::update_mbm_init_C(prior_mbm, orig_comps, Z, PD, thr, H, R, W);

    // If there were no components, create one global hypothesis - all potential components
    if (orig_comps == 0) {
        prior_mbm.clear_globals();
        std::vector<Idx> tmp;
        if (prior_poiss.size() > 0) {
            for (size_t i = 0; i < n_meas; i++) {
                Idx idx = W.map_elem(W.UNDEF, i);
                if (idx != W.UNDEF)
                    tmp.push_back(idx);
            }
        }
        prior_mbm.add_global(1., std::move(tmp));
        return;
    }
    assert(!prior_mbm.globals.empty());
    
    auto asss = internal::get_assocs(prior_mbm, n_meas, max_hypots, sparsify, max_per_row, W);
    prior_mbm.clear_globals();

    constexpr Idx EMPTY = murty::Assignment<Scalar, Idx>::EMPTY;

    // Create new glboal hypothesis
    for (size_t i = 0; i < asss.size(); i++) {
        auto& ass = asss[i].ass;
        auto cost = asss[i].cost;
        W.global.clear();

        std::fill_n(W.meas_mbm.begin(), n_meas, false);

        for (auto [row, col] : ass) {
            assert(row != EMPTY || col != EMPTY);
            if (row == EMPTY) row = W.UNDEF;
            if (col == EMPTY) {
                col = W.UNDEF;
            } else {
                col = W.meas_list[col];
                W.meas_mbm[col] = true;
            }

            Idx idx = W.map_elem(row, col);
            if (idx != W.UNDEF)
                W.global.push_back(idx);
        }
        for (auto meas : W.meas_poiss) {
            if (W.meas_mbm[meas]) continue; 
            Idx idx = W.map_elem(W.UNDEF, meas);
            if (idx != W.UNDEF)
                W.global.push_back(idx);
        }

        #ifndef NDEBUG
        std::vector<Idx> tmp(W.global), ids, last_meas;
        if (!tmp.empty()) {
            std::sort(tmp.begin(), tmp.end());
            for (size_t j = 1; j < tmp.size(); j++) {
                assert(tmp[j - 1] != tmp[j] && "Global hypothesis contains duplicate components");
                ids.push_back(prior_mbm.ids[tmp[j]]);
                last_meas.push_back(prior_mbm.last_meas[tmp[j]]);
            }
            ids.push_back(prior_mbm.ids[tmp[0]]);
            last_meas.push_back(prior_mbm.last_meas[tmp[0]]);
            std::sort(ids.begin(), ids.end());
            std::sort(last_meas.begin(), last_meas.end());
            for (size_t j = 1; j < ids.size(); j++) {
                assert((ids[j] == W.UNDEF || ids[j - 1] != ids[j]) && "Global hypothesis contains several local hypothesis for one track");
            }
            for (size_t j = 1; j < last_meas.size(); j++) {
                assert((last_meas[j] == W.UNDEF || last_meas[j - 1] != last_meas[j]) && "Global hypothesis contains several local hypothesis associated to the same measurement");
            }
        }
        #endif
        prior_mbm.add_global(-cost, W.global);
    }
    // Weight normalization
    prior_mbm.log_sum_exp_norm_globals();
}

} // namespace mtt::pmbm