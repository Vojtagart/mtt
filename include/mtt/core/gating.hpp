/**
 * @file      gating.hpp
 * @brief     Gating implementation
 * @author    @vojtagart
 * @date      17/02/2026
 * @see       https://github.com/Vojtagart/mtt
 * * @par Credits & Acknowledgments
 * - Documentation formatted and refined with the assistance of Google Gemini
 * * @copyright Copyright (c) 2026 @vojtagart. Released under the MIT License.
 */

#pragma once
#include <vector>
#include <utility>
#include <type_traits>
#include <cassert>
#include <algorithm>
#include <Eigen/Core>
#include <Eigen/Cholesky>

#include <Eigen/Eigenvalues>
#include <nanoflann.hpp>

#include "gaussian.hpp"
#include "eigen_concepts.hpp"


namespace mtt {

/**
 * @brief Adaptor to allow nanoflann read from Eigen::Matrix
 */
template <typename Scalar, int MDIM>
struct EigenMatrixAdaptor {
    const Eigen::Matrix<Scalar, MDIM, Eigen::Dynamic>& mat;
    EigenMatrixAdaptor(const Eigen::Matrix<Scalar, MDIM, Eigen::Dynamic>& m) : mat(m) {}
    inline size_t kdtree_get_point_count() const {return static_cast<size_t>(mat.cols());}
    inline Scalar kdtree_get_pt(const size_t idx, const size_t dim) const {return mat(dim, idx);}
    template <class BBOX> bool kdtree_get_bbox(BBOX&) const {return false;}
};

/**
 * @brief Structure used to perform gating over a fixed set of measurements
 * 
 * @tparam Scalar Type used for values
 * @tparam MDIM Dimension of the measurement
 */
template <typename Scalar, int MDIM=Eigen::Dynamic>
struct Gater {

    Eigen::Matrix<Scalar, MDIM, Eigen::Dynamic> measurements;      ///< Stored measurements
    std::vector<int> idxs;                                         ///< Indexes of measurement within the gate (from the last gate call)
    std::vector<Scalar> distances;                                 ///< Squared Mahal. distances to measurement within the gate (from the last gate call)
    bool is_tree = false;
    static constexpr int TREE_THR = 150;
    static constexpr size_t LEAF_SIZE = 16;

    using KDTreeAdaptor = EigenMatrixAdaptor<Scalar, MDIM>;
    using KDTree = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<Scalar, KDTreeAdaptor>, KDTreeAdaptor, MDIM, int>;

    std::unique_ptr<KDTreeAdaptor> tree_adaptor;
    std::unique_ptr<KDTree> kdtree;
    std::vector<nanoflann::ResultItem<int, Scalar>, std::allocator<nanoflann::ResultItem<int, Scalar>>> kdtree_ret;

    Gater(int dim = MDIM) {
        if constexpr (MDIM != Eigen::Dynamic) {
            assert(dim == MDIM && "Runtime dimension must match compile time dimension");
            (void)dim;
        } else {
            assert(dim != Eigen::Dynamic && dim > 0 && "Dimension must be positive");
            measurements.resize(dim, 0);
        }
    }

    /**
     * @brief Sets a new set of measurements
     * 
     * @param Z Matrix of measurements with measurement i in i'th column
     */
    template <typename Derived>
    void set_measurements(const Eigen::MatrixBase<Derived>& Z) {
        assert(Z.rows() == get_dim() && "Dimension do not match");
        measurements = Z.derived();

        if (Z.cols() > TREE_THR) {
            is_tree = true;
            tree_adaptor = std::make_unique<KDTreeAdaptor>(measurements);
            kdtree = std::make_unique<KDTree>(get_dim(), *tree_adaptor, nanoflann::KDTreeSingleIndexAdaptorParams(LEAF_SIZE));
            kdtree->buildIndex();
        } else {
            is_tree = false;
        }
    }

    /**
     * @brief Performs gating with respect to given Gaussian parameters
     * 
     * The measurements inside the gate are stored in idxs and their mahalanobis
     * distance is stored in distances so it can be reused
     * 
     * @param mu Mean of the Gaussian
     * @param L_inv Inverse of the Lower Cholesky factor of covariance of the Gaussian
     * @param threshold Maximum squared mahalanobis distance for measurement to be considered
     * @return Count of measurements inside the gate
     */
    template <typename DerivedA, typename DerivedB>
    requires (is_col_vector<DerivedA> && can_be_square<DerivedB>)
    size_t gate_Linv(const Eigen::MatrixBase<DerivedA>& mu, const Eigen::MatrixBase<DerivedB>& L_inv, Scalar threshold) {
        assert(measurements.rows() == mu.rows() && "Gaussian doesnt match the dimension of measurements");
        assert(internal::is_square(L_inv) && mu.rows() == L_inv.rows() && "Covariance must be square with the same dim as mean");
        if (!is_tree) {
            return _gate(threshold, [&](const auto& z) {
                return mahalanobis_distance_Linv(z, mu, L_inv);
            });
        } else {
            using MatrixType = Eigen::Matrix<Scalar, DerivedB::RowsAtCompileTime, DerivedB::ColsAtCompileTime>;
            MatrixType cov_inv = L_inv.transpose() * L_inv;
            Eigen::SelfAdjointEigenSolver<MatrixType>es(cov_inv, Eigen::EigenvaluesOnly);
            Scalar max_dist = (Scalar{1} / es.eigenvalues().minCoeff()) * threshold;
            return _gate(mu, max_dist, threshold, [&](const auto& z) {
                return mahalanobis_distance_Linv(z, mu, L_inv);
            });
        }
    }
    /**
     * @brief Performs gating with respect to given Gaussian parameters
     * 
     * The measurements inside the gate are stored in idxs and their mahalanobis
     * distance is stored in distances so it can be reused
     * 
     * @param mu Mean of the Gaussian
     * @param cov Covariance of the Gaussian
     * @param threshold Maximum squared mahalanobis distance for measurement to be considered
     * @return Count of measurements inside the gate
     */
    template <typename DerivedA, typename DerivedB>
    requires (is_col_vector<DerivedA> && can_be_square<DerivedB>)
    size_t gate(const Eigen::MatrixBase<DerivedA>& mu, const Eigen::MatrixBase<DerivedB>& cov, Scalar threshold) {
        assert(measurements.rows() == mu.rows() && "Gaussian doesnt match the dimension of measurements");
        assert(internal::is_square(cov) && mu.rows() == cov.rows() && "Covariance must be square with the same dim as mean");
        auto llt = cov.derived().llt();
        if (llt.info() != Eigen::Success)
            throw std::runtime_error("cov isn't PD, try adding jitter");
        Eigen::Matrix<Scalar, MDIM, MDIM> L_inv = internal::get_Linv(llt.matrixL()).eval();
        if (!is_tree) {
            return gate_Linv(mu, L_inv, threshold);
        } else {
            using MatrixType = Eigen::Matrix<Scalar, DerivedB::RowsAtCompileTime, DerivedB::ColsAtCompileTime>;
            Eigen::SelfAdjointEigenSolver<MatrixType>es(cov, Eigen::EigenvaluesOnly);
            Scalar max_dist = es.eigenvalues().maxCoeff() * threshold;
            return _gate(mu, max_dist, threshold, [&](const auto& z) {
                return mahalanobis_distance_Linv(z, mu, L_inv);
            });
        }
    }

    /**
     * @return Number of stored measurements
     */
    [[nodiscard]] size_t size() const noexcept {
        return measurements.cols();
    }
    /**
     * @return Number measurements from the last gate()
     */
    [[nodiscard]] size_t gated_size() const noexcept {
        return idxs.size();
    }

    [[nodiscard]] int get_dim() const noexcept {
        if constexpr (MDIM != Eigen::Dynamic)
            return MDIM;
        else
            return measurements.rows();
    }
private:
    
    template <typename Fn>
    size_t _gate(Scalar threshold, Fn mahal_dist) {
        idxs.clear();
        distances.clear();
        for (size_t i = 0; i < size(); i++) {
            Scalar dist = mahal_dist(measurements.col(i));
            if (dist <= threshold) {
                idxs.push_back(static_cast<int>(i));
                distances.push_back(dist);
            }
        }
        return idxs.size();
    }
    template <typename DerivedA, typename Fn>
    size_t _gate(const Eigen::MatrixBase<DerivedA>& mu, Scalar max_dist, Scalar threshold, Fn mahal_dist) {
        idxs.clear();
        distances.clear();
        if (!kdtree || size() == 0)
            return 0;

        nanoflann::SearchParameters params;
        kdtree_ret.clear();
        kdtree->radiusSearch(mu.derived().data(), max_dist, kdtree_ret, params);

        for (const auto& x : kdtree_ret) {
            int idx = x.first;
            Scalar dist = mahal_dist(measurements.col(idx));
            if (dist <= threshold) {
                idxs.push_back(idx);
                distances.push_back(dist);
            }
        }
        kdtree_ret.clear();
        return idxs.size();
    }
};

} // namespace mtt