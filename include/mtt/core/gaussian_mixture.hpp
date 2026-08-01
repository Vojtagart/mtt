/**
 * @file      gaussian_mixture.hpp
 * @brief     Gaussian mixture implementation
 * @author    @vojtagart
 * @date      17/02/2026
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

#include "gaussian.hpp"


namespace mtt {

/**
 * @brief Container representing a Gaussian mixture
 * 
 * @tparam Scalar Type used for scalars
 * @tparam DIM Dimension of Gaussians
 */
template <typename Scalar, int DIM=Eigen::Dynamic>
struct GaussianMixture {
    using MeanT = Eigen::Matrix<Scalar, DIM, 1>;
    using CovT  = Eigen::Matrix<Scalar, DIM, DIM>;
    using GaussT = mtt::Gaussian<Scalar, DIM>;

    static constexpr int COV_ROWS = (DIM == Eigen::Dynamic ? Eigen::Dynamic : DIM * DIM);

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> _W;            ///< Vector of weights
    Eigen::Matrix<Scalar, DIM, Eigen::Dynamic> _M;          ///< Matrix of Means
    Eigen::Matrix<Scalar, COV_ROWS, Eigen::Dynamic> _C;     ///< Matrix of covariances

    constexpr static size_t INIT_CAP = 8;

    size_t _size = 0;

    explicit GaussianMixture(int dim = DIM) {
        if constexpr (DIM != Eigen::Dynamic) {
            assert(dim == DIM && "Runtime dimension must match compile time dimension");
            (void)dim;
        } else {
            assert(dim != Eigen::Dynamic && dim > 0 && "Dimension must be positive");
            _M.resize(dim, 0);
            _C.resize(dim * dim, 0);
        }
    }

    /**
     * @brief Constructs a component in place
     * @param w weight of the component
     * @param mu Mean of the Gaussian
     * @param cov Covariance of the Gaussian
     */
    template <typename DerivedA, typename DerivedB>
    requires is_col_vector<DerivedA> && can_be_square<DerivedB>
    void push(Scalar w, const Eigen::MatrixBase<DerivedA>& mu, const Eigen::MatrixBase<DerivedB>& cov) {
        assert(mu.rows() == get_dim() && "Mean dimension mismatch");
        assert(cov.rows() == get_dim() && cov.cols() == get_dim() && "Covariance dimension mismatch");
        assert(w >= Scalar(0));
        ensure_cap();
        // So that assert do not make problem
        size_t idx = _size;
        _size++;
        this->w(idx) = w;
        this->mu(idx) = mu;
        this->cov(idx) = cov;
    }
    /**
     * @brief Erases the component at index idx
     * @param idx index of the erased component
     */
    void erase(size_t idx) {
        assert(idx < _size);
        size_t last = _size - 1;
        if (last != idx) {
            _W(idx) = _W(last);
            _M.col(idx) = _M.col(last);
            _C.col(idx) = _C.col(last);
        }
        _size--;
    }

    /**
     * @brief Removes components with weight below min_w
     * @param min_w threshold weight
     */
    void filter_out(Scalar min_w) {
        for (size_t i = 0; i < _size;) {
            if (w(i) < min_w)
                erase(i);
            else
                i++;
        }
    }
    /**
     * @brief Multiplies all component weights by val
     * @param val value to multiply weight with
     */
    void scale_weight(Scalar val) {
        _W.topRows(_size) *= val;
    }

    /**
     * @brief Reserves capacity for weights and components
     * @param cap capacity to be reserved
     */
    void reserve(size_t cap) {
        if (cap <= capacity()) return;
        _W.conservativeResize(cap);
        _M.conservativeResize(get_dim(), cap);
        _C.conservativeResize(get_dim() * get_dim(), cap);
    }
    /** @brief Remove all components from the mixture, preserves capacity */
     void clear() {
        _size = 0;
    }

    /** @brief Number of mixture components */
    [[nodiscard]] size_t size() const noexcept {
        return _size;
    }
    [[nodiscard]] bool empty() const noexcept {
        return _size == 0;
    }
    /** @brief Number of mixture components */
    [[nodiscard]] size_t capacity() const noexcept {
        return static_cast<size_t>(_W.rows());
    }

    /** @brief Mutable access to component weight by index */
    [[nodiscard]] Scalar& w(size_t idx) {
        assert(idx < _size);
        return _W(idx);
    }
    /** @brief Const access to component weight by index */
    [[nodiscard]] Scalar w(size_t idx) const {
        assert(idx < _size);
        return _W(idx);
    }

    /** @brief Mutable access to component mean by index */
    [[nodiscard]] Eigen::Map<MeanT> mu(size_t idx) {
        assert(idx < _size);
        return Eigen::Map<MeanT>(_M.col(idx).data(), get_dim());
    }
    /** @brief Const access to component mean by index */
    [[nodiscard]] Eigen::Map<const MeanT> mu(size_t idx) const {
        assert(idx < _size);
        return Eigen::Map<const MeanT>(_M.col(idx).data(), get_dim());
    }

    /** @brief Mutable access to component covariance by index */
    [[nodiscard]] Eigen::Map<CovT> cov(size_t idx) {
        assert(idx < _size);
        return Eigen::Map<CovT>(_C.col(idx).data(), get_dim(), get_dim());
    }
    /** @brief Const access to component covariance by index */
    [[nodiscard]] Eigen::Map<const CovT> cov(size_t idx) const {
        assert(idx < _size);
        return Eigen::Map<const CovT>(_C.col(idx).data(), get_dim(), get_dim());
    }

    /**
     * @brief Exchanges content with other
     * 
     * @param other other GaussianMixture
     */
     void swap(GaussianMixture& other) noexcept {
        _W.swap(other._W);
        _M.swap(other._M);
        _C.swap(other._C);
        std::swap(_size, other._size);
    }
    /**
     * @brief Exchanges content of the GaussianMixtures
     */
    friend void swap(GaussianMixture& lhs, GaussianMixture& rhs) noexcept {
        lhs.swap(rhs);
    }

    /**
     * @brief Aproximate Gaussian mixture as a single Gaussian
     * 
     * performs moment matching to find Gaussians meand and covariance
     * 
     * @return GaussT produced Gaussian
     */
    [[nodiscard]] GaussT as_gaussian(Scalar jitter = 1e-8) const {
        GaussT ret(get_dim());

        auto weights = _W.topRows(_size);
        Scalar total_w = weights.sum();
        if (total_w == 0) {
            ret.mu.setZero();
            ret.cov.setIdentity();
            return ret;
        }
        auto means = _M.leftCols(_size);
        ret.mu.noalias() = means * weights;
        ret.mu /= total_w;

        ret.cov.setZero();
        for (size_t i = 0; i < _size; i++) {
            auto dif = (ret.mu - mu(i)).eval();
            ret.cov.noalias() += w(i) * (cov(i) + dif * dif.transpose());
        }
        ret.cov /= total_w;
        ret.cov = Scalar{0.5} * (ret.cov + ret.cov.transpose());
        ret.cov.diagonal().array() += Scalar(jitter);
        
        return ret;
    }

    /**
     * @return Dimension of the GaussianMixture
     */
    [[nodiscard]] int get_dim() const noexcept {
        if constexpr (DIM == Eigen::Dynamic)
            return static_cast<int>(_M.rows());
        else
            return DIM;
    }
private:
    void ensure_cap() {
        if (_size >= capacity()) {
            size_t cap = (capacity() == 0 ? INIT_CAP : capacity() * 2);
            reserve(cap);
        }
    }
};

/**
 * @brief Helper to aggregate a mixture into a single Gaussian
 * 
 * Faster and easier to maintain if the GaussianMixture isn't constructed
 * implicitely, but less stable than the standard 2-pass
 */
template <typename Scalar, int DIM=Eigen::Dynamic>
struct MixtureToGaussian {
    using GaussT = Gaussian<Scalar, DIM>;
    using MeanT = typename GaussT::MeanT;
    using CovT = typename GaussT::CovT;

    Scalar _w;
    MeanT _mu;
    CovT _cov;

    /**
     * @brief Constructs a new MixtureToGaussian with given dimension
     * 
     * @param dim dimension of the Gaussians (required if DIM is Dynamic)
     */
    explicit MixtureToGaussian(int dim = DIM) : _w(0) {
        if constexpr (DIM != Eigen::Dynamic)
            assert(dim == DIM && "Runtime dimension must match compile time dimension");
        else
            assert(dim > 0 && dim != Eigen::Dynamic && "Dimension must be positive");
        _mu = MeanT::Zero(dim);
        _cov = CovT::Zero(dim, dim);
    }

    /**
     * @brief Add a weighted Gaussian to the accumulator
     * @param w Weight of the Gaussian
     * @param mu Mean vector of the Gaussian
     * @param cov Covariance of the Gaussian
     */
    template <typename DerivedA, typename DerivedB>
    void add_gauss(Scalar w, const Eigen::MatrixBase<DerivedA>& mu, const Eigen::MatrixBase<DerivedB>& cov) {
        assert(w >= Scalar(0) && "Weight must be non-negative");
        assert(mu.rows() == get_dim() && "Mean dimension mismatch");
        assert(cov.rows() == get_dim() && cov.cols() == get_dim() && "Covariance dimension mismatch");
        _w += w;
        _mu.noalias() += w * mu;
        _cov.noalias() += w * cov;
        _cov.noalias() += w * mu * mu.transpose();
    }

    /**
     * @brief Produce a single Gaussian from accumulated weighted moments
     *
     * @return Gaussian produced by moment matching from added Gaussians
     */
    [[nodiscard]] GaussT get_gauss(Scalar jitter = 1e-8) const {
        GaussT ret(get_dim());
        if (!contains_gauss()) {
            ret.mu.setZero();
            ret.cov.setIdentity();
            return ret;
        }
        ret.mu = _mu / _w;
        ret.cov.noalias() = _cov / _w - ret.mu * ret.mu.transpose();
        ret.cov = Scalar(0.5) * (ret.cov + ret.cov.transpose());
        ret.cov.diagonal().array() += Scalar(jitter);
        return ret;
    }

    /**
     * @brief Returns true if accumulator contains some weight
    */
    [[nodiscard]] bool contains_gauss() const noexcept {
        return _w > Scalar(0);
    }

    /**
     * @return Dimension of the MixtureToGaussian
     */
    [[nodiscard]] int get_dim() const noexcept {
        if constexpr (DIM == Eigen::Dynamic)
            return _mu.rows();
        else
            return DIM;
    }

    void clear() {
        _w = 0;
        int dim = get_dim();
        _mu = MeanT::Zero(dim);
        _cov = CovT::Zero(dim, dim);
    }
};

} // namespace mtt