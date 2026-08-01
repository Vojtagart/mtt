/**
 * @file      multi_bernoulli.hpp
 * @brief     Container for Multi Bernoulli
 * @author    @vojtagart
 * @date      17/02/2026
 * @see       https://github.com/Vojtagart/mtt
 * * @par Credits & Acknowledgments
 * - kld_dist inspired by: M. Fontana, Á. F. García-Fernández and S. Maskell, "Data-Driven Clustering and Bernoulli Merging for the Poisson Multi-Bernoulli Mixture Filter," in IEEE Transactions on Aerospace and Electronic Systems, vol. 59, no. 5, pp. 5287-5301, Oct. 2023
 * - Documentation formatted and refined with the assistance of Google Gemini
 * * @copyright Copyright (c) 2026 @vojtagart. Released under the MIT License.
 */

#pragma once
#include <vector>
#include <algorithm>
#include <utility>
#include <cassert>
#include <Eigen/Core>

#include "eigen_concepts.hpp"

namespace mtt {

/**
 * @brief Container representing a Bernoulli
 * 
 * @tparam Scalar Type used for scalars
 * @tparam DIM Dimension of Bernoulli
 */
template <typename Scalar, int DIM=Eigen::Dynamic>
struct Bernoulli {
    using MeanT = Eigen::Matrix<Scalar, DIM, 1>;
    using CovT  = Eigen::Matrix<Scalar, DIM, DIM>;

    Scalar r;
    MeanT mu;
    CovT cov;

    /**
     * @brief Construct a new Bernoulli object
     * 
     * @param dim Dimension of the Bernoulli (required if DIM is Dynamic)
     */
    explicit Bernoulli(int dim = DIM) {
        if constexpr (DIM == Eigen::Dynamic) {
            assert(dim > 0 && dim != Eigen::Dynamic && "Runtime dim must be positive and fixed");
            mu.setZero(dim, 1);
            cov.setIdentity(dim, dim);
        } else {
            assert(dim == DIM && "Compile-time size must equal runtime size");
            (void)dim;
        }
    }

    /**
     * @brief Constructs new Bernoulli from Eigen expressions
     * 
     * @param mu Mean of the Bernoulli
     * @param cov Covariance of the Bernoulli
     */
    template <typename M, typename C>
    requires (is_col_vector<M> && can_be_square<C>)
    Bernoulli(M&& mu, C&& cov) : mu(std::forward<M>(mu)), cov(std::forward<C>(cov)) {
        assert(this->mu.rows() == this->cov.rows() && "Mean/Cov dimension mismatch");
        assert(internal::is_square(this->cov) && "Covariance must be a square");
        if constexpr (DIM != Eigen::Dynamic)
            assert(this->mu.rows() == DIM && "Input dimensions do not match compile-time DIM");
    }

    /**
     * @return Dimension of the Bernoulli
     */
    [[nodiscard]] int get_dim() const noexcept {
        if constexpr (DIM == Eigen::Dynamic)
            return static_cast<int>(mu.rows());
        else
            return DIM;
    }

    /**
     * @brief Swap contents with another Bernoulli
     * 
     * @param other Other Bernoulli
     */
    void swap(Bernoulli& other) noexcept {
        mu.swap(other.mu);
        cov.swap(other.cov);
    }
    /**
     * @brief Exchange the contents of two Bernoullis
     */
    friend void swap(Bernoulli& lhs, Bernoulli& rhs) noexcept {
        lhs.swap(rhs);
    }
};

//------------------------------------------------------------------------------------

/**
 * @brief Container representing a Multi Bernoulli
 * 
 * @tparam Scalar Type used for scalars
 * @tparam DIM Dimension of Bernoullis
 */
template <typename Scalar, int DIM=Eigen::Dynamic>
struct MultiBernoulli {
    using MeanT = Eigen::Matrix<Scalar, DIM, 1>;
    using CovT  = Eigen::Matrix<Scalar, DIM, DIM>;

    static constexpr int COV_ROWS = (DIM == Eigen::Dynamic ? Eigen::Dynamic : DIM * DIM);

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> _R;            ///< Vector of existence probs
    Eigen::Matrix<Scalar, DIM, Eigen::Dynamic> _M;          ///< Matrix of Means
    Eigen::Matrix<Scalar, COV_ROWS, Eigen::Dynamic> _C;     ///< Matrix of covariances

    constexpr static size_t INIT_CAP = 8;

    size_t _size = 0;

    explicit MultiBernoulli(int dim = DIM) {
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
     * @param r existence probability of the component
     * @param mu Mean
     * @param cov Covariance
     */
    template <typename DerivedA, typename DerivedB>
    requires (is_col_vector<DerivedA> && can_be_square<DerivedB>)
    void push(Scalar r, const Eigen::MatrixBase<DerivedA>& mu, const Eigen::MatrixBase<DerivedB>& cov) {
        assert(mu.rows() == get_dim() && "Mean dimension mismatch");
        assert(cov.rows() == get_dim() && cov.cols() == get_dim() && "Covariance dimension mismatch");
        assert(r >= Scalar(0));
        ensure_cap();
        // So that assert do not make problem
        size_t idx = _size;
        _size++;
        this->r(idx) = r;
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
            _R(idx) = _R(last);
            _M.col(idx) = _M.col(last);
            _C.col(idx) = _C.col(last);
        }
        _size--;
    }

    /**
     * @brief Removes components with exists. prob. below min_r
     * @param min_r threshold existence probability
     */
    void filter_out(Scalar min_r) {
        for (size_t i = 0; i < _size;) {
            if (r(i) < min_r)
                erase(i);
            else
                i++;
        }
    }
    /**
     * @brief Multiplies all component existence probabilities by val
     * @param val value to multiply exist. prob. with
     */
    void scale_exist_prob(Scalar val) {
        _R.topRows(_size) *= val;
    }

    /**
     * @brief Reserves capacity for weights and components
     * @param cap capacity to be reserved
     */
    void reserve(size_t cap) {
        if (cap <= capacity()) return;
        _R.conservativeResize(cap);
        _M.conservativeResize(get_dim(), cap);
        _C.conservativeResize(get_dim() * get_dim(), cap);
    }
    /** @brief Remove all components from the mixture, preserves capacity */
    void clear() {
        _size = 0;
    }
    /** @brief Resizes the buffer */
    void resize(size_t nsize) {
        if (nsize > capacity())
            reserve(nsize);
        _size = nsize;
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
        return static_cast<size_t>(_R.rows());
    }

    /** @brief Mutable access to component exist. prob. by index */
    [[nodiscard]] Scalar& r(size_t idx) {
        assert(idx < _size);
        return _R(idx);
    }
    /** @brief Const access to component exist. prob. by index */
    [[nodiscard]] Scalar r(size_t idx) const {
        assert(idx < _size);
        return _R(idx);
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
     * @brief Sets Bernoulli at index idx to given values
     * @param idx Index of the Bernoulli
     * @param r existence probability of the component
     * @param mu Mean
     * @param cov Covariance
     */
    template <typename DerivedA, typename DerivedB>
    requires (is_col_vector<DerivedA> && can_be_square<DerivedB>)
    void set(size_t idx, Scalar r, const Eigen::MatrixBase<DerivedA>& mu, const Eigen::MatrixBase<DerivedB>& cov) {
        assert(idx < _size);
        this->r(idx) = r;
        this->mu(idx) = mu;
        this->cov(idx) = cov;
    }

    /**
     * @brief Exchanges content with other
     * 
     * @param other other MultiBernoulli
     */
     void swap(MultiBernoulli& other) noexcept {
        _R.swap(other._R);
        _M.swap(other._M);
        _C.swap(other._C);
        std::swap(_size, other._size);
    }
    /**
     * @brief Exchanges content of the MultiBernoulli
     */
    friend void swap(MultiBernoulli& lhs, MultiBernoulli& rhs) noexcept {
        lhs.swap(rhs);
    }

    /**
     * @return Dimension of the MultiBernoulli
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

//------------------------------------------------------------------------------------

template <typename Scalar, typename DerivedA, typename DerivedB, typename DerivedC, typename DerivedD>
requires (is_col_vector<DerivedA> && is_col_vector<DerivedC> && can_be_square<DerivedB> && can_be_square<DerivedD>)
Scalar kld_dist(
        Scalar r1, const Eigen::MatrixBase<DerivedA>& mu1, const Eigen::MatrixBase<DerivedB>& cov1, Scalar logdet1,
        Scalar r2, const Eigen::MatrixBase<DerivedC>& mu2, const Eigen::LLT<DerivedD>& llt, Scalar logdet2) {
    constexpr Scalar INF = std::numeric_limits<Scalar>::max();
    constexpr Scalar EPS = std::numeric_limits<Scalar>::epsilon();
    
    assert(mu1.rows() == cov1.rows() && internal::is_square(cov1));
    assert(llt.rows() == mu2.rows());
    assert(mu1.rows() == mu2.rows());
    if (llt.info() != Eigen::Success)
        throw std::runtime_error("Passed failed LLT");
    
    if ((r2 <= EPS || r2 == 1) && std::abs(r1 - r2) > EPS)
        return INF;

    // For improved numerical stability
    r1 = std::min(r1, Scalar(1 - 1e-8));
    r2 = std::min(r2, Scalar(1 - 1e-8));

    if (r1 <= EPS)
        return std::log(1 / (1 - r2));

    Scalar mahal = mahalanobis_distance_L(mu1, mu2, llt.matrixL());
    Scalar tr = llt.solve(cov1).trace();
    Scalar dist = Scalar(0.5) * r1 * (tr - logdet1 + logdet2 - mu1.rows() + mahal);

    if (std::abs(r1 - r2) <= EPS)
        return dist;
    return (1 - r1) * std::log((1 - r1) / (1 - r2)) + r1 * std::log(r1 / r2) + dist;
}


template <typename Scalar, typename DerivedA, typename DerivedB, typename DerivedC, typename DerivedD>
requires (is_col_vector<DerivedA> && is_col_vector<DerivedC> && can_be_square<DerivedB> && can_be_square<DerivedD>)
Scalar kld_dist(
        Scalar r1, const Eigen::MatrixBase<DerivedA>& mu1, const Eigen::MatrixBase<DerivedB>& cov1,
        Scalar r2, const Eigen::MatrixBase<DerivedC>& mu2, const Eigen::MatrixBase<DerivedD>& cov2) {
    constexpr Scalar INF = std::numeric_limits<Scalar>::max();
    constexpr Scalar EPS = std::numeric_limits<Scalar>::epsilon();
    assert(cov1.rows() == cov2.rows() && internal::is_square(cov1) && internal::is_square(cov2));

    auto cov1_LLT = cov1.derived().llt();
    if (cov1_LLT.info() != Eigen::Success)
        throw std::runtime_error("cov1 isnt PD");
    auto L1 = cov1_LLT.matrixL();
    auto cov2_LLT = cov2.derived().llt();
    if (cov2_LLT.info() != Eigen::Success)
        throw std::runtime_error("cov2 isnt PD");
    auto L2 = cov2_LLT.matrixL();
    
    int dim = cov1.rows();
    Scalar logdet1 = 0, logdet2 = 0;
    for (Eigen::Index i = 0; i < dim; i++) {
        logdet1 += std::log(L1.derived()(i, i));
        logdet2 += std::log(L2.derived()(i, i));
    }
    return kld_dist(r1, mu1, cov1, logdet1, r2, mu2, cov2_LLT, logdet2);
}

//------------------------------------------------------------------------------------

/**
 * @brief Helper to aggregate a mixture into a single Gaussian
 * 
 * Faster and easier to maintain if the GaussianMixture isn't constructed
 * implicitely, but less stable than the standard 2-pass
 */
template <typename Scalar, int DIM=Eigen::Dynamic>
struct MixtureToBernoulli {
    using BernT = Bernoulli<Scalar, DIM>;
    using MeanT = typename BernT::MeanT;
    using CovT = typename BernT::CovT;

    Scalar _w;
    Scalar _r;
    MeanT _mu;
    CovT _cov;

    /**
     * @brief Constructs a new MixtureToBernoulli with given dimension
     * 
     * @param dim dimension of the Bernoullis (required if DIM is Dynamic)
     */
    explicit MixtureToBernoulli(int dim = DIM) : _w(0), _r(0) {
        if constexpr (DIM != Eigen::Dynamic)
            assert(dim == DIM && "Runtime dimension must match compile time dimension");
        else
            assert(dim > 0 && dim != Eigen::Dynamic && "Dimension must be positive");
        _mu = MeanT::Zero(dim);
        _cov = CovT::Zero(dim, dim);
    }

    /**
     * @brief Add a weighted Bernoulli to the accumulator
     * @param w Weight of the Bernoulli
     * @param r Existence probability of the Bernoulli
     * @param mu Mean vector of the Bernoulli
     * @param cov Covariance of the Bernoulli
     */
    template <typename DerivedA, typename DerivedB>
    void add_bern(Scalar w, Scalar r, const Eigen::MatrixBase<DerivedA>& mu, const Eigen::MatrixBase<DerivedB>& cov) {
        assert(w >= Scalar(0) && "Weight must be non-negative");
        assert(r >= Scalar(0) && r <= Scalar(1) && "Existence probability must be from [0, 1]");
        assert(mu.rows() == get_dim() && "Mean dimension mismatch");
        assert(cov.rows() == get_dim() && cov.cols() == get_dim() && "Covariance dimension mismatch");
        Scalar wr = w * r;
        _w += w;
        _r += wr;
        _mu.noalias() += wr * mu;
        _cov.noalias() += wr * cov;
        _cov.noalias() += wr * mu * mu.transpose();
    }

    /**
     * @brief Produce a single Bernoulli from accumulated weighted moments
     *
     * @return Bernoulli produced by moment matching from added Bernoullis
     */
    [[nodiscard]] BernT get_bern(bool normalize = true, Scalar jitter = 1e-8) const {
        BernT ret(get_dim());
        if (!contains_bern() || _r < std::numeric_limits<Scalar>::epsilon()) {
            ret.r = 0;
            ret.mu.setZero();
            ret.cov.setIdentity();
            return ret;
        }
        ret.r = (normalize ? _r / _w : _r);
        ret.mu = _mu / _r;
        ret.cov.noalias() = _cov / _r - ret.mu * ret.mu.transpose();
        ret.cov = Scalar(0.5) * (ret.cov + ret.cov.transpose());
        ret.cov.diagonal().array() += Scalar(jitter);
        return ret;
    }

    /**
     * @brief Returns true if accumulator contains some weight
    */
    [[nodiscard]] bool contains_bern() const noexcept {
        return _w > Scalar(0);
    }

    /**
     * @return Dimension of the MixtureToBernoulli
     */
    [[nodiscard]] int get_dim() const noexcept {
        if constexpr (DIM == Eigen::Dynamic)
            return _mu.rows();
        else
            return DIM;
    }

    void clear() {
        _w = 0;
        _r = 0;
        _mu.setZero();
        _cov.setZero();
    }
};

} // namespace mtt
