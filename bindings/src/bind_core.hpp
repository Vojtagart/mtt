/**
 * @file      bind_mtt.hpp
 * @brief     Bindings for mtt cpp module
 * @author    @vojtagart
 * @date      17/02/2026
 * @see       https://github.com/Vojtagart/mtt
 * * @par Credits & Acknowledgments
 * - Documentation formatted and refined with the assistance of Google Gemini
 * * @copyright Copyright (c) 2026 @vojtagart. Released under the MIT License.
 */

#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <type_traits>
#include <string>
#include <utility>

#include "bind_utils.hpp"
#include "dispatchers.hpp"
#include "../../include/mtt/core/core.hpp"

namespace py = pybind11;

//------------------------------------------------------------------------------------

template <typename ArrT>
void check_mu_cov(const ArrT& mu_arr, const ArrT& cov_arr, int dim) {
    py::ssize_t pydim = static_cast<py::ssize_t>(dim);
    if (mu_arr.size() != pydim)
        throw std::invalid_argument("Mean dimension mismatch: expected " + std::to_string(dim) + ", got " + std::to_string(mu_arr.size()));
    if (cov_arr.size() != pydim * pydim)
        throw std::invalid_argument("Covariance total size mismatch: expected " + std::to_string(dim * dim) + ", got " + std::to_string(cov_arr.size()));
    auto info = cov_arr.request();
    if (info.ndim == 2 && (info.shape[0] != pydim || info.shape[1] != pydim)) {
        throw std::invalid_argument("Covariance shape mismatch: expected (" + std::to_string(dim) + ", " + std::to_string(dim) + ")");
    }
}

//------------------------------------------------------------------------------------

template <typename Scalar, int DIM>
void bind_gaussian_mixture(py::module& m, const std::string& name) {

    using T = mtt::GaussianMixture<Scalar, DIM>;
    using ArrT = py::array_t<Scalar, py::array::c_style | py::array::forcecast>;

    py::class_<T> cls(m, name.c_str());

    add_dim_constructor<T, DIM>(cls)
    .def("push", [](T& self, Scalar w, const ArrT& mu_arr, const ArrT& cov_arr) {
        if (w < 0)
            throw std::invalid_argument("Weight must be non-negative");
        const int dim = self.get_dim();
        check_mu_cov(mu_arr, cov_arr, dim);
        auto mu = map_dense<Scalar, DIM, 1>(mu_arr, dim, 1);
        auto cov = map_dense<Scalar, DIM, DIM>(cov_arr, dim, dim);
        self.push(w, mu, cov);
    }, py::arg("weight"), py::arg("mean"), py::arg("cov"))

    .def("erase", [](T& self, int idx) {
        if (idx < 0) idx += static_cast<int>(self.size());
        if (idx < 0 || idx >= static_cast<int>(self.size()))
            throw std::runtime_error("Index out of bounds");
        self.erase(idx);
    }, py::arg("idx"))

    .def("filter_out", &T::filter_out, py::arg("min_weight"))
    .def("scale_weight", &T::scale_weight, py::arg("multiplier"))
    .def("reserve", &T::reserve, py::arg("new_cap"))
    .def("clear", &T::clear)
    .def("size", &T::size)
    .def("empty", &T::empty)
    .def("capacity", &T::capacity)
    // W: Shape (N,)
    .def_property("W",
        [](T& self) -> py::array_t<Scalar> {
            auto block = self._W.head(self.size());
            return eigen_to_view(block, {self.size()}, py::cast(self));
        },
        [](T& self, py::array_t<Scalar> arr) {
            auto block = self._W.head(self.size());
            fill_eigen_from_numpy(block, arr);
        }
    )
    // M: Shape (N, DIM)
    .def_property("M", 
        [](T& self) -> py::array_t<Scalar> {
            auto block = self._M.leftCols(self.size());
            return eigen_to_view(block, {self.size(), static_cast<size_t>(self.get_dim())}, py::cast(self));
        },
        [](T& self, py::array_t<Scalar> arr) {
            auto block = self._M.leftCols(self.size());
            fill_eigen_from_numpy(block, arr);
        }
    )
    // C: Shape (N, DIM, DIM)
    .def_property("C", 
        [](T& self) -> py::array_t<Scalar> {
            auto block = self._C.leftCols(self.size());
            return eigen_to_view(block, {self.size(), static_cast<size_t>(self.get_dim()), static_cast<size_t>(self.get_dim())}, py::cast(self));
        },
        [](T& self, py::array_t<Scalar> arr) {
            auto block = self._C.leftCols(self.size());
            fill_eigen_from_numpy(block, arr);
        }
    )
    .def("as_gaussian", [](T& self) {
        auto gauss = self.as_gaussian();
        return std::make_pair(gauss.mu, gauss.cov);
    });
}

//------------------------------------------------------------------------------------

template <typename Scalar, int DIM>
void bind_mixture_to_gaussian(py::module& m, const std::string& name) {

    using T = mtt::MixtureToGaussian<Scalar, DIM>;
    using ArrT = py::array_t<Scalar, py::array::c_style | py::array::forcecast>;

    py::class_<T> cls(m, name.c_str());

    add_dim_constructor<T, DIM>(cls)
    .def("add_gauss", [](T& self, Scalar w, const ArrT& mu_arr, const ArrT& cov_arr) {
        if (w < 0)
            throw std::invalid_argument("Weight must be non-negative");
        const int dim = self.get_dim();
        check_mu_cov(mu_arr, cov_arr, dim);
        auto mu = map_dense<Scalar, DIM, 1>(mu_arr, dim, 1);
        auto cov = map_dense<Scalar, DIM, DIM>(cov_arr, dim, dim);
        self.add_gauss(w, mu, cov);
    }, py::arg("weight"), py::arg("mean"), py::arg("cov"))

    .def("get_gauss", [](T& self, Scalar jitter) {
        auto gauss = self.get_gauss(jitter);
        return std::make_pair(gauss.mu, gauss.cov);
    }, py::arg("jitter") = Scalar(0));
}

//------------------------------------------------------------------------------------

template <typename Scalar, int DIM>
void bind_multi_bernoulli(py::module& m, const std::string& name) {

    using T = mtt::MultiBernoulli<Scalar, DIM>;
    using ArrT = py::array_t<Scalar, py::array::c_style | py::array::forcecast>;

    py::class_<T> cls(m, name.c_str());

    add_dim_constructor<T, DIM>(cls)
    .def("push", [](T& self, Scalar r, const ArrT& mu_arr, const ArrT& cov_arr) {
        if (r < 0 || r > 1)
            throw std::invalid_argument("Existence probability be in [0, 1]");
        const int dim = self.get_dim();
        check_mu_cov(mu_arr, cov_arr, dim);
        auto mu = map_dense<Scalar, DIM, 1>(mu_arr, dim, 1);
        auto cov = map_dense<Scalar, DIM, DIM>(cov_arr, dim, dim);
        self.push(r, mu, cov);
    }, py::arg("exist_prob"), py::arg("mean"), py::arg("cov"))

    .def("erase", [](T& self, int idx) {
        if (idx < 0) idx += static_cast<int>(self.size());
        if (idx < 0 || idx >= static_cast<int>(self.size()))
            throw std::runtime_error("Index out of bounds");
        self.erase(idx);
    }, py::arg("idx"))

    .def("filter_out", &T::filter_out, py::arg("min_exist_prob"))
    .def("scale_exist_prob", &T::scale_exist_prob, py::arg("multiplier"))
    .def("reserve", &T::reserve, py::arg("new_cap"))
    .def("clear", &T::clear)
    .def("size", &T::size)
    .def("empty", &T::empty)
    .def("capacity", &T::capacity)
    // R: Shape (N,)
    .def_property("R",
        [](T& self) -> py::array_t<Scalar> {
            auto block = self._R.head(self.size());
            return eigen_to_view(block, {self.size()}, py::cast(self));
        },
        [](T& self, py::array_t<Scalar> arr) {
            auto block = self._R.head(self.size());
            fill_eigen_from_numpy(block, arr);
        }
    )
    // M: Shape (N, DIM)
    .def_property("M", 
        [](T& self) -> py::array_t<Scalar> {
            auto block = self._M.leftCols(self.size());
            return eigen_to_view(block, {self.size(), static_cast<size_t>(self.get_dim())}, py::cast(self));
        },
        [](T& self, py::array_t<Scalar> arr) {
            auto block = self._M.leftCols(self.size());
            fill_eigen_from_numpy(block, arr);
        }
    )
    // C: Shape (N, DIM, DIM)
    .def_property("C", 
        [](T& self) -> py::array_t<Scalar> {
            auto block = self._C.leftCols(self.size());
            return eigen_to_view(block, {self.size(), static_cast<size_t>(self.get_dim()), static_cast<size_t>(self.get_dim())}, py::cast(self));
        },
        [](T& self, py::array_t<Scalar> arr) {
            auto block = self._C.leftCols(self.size());
            fill_eigen_from_numpy(block, arr);
        }
    );
}

//------------------------------------------------------------------------------------

template <typename Scalar, int DIM>
void bind_mixture_to_bernoulli(py::module& m, const std::string& name) {

    using T = mtt::MixtureToBernoulli<Scalar, DIM>;
    using ArrT = py::array_t<Scalar, py::array::c_style | py::array::forcecast>;

    py::class_<T> cls(m, name.c_str());

    add_dim_constructor<T, DIM>(cls)
    .def("add_bern", [](T& self, Scalar w, Scalar r, const ArrT& mu_arr, const ArrT& cov_arr) {
        if (w < 0)
            throw std::invalid_argument("Weight must be non-negative");
        if (r < 0 || r > 1)
            throw std::invalid_argument("Existence probability must be from [0, 1]");
        const int dim = self.get_dim();
        check_mu_cov(mu_arr, cov_arr, dim);
        auto mu = map_dense<Scalar, DIM, 1>(mu_arr, dim, 1);
        auto cov = map_dense<Scalar, DIM, DIM>(cov_arr, dim, dim);
        self.add_bern(w, r, mu, cov);
    }, py::arg("weight"), py::arg("r"), py::arg("mean"), py::arg("cov"))

    .def("get_bern", [](T& self, bool normalize, Scalar jitter) {
        auto bern = self.get_bern(normalize, jitter);
        return std::make_tuple(bern.r, bern.mu, bern.cov);
    }, py::arg("normalize") = true, py::arg("jitter") = Scalar(0));
}

//------------------------------------------------------------------------------------

template <typename Scalar, int DIM>
void bind_gater(py::module& m, const std::string& name) {

    using T = mtt::Gater<Scalar, DIM>;
    using ArrT = py::array_t<Scalar, py::array::c_style | py::array::forcecast>;

    py::class_<T> cls(m, name.c_str());
    
    add_dim_constructor<T, DIM>(cls)
    .def("set_measurements", [](T& self, const ArrT& Z) {
        auto info = Z.request();
        if (info.ndim != 2)
            throw std::invalid_argument("Measurements must be 2D array (N, DIM)");

        if constexpr (DIM != Eigen::Dynamic) {
            if (info.shape[1] != static_cast<py::ssize_t>(DIM))
                throw std::invalid_argument("Measurements dimension do not match expected dimension");
        }
        auto map = map_dense<Scalar, DIM, Eigen::Dynamic>(Z, info.shape[1], info.shape[0]);
        self.set_measurements(map);
    }, py::arg("measurements"))

    .def("gate", [](T& self, const ArrT& mu_arr, const ArrT& cov_arr, Scalar threshold) {
        using IdxT = typename std::decay_t<decltype(self.idxs)>::value_type;
    
        const int dim = self.get_dim();
        check_mu_cov(mu_arr, cov_arr, dim);
        auto mu = map_dense<Scalar, DIM, 1>(mu_arr, dim, 1);
        auto cov = map_dense<Scalar, DIM, DIM>(cov_arr, dim, dim);

        self.gate(mu, cov, threshold);

        auto ret = py::array_t<IdxT>(self.gated_size());
        if (self.gated_size() > 0)
            std::memcpy(ret.mutable_data(), self.idxs.data(), self.gated_size() * sizeof(IdxT));
        return ret;
    }, py::arg("mean"), py::arg("cov"), py::arg("threshold"));
}

//------------------------------------------------------------------------------------

template <typename Scalar>
void bind_math_utils(py::module& m) {

    using ArrT = py::array_t<Scalar, py::array::c_style | py::array::forcecast>;

    m.def("mvn_logpdf", [](const ArrT& x_arr, const ArrT& mu_arr, const ArrT& cov_arr) {
        const int dim = static_cast<int>(x_arr.size());
        return dispatch_dim(dim, [&]<int DIM>() {
            check_mu_cov(mu_arr, cov_arr, dim);
            auto x = map_dense<Scalar, DIM, 1>(x_arr, dim, 1);
            auto mu = map_dense<Scalar, DIM, 1>(mu_arr, dim, 1);
            auto cov = map_dense<Scalar, DIM, DIM>(cov_arr, dim, dim);
            return mtt::mvn_logpdf(x, mu, cov);
        });
    }, "Computes log PDF of Multivariate Normal", py::arg("x"), py::arg("mean"), py::arg("cov"));

    m.def("mahalanobis_distance", [](const ArrT& x_arr, const ArrT& mu_arr, const ArrT& cov_arr) {
        const int dim = static_cast<int>(x_arr.size());
        return dispatch_dim(dim, [&]<int DIM>() {
            check_mu_cov(mu_arr, cov_arr, dim);
            auto x = map_dense<Scalar, DIM, 1>(x_arr, dim, 1);
            auto mu = map_dense<Scalar, DIM, 1>(mu_arr, dim, 1);
            auto cov = map_dense<Scalar, DIM, DIM>(cov_arr, dim, dim);
            return mtt::mahalanobis_distance(x, mu, cov);
        });
    }, "Computes squared Mahalanobis distance", py::arg("x"), py::arg("mean"), py::arg("cov"));
}