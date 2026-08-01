/**
 * @file      dispatchers.hpp
 * @brief     Dispatcher to catch template parameters
 * @author    @vojtagart
 * @date      17/02/2026
 * @see       https://github.com/Vojtagart/mtt
 * * @par Credits & Acknowledgments
 * - Documentation formatted and refined with the assistance of Google Gemini
 * * @copyright Copyright (c) 2026 @vojtagart. Released under the MIT License.
 */

#pragma once
#include <pybind11/pybind11.h>
#include <Eigen/Core>
#include <stdexcept>

#include "../../include/mtt/core/core.hpp"

namespace py = pybind11;


template <typename Func>
decltype(auto) dispatch_dim(int dim, Func&& f) {
    switch (dim) {
        //case 1: return std::forward<Func>(f).template operator()<1>();
        case 2: return std::forward<Func>(f).template operator()<2>();
        //case 3: return std::forward<Func>(f).template operator()<3>();
        case 4: return std::forward<Func>(f).template operator()<4>();
        //case 6: return std::forward<Func>(f).template operator()<6>();
        default: return std::forward<Func>(f).template operator()<Eigen::Dynamic>();
    }
}

template <typename Func>
decltype(auto) dispatch_sdim(int sdim, Func&& f) {
    switch (sdim) {
        //case 2:  return std::forward<Func>(f).template operator()<2>();
        case 4:  return std::forward<Func>(f).template operator()<4>();
        //case 6:  return std::forward<Func>(f).template operator()<6>();
        default: return std::forward<Func>(f).template operator()<Eigen::Dynamic>();
    }
}

template <typename Func>
decltype(auto) dispatch_mdim(int mdim, Func&& f) {
    switch (mdim) {
        case 2:  return std::forward<Func>(f).template operator()<2>();
        //case 3:  return std::forward<Func>(f).template operator()<3>();
        //case 4:  return std::forward<Func>(f).template operator()<4>();
        default: return std::forward<Func>(f).template operator()<Eigen::Dynamic>();
    }
}

template <typename Func>
decltype(auto) dispatch_dims(int sdim, int mdim, Func&& f) {
    return dispatch_sdim(sdim, [&]<int SDIM>() {
        return dispatch_mdim(mdim, [&]<int MDIM>() {
            return std::forward<Func>(f).template operator()<SDIM, MDIM>();
        });
    });
}

template <typename Scalar, typename Func>
decltype(auto) dispatch_gauss_mixture(py::handle obj, Func&& f) {
    if (py::isinstance<mtt::GaussianMixture<Scalar, 2>>(obj)) return std::forward<Func>(f).template operator()<2>();
    //if (py::isinstance<mtt::GaussianMixture<Scalar, 3>>(obj)) return std::forward<Func>(f).template operator()<3>();
    if (py::isinstance<mtt::GaussianMixture<Scalar, 4>>(obj)) return std::forward<Func>(f).template operator()<4>();
    //if (py::isinstance<mtt::GaussianMixture<Scalar, 6>>(obj)) return std::forward<Func>(f).template operator()<6>();
    if (py::isinstance<mtt::GaussianMixture<Scalar, Eigen::Dynamic>>(obj)) return std::forward<Func>(f).template operator()<Eigen::Dynamic>();
    throw std::invalid_argument("Unknown GaussianMixture type (invalid DIM or not GaussianMixture)");
}

template <typename Scalar, typename Func>
decltype(auto) dispatch_multi_bern(py::handle obj, Func&& f) {
    if (py::isinstance<mtt::MultiBernoulli<Scalar, 2>>(obj)) return std::forward<Func>(f).template operator()<2>();
    //if (py::isinstance<mtt::MultiBernoulli<Scalar, 3>>(obj)) return std::forward<Func>(f).template operator()<3>();
    if (py::isinstance<mtt::MultiBernoulli<Scalar, 4>>(obj)) return std::forward<Func>(f).template operator()<4>();
    //if (py::isinstance<mtt::MultiBernoulli<Scalar, 6>>(obj)) return std::forward<Func>(f).template operator()<6>();
    if (py::isinstance<mtt::MultiBernoulli<Scalar, Eigen::Dynamic>>(obj)) return std::forward<Func>(f).template operator()<Eigen::Dynamic>();
    throw std::invalid_argument("Unknown MultiBernoulli type (invalid DIM or not MultiBernoulli)");
}