/**
 * @file      tracker_utils.hpp
 * @brief     Utilities for C++ to python binding
 * @author    @vojtagart
 * @date      1/05/2026
 * @see       https://github.com/Vojtagart/mtt
 * * @par Credits & Acknowledgments
 * - Documentation formatted and refined with the assistance of Google Gemini
 * * @copyright Copyright (c) 2026 @vojtagart. Released under the MIT License.
 */

#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Core>
#include <algorithm>
#include <stdexcept>
#include <cstring>
#include <string>
#include <vector>
#include <utility>

namespace py = pybind11;


//------------------------------------------------------------------------------------

template <typename Scalar, int MDIM, typename ArrT>
auto map_meas(const ArrT& Z_arr) {
    auto info = Z_arr.request();
    if (info.ndim != 2)
        throw std::invalid_argument("Measurements must be (N, MDIM)");
    if constexpr (MDIM != Eigen::Dynamic) {
        if (info.shape[1] != static_cast<py::ssize_t>(MDIM)) 
            throw std::invalid_argument("Measurements dimension do not match expected MDIM");
    }
    return map_dense<Scalar, MDIM, Eigen::Dynamic>(Z_arr, static_cast<int>(info.shape[1]), static_cast<int>(info.shape[0]));
}

//------------------------------------------------------------------------------------

template <typename FT, typename QT, typename HT, typename RT>
void check_model(int sdim, int mdim, const FT& F, const QT& Q, const HT& H, const RT& R) {
    Eigen::Index esdim = sdim;
    Eigen::Index emdim = mdim;
    if (F.rows() != esdim || F.cols() != esdim)
        throw std::invalid_argument("F must be (SDIM, SDIM)");
    if (Q.rows() != esdim || Q.cols() != esdim)
        throw std::invalid_argument("Q must be (SDIM, SDIM)");
    if (H.rows() != emdim || H.cols() != esdim)
        throw std::invalid_argument("H must be (MDIM, SDIM)");
    if (R.rows() != emdim || R.cols() != emdim)
        throw std::invalid_argument("R must be (MDIM, MDIM)");
}

//------------------------------------------------------------------------------------

template <typename T, typename Scalar>
void bind_ct(py::class_<T>& cls) {

    cls
    .def("confirmed_tracks", [](const T& self, int type) {
        using PairT = decltype(self.confirmed_tracks(type));

        auto ptr = std::make_unique<PairT>(self.confirmed_tracks(type));
        auto& mus = ptr->first;
        auto& covs = ptr->second;
        const size_t n = static_cast<size_t>(mus.cols());
        const size_t sdim = static_cast<size_t>(mus.rows());
        
        py::capsule free_when_done(ptr.release(), [](void* ptr_) {
            delete reinterpret_cast<PairT*>(ptr_);
        });
        auto mus_arr = create_array<Scalar>({n, sdim}, mus.data(), free_when_done);
        auto covs_arr = create_array<Scalar>({n, sdim, sdim}, covs.data(), free_when_done);
        return std::make_pair(mus_arr, covs_arr);
    }, py::arg("type") = 1)
    .def("unconfirmed_tracks", [](const T& self, int type) {
        using PairT = decltype(self.unconfirmed_tracks(type));

        auto ptr = std::make_unique<PairT>(self.unconfirmed_tracks(type));
        auto& mus = ptr->first;
        auto& covs = ptr->second;
        const size_t n = static_cast<size_t>(mus.cols());
        const size_t sdim = static_cast<size_t>(mus.rows());
        
        py::capsule free_when_done(ptr.release(), [](void* ptr_) {
            delete reinterpret_cast<PairT*>(ptr_);
        });
        auto mus_arr = create_array<Scalar>({n, sdim}, mus.data(), free_when_done);
        auto covs_arr = create_array<Scalar>({n, sdim, sdim}, covs.data(), free_when_done);
        return std::make_pair(mus_arr, covs_arr);
    }, py::arg("type") = 1);
}

//------------------------------------------------------------------------------------

template <typename T, typename Scalar>
void bind_ct_ids(py::class_<T>& cls) {

    cls
    .def("confirmed_tracks", [](const T& self, int type) {
        using TupleT = decltype(self.confirmed_tracks(type));

        auto ptr = std::make_unique<TupleT>(self.confirmed_tracks(type));
        auto& mus = std::get<0>(*ptr);
        auto& covs = std::get<1>(*ptr);
        auto& ids = std::get<2>(*ptr);
        const size_t n = static_cast<size_t>(mus.cols());
        const size_t sdim = static_cast<size_t>(mus.rows());
        
        py::capsule free_when_done(ptr.release(), [](void* ptr_) {
            delete reinterpret_cast<TupleT*>(ptr_);
        });
        auto mus_arr = create_array<Scalar>({n, sdim}, mus.data(), free_when_done);
        auto covs_arr = create_array<Scalar>({n, sdim, sdim}, covs.data(), free_when_done);
        auto ids_arr = create_array<int>({n}, ids.data(), free_when_done);
        return std::make_tuple(mus_arr, covs_arr, ids_arr);
    }, py::arg("type") = 1)
    .def("unconfirmed_tracks", [](const T& self, int type) {
        using TupleT = decltype(self.unconfirmed_tracks(type));

        auto ptr = std::make_unique<TupleT>(self.unconfirmed_tracks(type));
        auto& mus = std::get<0>(*ptr);
        auto& covs = std::get<1>(*ptr);
        auto& ids = std::get<2>(*ptr);
        const size_t n = static_cast<size_t>(mus.cols());
        const size_t sdim = static_cast<size_t>(mus.rows());
        
        py::capsule free_when_done(ptr.release(), [](void* ptr_) {
            delete reinterpret_cast<TupleT*>(ptr_);
        });
        auto mus_arr = create_array<Scalar>({n, sdim}, mus.data(), free_when_done);
        auto covs_arr = create_array<Scalar>({n, sdim, sdim}, covs.data(), free_when_done);
        auto ids_arr = create_array<int>({n}, ids.data(), free_when_done);
        return std::make_tuple(mus_arr, covs_arr, ids_arr);
    }, py::arg("type") = 1);
}