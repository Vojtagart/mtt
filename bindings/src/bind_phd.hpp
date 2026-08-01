/**
 * @file      bind_phd.hpp
 * @brief     Bindings for phd module
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
#include <string>
#include <utility>
#include <optional>

#include "bind_utils.hpp"
#include "dispatchers.hpp"
#include "../../include/mtt/phd/phd_tracker.hpp"
#include "tracker_utils.hpp"

namespace py = pybind11;


//------------------------------------------------------------------------------------

template <typename Scalar, int SDIM, int MDIM>
void bind_phd_tracker(py::module& m, const std::string& name) {
    using T = mtt::PhdTracker<Scalar, SDIM, MDIM>;
    using GaussMixT = mtt::GaussianMixture<Scalar, SDIM>;
    using ArrT = py::array_t<Scalar, py::array::c_style | py::array::forcecast>;
    using StateCovT = typename T::StateCovT;
    using MeasCovT = typename T::MeasCovT;
    using HT = typename T::HT;

    py::class_<T> cls(m, name.c_str());

    cls
    .def(py::init([](const GaussMixT& birth, const Eigen::Ref<const StateCovT>& F, const Eigen::Ref<const StateCovT>& Q,
                     const Eigen::Ref<const HT>& H, const Eigen::Ref<const MeasCovT>& R,
                     Scalar PD, Scalar PS, Scalar lambda, Scalar PG, Scalar trunc_thr, Scalar merge_thr, size_t max_comps, Scalar conf_thr,
                     std::optional<GaussMixT> birth0) {
        const int sdim = birth.get_dim();
        const int mdim = static_cast<int>(H.rows());
        check_model(sdim, mdim, F, Q, H, R);

        return std::make_unique<T>(birth, F, Q, H, R, PD, PS, lambda, PG, trunc_thr, merge_thr, max_comps, conf_thr, std::move(birth0));

    }), py::arg("birth_mixture"), py::arg("F"), py::arg("Q"), py::arg("H"), py::arg("R"),
        py::arg("PD"), py::arg("PS"), py::arg("clutter_int"), py::arg("PG") = 0.999, 
        py::arg("trunc_thr") = 1e-5, py::arg("merge_thr") = 4.0, 
        py::arg("max_components") = 250, py::arg("conf_thr") = 0.5, py::arg("birth_mixture0") = py::none())

    .def("step", [](T& self, const ArrT& Z_arr) {
        auto Z = map_meas<Scalar, MDIM, ArrT>(Z_arr);
        self.step(Z);
    }, py::arg("measurements"))

    .def("prediction", &T::prediction)
    .def("update", [](T& self, const ArrT& Z_arr) {  
        auto Z = map_meas<Scalar, MDIM, ArrT>(Z_arr);
        self.update(Z);
    }, py::arg("measurements"));

    bind_ct<T, Scalar>(cls);

    cls
    .def("reset", &T::reset)
    
    .def_readwrite("mixture", &T::mix)
    .def_readwrite("birth_mixture", &T::births)
    .def_readwrite("birth_mixture0", &T::births0)
    .def_readwrite("PD", &T::PD)
    .def_readwrite("PS", &T::PS)
    .def_readwrite("clutter_int", &T::lambda)
    .def_readwrite("PG", &T::PG)
    .def_readwrite("trunc_thr", &T::trunc_thr)
    .def_readwrite("merge_thr", &T::merge_thr)
    .def_readwrite("max_components", &T::max_comps)
    .def_readwrite("conf_thr", &T::conf_thr)
    .def_readwrite("t0", &T::t0);
}