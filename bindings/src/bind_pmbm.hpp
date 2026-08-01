/**
 * @file      bind_pmbm.hpp
 * @brief     Bindings for pmbm module
 * @author    @vojtagart
 * @date      1/03/2026
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
#include "../../include/mtt/pmbm/pmbm_tracker.hpp"
#include "../../include/mtt/pmbm/pmb_tracker.hpp"
#include "../../include/mtt/pmbm/mbm_tracker.hpp"

namespace py = pybind11;


//------------------------------------------------------------------------------------

template <typename Scalar, int SDIM, int MDIM>
void bind_pmbm_tracker(py::module& m, const std::string& name) {
    using T = mtt::PmbmTracker<Scalar, SDIM, MDIM>;
    using GaussMixT = mtt::GaussianMixture<Scalar, SDIM>;
    using ArrT = py::array_t<Scalar, py::array::c_style | py::array::forcecast>;
    using StateCovT = typename T::StateCovT;
    using MeasCovT = typename T::MeasCovT;
    using HT = typename T::HT;

    py::class_<T> cls(m, name.c_str());

    cls
    .def(py::init([](const GaussMixT& birth, const Eigen::Ref<const StateCovT>& F, const Eigen::Ref<const StateCovT>& Q,
                     const Eigen::Ref<const HT>& H, const Eigen::Ref<const MeasCovT>& R, Scalar PD, Scalar PS,
                     Scalar lambda, Scalar PG, Scalar min_hypot_w, Scalar min_bern_r, Scalar min_poiss_w, size_t max_hypots, Scalar conf_thr,
                     bool recyclate, bool merge_comps, Scalar merge_thr, bool sparsify, size_t max_per_row, std::optional<GaussMixT> birth0) {
        const int sdim = birth.get_dim();
        const int mdim = static_cast<int>(H.rows());
        check_model(sdim, mdim, F, Q, H, R);

        mtt::pmbm::Config<Scalar> config{
            .PG=PG, .min_hypot_w=min_hypot_w, .min_bern_r=min_bern_r, .min_poiss_w=min_poiss_w,
            .max_hypots=max_hypots, .conf_thr=conf_thr, .recyclate=recyclate, 
            .merge_comps=merge_comps, .merge_thr=merge_thr, .sparsify=sparsify, .max_per_row=max_per_row
        };

        return std::make_unique<T>(birth, F, Q, H, R, PD, PS, lambda, config, std::move(birth0));

    }), py::arg("birth_mixture"), py::arg("F"), py::arg("Q"), py::arg("H"), py::arg("R"),
        py::arg("PD"), py::arg("PS"), py::arg("clutter_int"), py::arg("PG") = 0.999, py::arg("min_global_hypot_weight") = 1e-5,
        py::arg("min_bernoulli_exist_prob") = 1e-5, py::arg("min_poisson_weight") = 1e-4, py::arg("max_hypothesis") = 200,
        py::arg("conf_thr") = 0.5, py::arg("recyclate") = true, py::arg("merge_comps") = false, py::arg("merge_thr") = 0.0,
        py::arg("sparsify") = true, py::arg("max_per_row") = 1000, py::arg("birth_mixture0") = py::none())

    .def("step", [](T& self, const ArrT& Z_arr) {  
        auto Z = map_meas<Scalar, MDIM, ArrT>(Z_arr);
        self.step(Z);
    }, py::arg("measurements"))

    .def("prediction", &T::prediction)
    .def("update", [](T& self, const ArrT& Z_arr) {  
        auto Z = map_meas<Scalar, MDIM, ArrT>(Z_arr);
        self.update(Z);
    }, py::arg("measurements"))
    .def("prune", &T::prune);

    bind_ct_ids<T, Scalar>(cls);
    
    cls
    .def("reset", &T::reset)
    
    .def_readwrite("birth_mixture", &T::births)
    .def_readwrite("birth_mixture0", &T::births0)
    .def_readwrite("poiss", &T::poiss)
    //.def_readwrite("mbm", &T::mbm)
    .def_readwrite("PD", &T::PD)
    .def_readwrite("PS", &T::PS)
    .def_readwrite("clutter_int", &T::lambda)
    .def_readwrite("t0", &T::t0)
    .def_property("PG",
        [](const T& self) {return self.config.PG;},
        [](T& self, Scalar val) {self.config.PG = val;})
    .def_property("min_global_hypot_weight",
        [](const T& self) {return self.config.min_hypot_w;},
        [](T& self, Scalar val) {self.config.min_hypot_w = val;})
    .def_property("min_bernoulli_exist_prob",
        [](const T& self) {return self.config.min_bern_r;},
        [](T& self, Scalar val) {self.config.min_bern_r = val;})
    .def_property("min_poisson_weight",
        [](const T& self) {return self.config.min_poiss_w;},
        [](T& self, Scalar val) {self.config.min_poiss_w = val;})
    .def_property("max_hypothesis",
        [](const T& self) {return self.config.max_hypots;},
        [](T& self, size_t val) {self.config.max_hypots = val;})
    .def_property("conf_thr",
        [](const T& self) {return self.config.conf_thr;},
        [](T& self, Scalar val) {self.config.conf_thr = val;})
    .def_property("recyclate",
        [](const T& self) {return self.config.recyclate;},
        [](T& self, bool val) {self.config.recyclate = val;})
    .def_property("merge_comps",
        [](const T& self) {return self.config.merge_comps;},
        [](T& self, bool val) {self.config.merge_comps = val;})
    .def_property("merge_thr",
        [](const T& self) {return self.config.merge_thr;},
        [](T& self, Scalar val) {self.config.merge_thr = val;})
    .def_property("sparsify",
        [](const T& self) {return self.config.sparsify;},
        [](T& self, bool val) {self.config.sparsify = val;})
    .def_property("max_per_row",
        [](const T& self) {return self.config.max_per_row;},
        [](T& self, size_t val) {self.config.max_per_row = val;});
}

template <typename Scalar, int SDIM, int MDIM>
void bind_mbm_tracker(py::module& m, const std::string& name) {
    using T = mtt::MbmTracker<Scalar, SDIM, MDIM>;
    using MultiBernT = mtt::MultiBernoulli<Scalar, SDIM>;
    using ArrT = py::array_t<Scalar, py::array::c_style | py::array::forcecast>;
    using StateCovT = typename T::StateCovT;
    using MeasCovT = typename T::MeasCovT;
    using HT = typename T::HT;

    py::class_<T> cls(m, name.c_str());

    cls
    .def(py::init([](const MultiBernT& birth, const Eigen::Ref<const StateCovT>& F, const Eigen::Ref<const StateCovT>& Q,
                     const Eigen::Ref<const HT>& H, const Eigen::Ref<const MeasCovT>& R, Scalar PD, Scalar PS,
                     Scalar lambda, Scalar PG, Scalar min_hypot_w, Scalar min_bern_r, size_t max_hypots, Scalar conf_thr,
                     bool merge_comps, Scalar merge_thr, bool sparsify, size_t max_per_row, std::optional<MultiBernT> birth0) {
        const int sdim = birth.get_dim();
        const int mdim = static_cast<int>(H.rows());
        check_model(sdim, mdim, F, Q, H, R);

        mtt::pmbm::MbmConfig<Scalar> config{
            .PG=PG, .min_hypot_w=min_hypot_w, .min_bern_r=min_bern_r,
            .max_hypots=max_hypots, .conf_thr=conf_thr, .merge_comps=merge_comps, 
            .merge_thr=merge_thr, .sparsify=sparsify, .max_per_row=max_per_row
        };

        return std::make_unique<T>(birth, F, Q, H, R, PD, PS, lambda, config, std::move(birth0));

    }), py::arg("birth_mixture"), py::arg("F"), py::arg("Q"), py::arg("H"), py::arg("R"),
        py::arg("PD"), py::arg("PS"), py::arg("clutter_int"), py::arg("PG") = 0.999, py::arg("min_global_hypot_weight") = 1e-5,
        py::arg("min_bernoulli_exist_prob") = 1e-5, py::arg("max_hypothesis") = 200, py::arg("conf_thr") = 0.5,
        py::arg("merge_comps") = false, py::arg("merge_thr") = 0.0, py::arg("sparsify") = true, py::arg("max_per_row") = 1000,
        py::arg("birth_mixture0") = py::none())

    .def("step", [](T& self, const ArrT& Z_arr) {
        auto Z = map_meas<Scalar, MDIM, ArrT>(Z_arr);
        self.step(Z);
    }, py::arg("measurements"))

    .def("prediction", &T::prediction)
    .def("update", [](T& self, const ArrT& Z_arr) {  
        auto Z = map_meas<Scalar, MDIM, ArrT>(Z_arr);
        self.update(Z);
    }, py::arg("measurements"))
    .def("prune", &T::prune);

    bind_ct_ids<T, Scalar>(cls);

    cls
    .def("reset", &T::reset)
    
    .def_readwrite("birth_mixture", &T::births)
    .def_readwrite("birth_mixture0", &T::births0)
    //.def_readwrite("mbm", &T::mbm)
    .def_readwrite("PD", &T::PD)
    .def_readwrite("PS", &T::PS)
    .def_readwrite("clutter_int", &T::lambda)
    .def_readwrite("t0", &T::t0)
    .def_property("PG",
        [](const T& self) {return self.config.PG;},
        [](T& self, Scalar val) {self.config.PG = val;})
    .def_property("min_global_hypot_weight",
        [](const T& self) {return self.config.min_hypot_w;},
        [](T& self, Scalar val) {self.config.min_hypot_w = val;})
    .def_property("min_bernoulli_exist_prob",
        [](const T& self) {return self.config.min_bern_r;},
        [](T& self, Scalar val) {self.config.min_bern_r = val;})
    .def_property("max_hypothesis",
        [](const T& self) {return self.config.max_hypots;},
        [](T& self, size_t val) {self.config.max_hypots = val;})
    .def_property("conf_thr",
        [](const T& self) {return self.config.conf_thr;},
        [](T& self, Scalar val) {self.config.conf_thr = val;})
    .def_property("merge_comps",
        [](const T& self) {return self.config.merge_comps;},
        [](T& self, bool val) {self.config.merge_comps = val;})
    .def_property("merge_thr",
        [](const T& self) {return self.config.merge_thr;},
        [](T& self, Scalar val) {self.config.merge_thr = val;})
    .def_property("sparsify",
        [](const T& self) {return self.config.sparsify;},
        [](T& self, bool val) {self.config.sparsify = val;})
    .def_property("max_per_row",
        [](const T& self) {return self.config.max_per_row;},
        [](T& self, size_t val) {self.config.max_per_row = val;});
}

template <typename Scalar, int SDIM, int MDIM>
void bind_pmb_tracker(py::module& m, const std::string& name) {
    using T = mtt::PmbTracker<Scalar, SDIM, MDIM>;
    using GaussMixT = mtt::GaussianMixture<Scalar, SDIM>;
    using ArrT = py::array_t<Scalar, py::array::c_style | py::array::forcecast>;
    using StateCovT = typename T::StateCovT;
    using MeasCovT = typename T::MeasCovT;
    using HT = typename T::HT;

    py::class_<T> cls(m, name.c_str());

    cls
    .def(py::init([](const GaussMixT& birth, const Eigen::Ref<const StateCovT>& F, const Eigen::Ref<const StateCovT>& Q,
                     const Eigen::Ref<const HT>& H, const Eigen::Ref<const MeasCovT>& R, Scalar PD, Scalar PS,
                     Scalar lambda, Scalar PG, bool track_oriented, Scalar alpha, Scalar min_hypot_w, Scalar min_bern_r,
                     Scalar min_poiss_w, size_t max_hypots, Scalar conf_thr, bool recyclate, bool sparsify, size_t max_per_row,
                     std::optional<GaussMixT> birth0) {
        const int sdim = birth.get_dim();
        const int mdim = static_cast<int>(H.rows());
        check_model(sdim, mdim, F, Q, H, R);

        mtt::pmbm::PmbConfig<Scalar> config{
            .PG=PG, .min_hypot_w=min_hypot_w, .min_bern_r=min_bern_r, .min_poiss_w=min_poiss_w,
            .max_hypots=max_hypots, .conf_thr=conf_thr, .recyclate=recyclate, 
            .track_oriented=track_oriented, .alpha=alpha, .sparsify=sparsify, .max_per_row=max_per_row
        };

        return std::make_unique<T>(birth, F, Q, H, R, PD, PS, lambda, config, std::move(birth0));

    }), py::arg("birth_mixture"), py::arg("F"), py::arg("Q"), py::arg("H"), py::arg("R"),
        py::arg("PD"), py::arg("PS"), py::arg("clutter_int"), py::arg("PG") = 0.999, py::arg("track_oriented") = true, py::arg("alpha") = 0.0,
        py::arg("min_global_hypot_weight") = 1e-5, py::arg("min_bernoulli_exist_prob") = 1e-5, py::arg("min_poisson_weight") = 1e-4, 
        py::arg("max_hypothesis") = 200, py::arg("conf_thr") = 0.5, py::arg("recyclate") = true, py::arg("sparsify") = true, py::arg("max_per_row") = 1000,
        py::arg("birth_mixture0") = py::none())

    .def("step", [](T& self, const ArrT& Z_arr) {
        auto Z = map_meas<Scalar, MDIM, ArrT>(Z_arr);
        self.step(Z);
    }, py::arg("measurements"))

    .def("prediction", &T::prediction)
    .def("update", [](T& self, const ArrT& Z_arr) {  
        auto Z = map_meas<Scalar, MDIM, ArrT>(Z_arr);
        self.update(Z);
    }, py::arg("measurements"))
    .def("prune", &T::prune);

    bind_ct_ids<T, Scalar>(cls);

    cls
    .def("reset", &T::reset)
    
    .def_readwrite("birth_mixture", &T::births)
    .def_readwrite("birth_mixture0", &T::births0)
    .def_readwrite("poiss", &T::poiss)
    //.def_readwrite("mbm", &T::mbm)
    .def_readwrite("PD", &T::PD)
    .def_readwrite("PS", &T::PS)
    .def_readwrite("clutter_int", &T::lambda)
    .def_readwrite("t0", &T::t0)
    .def_property("PG",
        [](const T& self) {return self.config.PG;},
        [](T& self, Scalar val) {self.config.PG = val;})
    .def_property("min_global_hypot_weight",
        [](const T& self) {return self.config.min_hypot_w;},
        [](T& self, Scalar val) {self.config.min_hypot_w = val;})
    .def_property("min_bernoulli_exist_prob",
        [](const T& self) {return self.config.min_bern_r;},
        [](T& self, Scalar val) {self.config.min_bern_r = val;})
    .def_property("min_poisson_weight",
        [](const T& self) {return self.config.min_poiss_w;},
        [](T& self, Scalar val) {self.config.min_poiss_w = val;})
    .def_property("max_hypothesis",
        [](const T& self) {return self.config.max_hypots;},
        [](T& self, size_t val) {self.config.max_hypots = val;})
    .def_property("conf_thr",
        [](const T& self) {return self.config.conf_thr;},
        [](T& self, Scalar val) {self.config.conf_thr = val;})
    .def_property("recyclate",
        [](const T& self) {return self.config.recyclate;},
        [](T& self, bool val) {self.config.recyclate = val;})
    .def_property("sparsify",
        [](const T& self) {return self.config.sparsify;},
        [](T& self, bool val) {self.config.sparsify = val;})
    .def_property("max_per_row",
        [](const T& self) {return self.config.max_per_row;},
        [](T& self, size_t val) {self.config.max_per_row = val;})
    .def_property("track_oriented",
        [](const T& self) {return self.config.track_oriented;},
        [](T& self, bool val) {self.config.track_oriented = val;})
    .def_property("alpha",
        [](const T& self) {return self.config.alpha;},
        [](T& self, Scalar val) {self.config.alpha = val;});
}