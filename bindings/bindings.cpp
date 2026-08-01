/**
 * @file      bindings.cpp
 * @brief     Master bindings, binds concrete dimensions
 * @author    @vojtagart
 * @date      17/02/2026
 * @see       https://github.com/Vojtagart/mtt
 * * @par Credits & Acknowledgments
 * - Documentation formatted and refined with the assistance of Google Gemini
 * * @copyright Copyright (c) 2026 @vojtagart. Released under the MIT License.
 */

#include <pybind11/pybind11.h>
#include <string>

#include "src/bind_core.hpp"
#include "src/bind_phd.hpp"
#include "src/bind_pmbm.hpp"

namespace py = pybind11;


//------------------------------------------------------------------------------------

template <typename Scalar, int SDIM>
void bind_sdims(py::module& m, const std::string& suffix) {
    bind_gaussian_mixture<Scalar, SDIM>(m, "_GaussianMixture" + suffix);
    bind_mixture_to_gaussian<Scalar, SDIM>(m, "_MixtureToGaussian" + suffix);
    bind_mixture_to_bernoulli<Scalar, SDIM>(m, "_MixtureToBernoulli" + suffix);
    bind_multi_bernoulli<Scalar, SDIM>(m, "_MultiBernoulli" + suffix);
}

template <typename Scalar, int MDIM>
void bind_mdims(py::module& m, const std::string& suffix) {
    bind_gater<Scalar, MDIM>(m, "_Gater" + suffix);
}

template <typename Scalar, int SDIM, int MDIM>
void bind_both(py::module& m, const std::string& suffix) {
    bind_phd_tracker<Scalar, SDIM, MDIM>(m, "_PhdTracker" + suffix);

    bind_pmbm_tracker<Scalar, SDIM, MDIM>(m, "_PmbmTracker" + suffix);
    bind_mbm_tracker<Scalar, SDIM, MDIM>(m, "_MbmTracker" + suffix);
    bind_pmb_tracker<Scalar, SDIM, MDIM>(m, "_PmbTracker" + suffix);
}

template <typename T>
void bind_mtt(py::module& m, const std::string& type_str) {

    std::string type_suf = "_" + type_str;

    // MTT
    bind_math_utils<T>(m);

    // DIMS
    //bind_sdims<T, 2>(m, type_suf +  "_2");
    bind_sdims<T, 4>(m, type_suf +  "_4");
    //bind_sdims<T, 6>(m, type_suf +  "_6");
    
    bind_mdims<T, 2>(m, type_suf +  "_2");
    //bind_mdims<T, 3>(m, type_suf +  "_3");
    //bind_mdims<T, 4>(m, type_suf +  "_4");

    bind_both<T, 4, 2>(m, type_suf +  "_4_2");
    //bind_both<T, 6, 2>(m, type_suf +  "_6_2");

    bind_sdims<T, Eigen::Dynamic>(m, type_suf);
    bind_mdims<T, Eigen::Dynamic>(m, type_suf);
    bind_both<T, Eigen::Dynamic, Eigen::Dynamic>(m, type_suf);
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "Bindings for MTT";

    bind_mtt<double>(m, "d");
}