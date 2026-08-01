/**
 * @file      bind_utils.hpp
 * @brief     Utilities for C++ to python binding
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
#include <Eigen/Core>
#include <algorithm>
#include <stdexcept>
#include <cstring>
#include <string>
#include <vector>
#include <utility>

namespace py = pybind11;

//------------------------------------------------------------------------------------

template <typename T, int SDIM>
py::class_<T>& add_dim_constructor(py::class_<T>& cls) {
    if constexpr (SDIM == Eigen::Dynamic) {
        cls.def(py::init([](int sdim) {
            if (sdim <= 0) throw std::invalid_argument("Dimension must be > 0");
            return std::make_unique<T>(sdim);
        }), py::arg("dim"));
    } else {
        cls.def(py::init([](int sdim) {
            if (sdim != SDIM) throw std::invalid_argument("Runtime sdim must equal compile time DIM");
            return std::make_unique<T>();
        }), py::arg("dim") = SDIM);
    }
    return cls;
}

//------------------------------------------------------------------------------------

/**
 * @return Returns whether given matrix data are continuous
 */
template <typename Derived>
bool is_eigen_contiguous(const Eigen::MatrixBase<Derived>& mat) {
    if (mat.innerStride() != 1) return false;
    if (mat.cols() == 1 || mat.rows() == 1) return true;
    
    if constexpr (Derived::IsRowMajor)
        return mat.outerStride() == mat.cols();
    else
        return mat.outerStride() == mat.rows();
}

template <typename ArrT>
bool is_numpy_contiguous(const ArrT& arr) {
    return (arr.flags() & py::array::c_style);
}

//------------------------------------------------------------------------------------

/**
 * @brief Fills Eigen matrix from numpy array
 * 
 * Respects the element order, do not respect colMajor/ rowMajor diferences
 * (calling it with rowMajor numpy matrix on colMajor Eigen matrix will result in transpose matrix)
 * 
 * @param mat Matrix to be filled
 * @param arr Source array
 */
template <typename Derived>
void fill_eigen_from_numpy(Eigen::MatrixBase<Derived>& mat, const py::array_t<typename Derived::Scalar>& arr) {

    if (static_cast<py::ssize_t>(mat.size()) != arr.size())
        throw std::runtime_error("Size mismatch: Dst size = " + std::to_string(mat.size()) + ", src size = " + std::to_string(arr.size()));
    auto contig_arr = py::array_t<typename Derived::Scalar, py::array::c_style | py::array::forcecast>(arr);

    if (is_eigen_contiguous(mat)) {
        std::memcpy(mat.derived().data(), contig_arr.data(), sizeof(typename Derived::Scalar) * mat.size());
    } else {
        auto ptr = contig_arr.template unchecked<1>();
        for (Eigen::Index i = 0; i < mat.size(); i++) {
            mat.derived()(i) = ptr(i);
        }
    }
}

//------------------------------------------------------------------------------------

/**
 * @brief Fills numpy array from Eigen matrix
 * 
 * Respects the element order, do not respect colMajor/ rowMajor diferences
 * (calling it with rowMajor numpy matrix on colMajor Eigen matrix will result in transpose matrix)
 * 
 * @param mat source Eigen matrix
 * @param arr Destination numpy array
 */
template <typename Derived>
void fill_numpy_from_eigen(const Eigen::MatrixBase<Derived>& mat, py::array_t<typename Derived::Scalar>& arr) {

    if (static_cast<py::ssize_t>(mat.size()) != arr.size())
        throw std::runtime_error("Size mismatch: Dst size = " + std::to_string(arr.size()) + ", src size = " + std::to_string(mat.size()));
    if (!is_numpy_contiguous(arr))
        throw std::runtime_error("Target NumPy array must be C-contiguous to be overwritten by Eigen.");

    if (is_eigen_contiguous(mat)) {
        std::memcpy(arr.mutable_data(), mat.derived().data(), sizeof(typename Derived::Scalar) * mat.size());
    } else {
        auto flat = arr.reshape({arr.size()});
        auto ptr = flat.template mutable_unchecked<1>();
        for (Eigen::Index i = 0; i < mat.size(); i++) {
            ptr(i) = mat.derived()(i);
        }
    }
}

//------------------------------------------------------------------------------------

/**
 * @brief Create array_t with given shape
 * 
 * @tparam Scalar Value type
 * @param shape shape
 * @param ptr pointer to data
 * @param base Base object
 * @return Created array
 */
template <typename Scalar>
py::array_t<Scalar> create_array(const std::vector<size_t>& shape, Scalar* ptr = nullptr, py::handle base = py::handle()) {
    if (ptr)
        return py::array_t<Scalar>(shape, ptr, base);
    return py::array_t<Scalar>(shape);
}

/**
 * @brief Create array_t with given shape
 * 
 * @tparam Scalar Value type
 * @param shape shape
 * @param strides Strides
 * @param ptr pointer to data
 * @param base Base object
 * @return Created array
 */
template <typename Scalar>
py::array_t<Scalar> create_array(
        const std::vector<size_t>& shape, const std::vector<size_t>& strides, Scalar* ptr = nullptr, py::handle base = py::handle()) {
    if (ptr)
        return py::array_t<Scalar>(shape, strides, ptr, base);
    return py::array_t<Scalar>(shape, strides);
}

//------------------------------------------------------------------------------------

/**
 * @brief Returns shape and strides of provided matrix
 * 
 * The returned shape and strides are calculated such that it matches rowMajor or colMajor matrix
 * into a rowMajor numpy array, treating their binary data representation equal - colMajor to 2D will effectively transpose the matrix
 * 
 * @param mat Eigen matrix
 * @param shape 1D-2D-3D shape override or empty
 * @return (shape, strides)
 */
template <typename Derived>
std::pair<std::vector<size_t>, std::vector<size_t>> get_shape_and_strides(
        const Eigen::MatrixBase<Derived>& mat, std::vector<size_t> shape = {}) {
    using Scalar = typename Derived::Scalar;
    size_t elem_size = sizeof(Scalar);
    
    constexpr bool is_col_major = !Derived::IsRowMajor;
    size_t rows = static_cast<size_t>(mat.rows());
    size_t cols = static_cast<size_t>(mat.cols());
    size_t inner_stride = static_cast<size_t>(mat.innerStride()) * elem_size;
    size_t outer_stride = static_cast<size_t>(mat.outerStride()) * elem_size;

    const size_t major = (is_col_major ? cols : rows);
    const size_t minor = (is_col_major ? rows : cols);

    std::vector<size_t> strides;

    if (shape.size() > 3)
        throw std::runtime_error("3D shape max is allowed");

    if (shape.size() == 1) {
        if (shape[0] != static_cast<size_t>(mat.size()) || !is_eigen_contiguous(mat))
            throw std::runtime_error("View shape (" + std::to_string(shape[0]) + ") does not match matrix size " + std::to_string(mat.size()) + " or Eigen matrix is not continuous");
        strides = {inner_stride};
    } else if (shape.empty() || shape.size() == 2) {
        if (shape.size() == 2 && (shape[0] != major || shape[1] != minor))
            throw std::runtime_error("2D shape must match the Matrix - major axis at shape[0]");
        shape = {major, minor};
        strides = {outer_stride, inner_stride};
    } else {
        size_t inner_size = shape[1] * shape[2];
        size_t total_size = inner_size * shape[0];
        if (total_size != static_cast<size_t>(mat.size())) {
            throw std::runtime_error("View shape (" + std::to_string(shape[0]) + "," + std::to_string(shape[1]) + "," + std::to_string(shape[2]) + ") does not match matrix size " + std::to_string(mat.size()));
        }
        if (inner_size != minor) {
            throw std::runtime_error("Inner shape (" + std::to_string(shape[1]) + "," + std::to_string(shape[2]) + ") does not match matrix minor " + std::to_string(minor));
        }
        strides = {outer_stride, inner_stride * shape[2], inner_stride};
    }
    return {shape, strides};
}

//------------------------------------------------------------------------------------

/**
 * @brief Creates a mutable view over Eigen matrix
 * 
 * @param mat Eigen matrix
 * @param def_shape Default shape
 * @param base Base object
 * @return Created view
 */
template <typename Derived>
py::array_t<typename Derived::Scalar> eigen_to_view(Eigen::MatrixBase<Derived>& mat, std::vector<size_t> def_shape = {}, py::handle base = py::handle()) {
    auto [shape, strides] = get_shape_and_strides(mat, def_shape);
    return py::array_t<typename Derived::Scalar>(shape, strides, mat.derived().data(), base);
}

/**
 * @brief Creates a const view over Eigen matrix
 * 
 * @param mat Eigen matrix
 * @param def_shape Default shape
 * @param base Base object
 * @return Created view
 */
template <typename Derived>
py::array_t<typename Derived::Scalar> eigen_to_view(const Eigen::MatrixBase<Derived>& mat, std::vector<size_t> def_shape = {}, py::handle base = py::handle()) {
    auto [shape, strides] = get_shape_and_strides(mat, def_shape);
    auto* ptr = const_cast<typename Derived::Scalar*>(mat.derived().data());
    auto arr = py::array_t<typename Derived::Scalar>(shape, strides, ptr, base);
    arr.attr("flags").attr("writeable") = false;
    return arr;
}

//------------------------------------------------------------------------------------

/**
 * @brief Creates a map view over numpy array
 * 
 * If numpy array isn't c_style = contiguous, it creates a copy
 * 
 * @tparam Scalar Value type
 * @tparam Rows Compile time rows
 * @tparam Cols compile time cols
 * @param arr numpy array
 * @param rows number of rows
 * @param cols number of cols
 * @return Map view
 */
template <typename Scalar, int Rows = Eigen::Dynamic, int Cols = Eigen::Dynamic>
Eigen::Map<const Eigen::Matrix<Scalar, Rows, Cols>> map_dense(const py::array_t<Scalar>& arr, int rows = Rows, int cols = Cols) {
    if (!is_numpy_contiguous(arr))
        throw std::runtime_error("Array have to be c_style contiguous");
    
    if (arr.size() != rows * cols)
        throw std::runtime_error("Array size mismatch");

    return Eigen::Map<const Eigen::Matrix<Scalar, Rows, Cols>>(static_cast<const Scalar*>(arr.data()), rows, cols);
}