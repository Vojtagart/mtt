/**
 * @file      eigen_concepts.hpp
 * @brief     Implements eigen concepts
 * @author    @vojtagart
 * @date      17/02/2026
 * @see       https://github.com/Vojtagart/mtt
 * * @par Credits & Acknowledgments
 * - Documentation formatted and refined with the assistance of Google Gemini
 * * @copyright Copyright (c) 2026 @vojtagart. Released under the MIT License.
 */

#pragma once
#include <type_traits>
#include <Eigen/Core>


namespace mtt {

namespace internal {
    template <typename T>
    constexpr int rows_v = std::remove_cvref_t<T>::RowsAtCompileTime;

    template <typename T>
    constexpr int cols_v = std::remove_cvref_t<T>::ColsAtCompileTime;
} // namespace internal

/**
 * @brief Type triat if T has a static members RowsAtCompileTime and ColsAtCompileTime
 */
template <typename T>
concept has_eigen_dims = requires {
    std::remove_cvref_t<T>::RowsAtCompileTime;
    std::remove_cvref_t<T>::ColsAtCompileTime;
};

/**
 * @brief Type trait if T has static number of rows
 */
template <typename T>
concept is_rows_static = has_eigen_dims<T> && (internal::rows_v<T> != Eigen::Dynamic);
/**
 * @brief Type trait if T has static number of columns
 */
template <typename T>
concept is_cols_static = has_eigen_dims<T> && (internal::cols_v<T> != Eigen::Dynamic);

/**
 * @brief Type trait if T represents a column vector
 */
template <typename T>
concept is_col_vector = has_eigen_dims<T> && (internal::cols_v<T> == 1);

/**
 * @brief Type trait if T represents a row vector
 */
template <typename T>
concept is_row_vector = has_eigen_dims<T> && (internal::rows_v<T> == 1);

/**
 * @brief Type trait if T represents a vector
 */
template <typename T>
concept is_vector = is_col_vector<T> || is_row_vector<T>;

/**
 * @brief Type trait whether T might be square matrix
 */
template <typename T>
concept can_be_square = has_eigen_dims<T> &&
                        (internal::rows_v<T> == Eigen::Dynamic || 
                         internal::cols_v<T> == Eigen::Dynamic ||
                         internal::rows_v<T> == internal::cols_v<T>);

} // namespace mtt