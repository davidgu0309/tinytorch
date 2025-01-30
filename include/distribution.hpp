/**
 * @file distribution.hpp
 * 
 * @brief Distribution functions (do we actually need this file?).
 * 
 * @author David Gu
 * @author Mirco Paul
 * 
 * @date \today
 */
#pragma once

#include <random>

/**
 * @namespace tinytorch
 * 
 * @brief Namespace of the entire framework.
 * 
 */
namespace tinytorch {
    std::random_device rd;
    std::mt19937 gen(rd());
}


// namespace tinytorch {
//     template <typename T>
//     T uniform
// }