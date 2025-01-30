/**
 * @file function_traits.hpp
 * 
 * @brief Function traits.
 * 
 * @author David Gu
 * @author Mirco Paul
 * 
 * @date \today
 */

#pragma once

#include <functional>
#include <stdlib.h>

/**
 * @struct FunctionTraits
 * 
 * Helper template to extract argument types of a function.
 * 
 */
template <typename Func>
struct FunctionTraits;

/**
 * @struct FunctionTraits
 * 
 * Specialization for function pointers.
 * 
 */
template <typename Ret, typename... Args>
struct FunctionTraits<Ret(*)(Args...)> {
    using ReturnType = Ret;
    using ArgumentTypes = std::tuple<Args...>;
};

/**
 * @struct FunctionTraits
 * 
 * Specialization for std::function.
 * 
 */
template <typename Ret, typename... Args>
struct FunctionTraits<std::function<Ret(Args...)>> {
    using ReturnType = Ret;
    using ArgumentTypes = std::tuple<Args...>;
};

#include "function_traits.tpp"