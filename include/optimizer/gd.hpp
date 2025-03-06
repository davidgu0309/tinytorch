/**
 * @file gd.hpp
 * 
 * @brief Gradient descent for computational DAGs.
 * 
 * @author David Gu
 * @author Mirco Paul
 * 
 * @date \today
 */
#pragma once

#include "../computational_dag.hpp"

namespace tinytorch{

    template<typename T>
    void gradient_descent<T>(ComputationalDAG<T>& model, const Tensor<T>& data);

} // namespace tinytorch

#include "../../src/optimizer/gd.tpp"