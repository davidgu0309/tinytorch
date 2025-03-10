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
    /**
     * 
     * 
     * @pre data[0] is X, data[1] is y.
     * @pre Model is a computational DAG computing the loss.
     * 
     * @param node_data Node data.
     * 
     * @return Identifier of the new node.
     * 
     **/
    template<typename T>
    void gradient_descent_iteration<T>(ComputationalDAG<T>& model, const Tensor<T>& data, const T learning_rate);

    /**
     * 
     * 
     * @pre data[0] is X, data[1] is y.
     * @pre Model is a computational DAG computing the loss.
     * 
     * @param node_data Node data.
     * 
     * @return Identifier of the new node.
     * 
     **/
    template<typename T>
    void gradient_descent<T>(ComputationalDAG<T>& model, const Tensor<T>& data, const T learning_rate, const size_t epochs);

} // namespace tinytorch

#include "../../src/optimizer/gd.tpp"