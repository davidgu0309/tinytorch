/**
 * @file model.hpp
 * 
 * @brief Model interface and template for DAG models.
 * 
 * @author David Gu
 * @author Mirco Paul
 * 
 * @date \today
 */
#pragma once

#include "computational_dag.hpp"
#include "layer.hpp"
#include "tensor.hpp"

/**
 * @namespace tinytorch
 * 
 * @brief Namespace of the entire framework.
 * 
 */
namespace tinytorch {

/**
 * 
 * Model interface.
 * 
 * @tparam T Floating point data type for numerical computations.
 * 
 **/
template <typename T>
class ModelInterface {
    public:
        /**
         * 
         * Inference for potentially batched input data.
         * 
         * @param input Input data of shape {dim_in} or {batch_size, dim_in}.
         * 
         * @return Tensor of shape {dim_out} or {batch_size, dim_out} containing the predictions for the provided input data.
         * 
         **/
        virtual Tensor<T> forward(const Tensor<T>& input) = 0; 

};

/**
 * 
 * Model class for arbitrary computational DAGs.
 * @tparam T Floating point data type for numerical computations.
 * 
 **/
template <typename T>
class DAGModel : ModelInterface<T>{

        ComputationalDAG<T> computational_dag_;

    public:

        // TO DO: write constructors, remember to set is_topo_order_up_to_date!
        DAGModel();

        // TO DO: add functions to add (and potentially remove) nodes and edges, remember to clear is_topo_order_up_to_date!

        /**
         * 
         * Inference for potentially batched input data.
         * 
         * @param input Input data of shape {dim_in} or {batch_size, dim_in}.
         * 
         * @return Tensor of shape {dim_out} or {batch_size, dim_out} containing the predictions for the provided input data.
         * 
         **/
        Tensor<T> forward(const Tensor<T>& input); 
        
};

// template <typename T>
// class Sequential {
//     public:
//         Sequential(std::vector<Layer<T>> layerList);

//         forward(const Tensor<T>& input); 
//         // {
//         //     for (Layer<T> L: layerList) {
//         //         input = L.forward(input);
//         //     }
//         // }
// };

}

#include "../src/model.tpp"

