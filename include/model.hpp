#pragma once

#include "computational_dag.hpp"
#include "layer.hpp"


namespace tinytorch {

/**
 * 
 * Model interface.
 * @tparam T Floating point data type for numerical computations.
 * 
 **/
template <typename T>
class ModelInterface {
    public:
        /**
         * 
         * Inference for potentially batched input data.
         * @param input Input data of shape {dim_in} or {batch_size, dim_in}.
         * @return Tensor of shape {dim_out} or {batch_size, dim_out} containing the predictions for the provided input data.
         * 
         **/
        virtual Tensor<T> forward(const Tensor<T>& input) = 0; 

};

template <typename T>
class DAGModel : ModelInterface<T>{
    public:
        Tensor<T> forward(const Tensor<T>& input); 
    
    private:
        DAGNode<T> dagEntryPoint;

        
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

