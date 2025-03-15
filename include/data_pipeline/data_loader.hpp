/**
 * @file data_loader.hpp
 * 
 * @brief Data loader for .csv files.
 * 
 * @author David Gu
 * @author Mirco Paul
 * 
 * @date \today
 */
#pragma once

#include "../../tensor/include/tensor.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace tensor;

namespace tinytorch{

    /**
     * 
     * Load a .csv file of values of type T into a vector of batches suitable for the optimizers.
     * A batch is a Tensor<T> with data points stacked along axis 0.
     * 
     * @param path Path to the data to load.
     * @param batch_size Last batch may be smaller than the others.
     * 
     * @return Vector of batches.
     * 
     **/
    template<typename T>
    std::vector<Tensor<T>> load_csv(std::string path, size_t batch_size);

} // namespace tinytorch

#include "../../src/data_pipeline/data_loader.tpp"