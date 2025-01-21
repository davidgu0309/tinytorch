namespace tinytorch {

    // TO DO: scalar * tensor, maybe generic broadcasting

    // Unary operations
    template<typename T, typename U, U (*unaryOp)(T)>
    Tensor<U> applyUnaryOp(const Tensor<T>& a){
        const std::vector<T>& data_a = a.data();  
        Tensor<U> result(a.shape());
        std::vector<U>& result_data = result.data();
        for (int i=0; i<data_a.size(); i++) {
            result_data[i] = unaryOp(data_a[i]);
        }
        return result;
    }

    template <typename T>
    Tensor<T> neg(const Tensor<T>& a){
        return applyUnaryOp<T, T, neg<T>>(a);
    }

    template <typename T>
    Tensor<T> inv(const Tensor<T>& a){
        return applyUnaryOp<T, T, inv<T>>(a);
    }

    template <typename T>
    Tensor<T> relu(const Tensor<T>& a){
        return applyUnaryOp<T, T, relu<T>>(a);
    }

    template <typename T>
    Tensor<T> sigmoid(const Tensor<T>& a){
        return applyUnaryOp<T, T, sigmoid<T>>(a);
    }

    // Binary operations
    template<typename T, typename U, typename V, V (*binaryOp)(T, U)>
    Tensor<V> applyBinaryOp(const Tensor<T>& a, const Tensor<U>& b){
        assert(a.shapeEqual(b));
        const std::vector<T>& data_a = a.data();  
        const std::vector<U>& data_b = b.data();  
        Tensor<V> result(a.shape());
        std::vector<V>& result_data = result.data();
        for (int i=0; i<data_a.size(); i++) {
            result_data[i] = binaryOp(data_a[i], data_b[i]);
        }
        return result;
    }

    template <typename T>
    Tensor<T> add(const Tensor<T>& a, const Tensor<T>& b) {
        return applyBinaryOp<T, T, T, sum<T>>(a, b);
    }
    

    template <typename T>
    Tensor<T> mul(const Tensor<T>& a, const Tensor<T>& b) {
        return applyBinaryOp<T, T, T, product<T>>(a, b);
    }

    // a_shape = {a_1, ..., a_n}, b_shape = {b_1, ..., b_m}, ab_shape = {a_1, ..., a_(n - 1), b_2, ..., b_m}
    Shape matmulShape(const Shape a_shape, const Shape b_shape){
        
        size_t dim_a = a_shape.size(), dim_b = b_shape.size();
        
        // Compatibility test
        assert(dim_a && dim_b && a_shape.back() == b_shape.front());    // TO DO: return {} if one of the two is {}

        // Compute result_shape
        Shape result_shape = a_shape;
        result_shape.pop_back();
        for(size_t d = 1; d < dim_b; ++d) result_shape.push_back(b_shape[d]);

        return result_shape;
    }

    Shape combineIndexes(const MultiIndex& i, const MultiIndex& j){

        size_t dim_j = j.size();
        
        MultiIndex ij = i;
        for(size_t d = 1; d < dim_j; ++d) ij.push_back(j[d]);

        return ij;
    }

    // Unnecessary and inefficient, but nice
    std::vector<MultiIndex> indexesRowMajor(const Shape shape){
        std::queue<MultiIndex> indexes; // Multiindexes in "row-major" order
        indexes.push({});
        size_t n = shape.size();
        //shape = {2, 2, 3}
        //
        // {}
        // {0}, {1}
        // {0, 0}, {0, 1}, {1, 0}, {1, 1}
        // ...
        for(size_t d = 0; d < n; ++d){
            size_t m = indexes.size();
            // Iterate over all multiindexes of the previous dimension
            for(size_t i = 0; i < m; ++i){
                // For each one, add all possible indexes for the current dimension
                for(size_t j = 0; j < shape[d]; ++j){
                    MultiIndex index = indexes.front();
                    index.push_back(j);
                    indexes.push(index);
                }
                indexes.pop();
            }
        }
        std::vector<MultiIndex> row_major(indexes.size());
        size_t i = 0;
        while(indexes.size()){
            row_major[i] = indexes.front();
            indexes.pop();
            ++i;
        }
        return row_major;
    }

    template <typename T>
    Tensor<T> matmul(const Tensor<T>& a, const Tensor<T>& b){

        Shape a_shape = a.shape(), b_shape = b.shape();

        // Compute result shape
        Shape result_shape = matmulShape(a_shape, b_shape);
        // Initialize result with 0s
        Tensor<T> result = zeros<T>(result_shape);

        a_shape.pop_back();
        // Could replace b_indexes = indexesColumnMajor(b_shape) for cache
        std::vector<MultiIndex> a_indexes = indexesRowMajor(a_shape), b_indexes = indexesRowMajor(b_shape);

        /*
        a:
        shape = {3, 2} : {0} {1} {2}
        1 2 
        3 4 
        5 6

        b:
        shape = {2, 2} : {0, 0} {0, 1} {1, 0} {1, 1}
        1 2
        3 4

        result:
        0 0
        0 0
        0 0
        */

        // For 2D, this is O(n ** 3)
        for(const MultiIndex& i : a_indexes){
            for(const MultiIndex& j : b_indexes){
                MultiIndex ii(i); ii.push_back(j.front());
                result.getEntrySafe(combineIndexes(i, j)) += a.getEntrySafe(ii) * b.getEntrySafe(j);
            }
        }

        return result;
    }

}
