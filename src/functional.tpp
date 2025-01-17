#include "../include/utils.hpp"

namespace tinytorch {

    template<typename T, typename U, U (*unaryOp)(T)>
    Tensor<U> applyUnaryOp(const Tensor<T>& a){
        std::vector<T> data_a = *a.data();  
        Tensor<U> result(a.shape());
        std::vector<U>& result_data = *result.data();
        for (int i=0; i<data_a.size(); i++) {
            result_data[i] = unaryOp(data_a[i]);
        }
        return result;
    }

    template <typename T>
    T scalarNeg(const T& a){
        return -a;
    }

    template <typename T>
    Tensor<T> neg(const Tensor<T>& a){
        return applyUnaryOp<T, T, scalarNeg>(a);
    }

    template <typename T>
    T inv(const T& a){
        return 1 / a;
    }

    template <typename T>
    Tensor<T> inv(const Tensor<T>& a){
        return applyUnaryOp<T, T, inv>(a);
    }

    template<typename T, typename U, typename V, V (*binaryOp)(T, U)>
    Tensor<V> applyBinaryOp(const Tensor<T>& a, const Tensor<U>& b){
        assert(a.shapeEqual(b));
        std::vector<T> data_a = *a.data();  
        std::vector<U> data_b = *b.data();  
        Tensor<V> result(a.shape());
        std::vector<V>& result_data = *result.data();
        for (int i=0; i<data_a.size(); i++) {
            result_data[i] = binaryOp(data_a[i], data_b[i]);
        }
        return result;
    }

    template <typename T>
    T sum(const T& a, const T& b){
        return a + b;
    }
    
    template <typename T>
    Tensor<T> add(const Tensor<T>& a, const Tensor<T>& b) {
        // assert(a.shapeEqual(b));
        // Tensor<T> result(a.shape());    // Do we need new? No, as long as we return a copy and not a reference!
        // std::vector<T>& result_data = *result.data();
        // std::vector<T> dataA = *a.data();
        // std::vector<T> dataB = *b.data();   
        // for (int i=0; i<dataA.size(); i++) {
        //     result_data[i] = dataA[i] + dataB[i];
        // }
        // return result;
        return applyBinaryOp<T, T, T, sum>(a, b);
    }

    template <typename T>
    T product(const T& a, const T& b){
        return a * b;
    }

    template <typename T>
    Tensor<T> mul(const Tensor<T>& a, const Tensor<T>& b) {
        return applyBinaryOp<T, T, T, product>(a, b);
    }

}
