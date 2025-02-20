namespace tensor{

    // Unary operations
    template <typename T>
    T neg(const T x){
        return -x;
    }

    template <typename T>
    T inv(const T x){
        return 1 / x;
    }

    template <typename T>
    T relu(const T x){
        return x >= 0 ? x : 0;
    }

    template <typename T>
    T sigmoid(const T x){
        return 1 / (1 + exp(x));
    }

    // Binary operations
    template <typename T>
    T sum(const T x, const T y){
        return x + y;
    }

    template <typename T>
    T product(const T x, const T y){
        return x * y;
    }
    
}