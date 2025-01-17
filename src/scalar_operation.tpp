namespace tinytorch{

    // Unary operations
    template <typename T>
    T neg(const T& a){
        return -a;
    }

    template <typename T>
    T inv(const T& a){
        return 1 / a;
    }

    // Binary operations
    template <typename T>
    T sum(const T& a, const T& b){
        return a + b;
    }

    template <typename T>
    T product(const T& a, const T& b){
        return a * b;
    }
    
}