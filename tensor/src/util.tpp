namespace tensor{

    template<typename T>
    bool isEqual(const T x, const T y, const T eps){
        return abs(x - y) < eps;
    }
    
}