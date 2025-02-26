/**
 * @file scalar_operation.hpp
 * 
 * @brief Implementation of common scalar operations with derivative.
 * 
 * @author David Gu
 * @author Mirco Paul
 * 
 * @date \today
 */
#pragma once

#include "cmath"

/**
 * @namespace tinytorch
 * 
 * @brief Namespace of the entire framework.
 * 
 */
namespace tinytorch{

    // TODO: Doxygen document this concept for scalar ops
    template <typename scalarOp, typename T>
    concept ScalarOperation = requires(scalarOp op, const std::vector<T> operands, size_t idx) {
        { op(operands) } -> std::same_as<T>;                    /** () operator: vector<T> -> T */
        { op.backward(idx, operands) } -> std::same_as<T>;        /** backward(int, vector<T>) -> T */
    };

    /**
     * @struct ScalarAddition
     * 
     * @brief Scalar addition operation with derivative.
     * 
     * Functor-style scalar addition operation differentiable with respect to inputs.
     * 
     * @tparam T Floating point data type for numerical computations.
     * 
     **/
    template <typename T>
    struct ScalarAddition{

        ScalarAddition();

        /**
         * 
         * Scalar addition.
         * 
         * @param operands Operands.
         * 
         * @return Sum of the operands.
         * 
         **/
        T operator()(const std::vector<T> operands) const;

        /**
         * 
         * Partial derivative of the scalar addition with respect to input input_idx.
         * 
         * @param input_idx Index of input with respect to which the derivative is computed.
         * 
         * @param operands Operands (point at which the derivative is evaluated).
         * 
         * @return Partial derivative of the addition with respect to input input_idx.
         * 
         **/
        T backward(const size_t input_idx, const std::vector<T> operands) const;

    };

    /**
     * @struct ScalarMultiplication
     * 
     * @brief Scalar multiplication operation with derivative.
     * 
     * Functor-style scalar addition operation differentiable with respect to inputs.
     * 
     * @tparam T Floating point data type for numerical computations.
     * 
     **/
    template <typename T>
    struct ScalarMultiplication{

        ScalarMultiplication();

        /**
         * 
         * Scalar multiplication.
         * 
         * @param operands Operands.
         * 
         * @return Product of the operands.
         * 
         **/
        T operator()(const std::vector<T> operands) const;

        /**
         * 
         * Partial derivative of the scalar multiplication with respect to input input_idx.
         * 
         * @param input_idx Index of input with respect to which the derivative is computed.
         * 
         * @param operands Operands (point at which the derivative is evaluated).
         * 
         * @return Partial derivative of the multiplication with respect to input input_idx.
         * 
         **/
        T backward(const size_t input_idx, const std::vector<T> operands) const;

    };

    /**
     * @struct ScalarPow
     * 
     * @brief Scalar power operation with derivative.
     * 
     * Functor-style scalar power operation differentiable with respect to inputs.
     * 
     * @tparam T Floating point data type for numerical computations.
     * 
     **/
    template <typename T>
    struct ScalarPow{

        ScalarPow();

        /**
         * 
         * Scalar power.
         * 
         * @param operands Operands.
         * 
         * @return Scalar power operands[0] ^ operands[1].
         * 
         **/
        T operator()(const std::vector<T> operands) const;

        /**
         * 
         * Partial derivative of the scalar power operands[0] ^ operands[1].
         * 
         * @param input_idx 0 for base, 1 for exponent.
         * 
         * @param operands Operands (point at which the derivative is evaluated).
         * 
         * @return Partial derivative of the power with respect to input input_idx.
         * 
         * @pre operands.size() == 2
         * 
         **/
        T backward(const size_t input_idx, const std::vector<T> operands) const;

    };

    /**
     * @struct ScalarLog
     * 
     * @brief Scalar logarithm operation with derivative.
     * 
     * Functor-style scalar logarithm operation differentiable with respect to inputs.
     * 
     * @tparam T Floating point data type for numerical computations.
     * 
     **/
    template <typename T>
    struct ScalarLog{

        ScalarLog();

        /**
         * 
         * Scalar logarithm.
         * 
         * @param operands Operands.
         * 
         * @return Scalar logarithm log_operands[0](operands[1]).
         * 
         **/
        T operator()(const std::vector<T> operands) const;

        /**
         * 
         * Partial derivative of the scalar logarithm log_operands[0](operands[1]).
         * 
         * @param input_idx 0 for base, 1 for operand.
         * 
         * @param operands Operands (point at which the derivative is evaluated).
         * 
         * @return Partial derivative of the logarithm with respect to input input_idx.
         * 
         * @pre operands.size() == 2
         * 
         **/
        T backward(const size_t input_idx, const std::vector<T> operands) const;

    };

    /**
     * @struct ScalarReLU
     * 
     * @brief Scalar rectified linear operation with derivative.
     * 
     * Functor-style scalar rectified linear operation differentiable with respect to inputs.
     * 
     * @tparam T Floating point data type for numerical computations.
     * 
     **/
    template <typename T>
    struct ScalarReLU{

        ScalarReLU();

        /**
         * 
         * Scalar rectified linear unit.
         * 
         * @param operands Operand.
         * 
         * @return Rectified linear unit of the operand.
         * 
         **/
        T operator()(const std::vector<T> operands) const;

        /**
         * 
         * Partial derivative of the scalar rectified linear unit with respect to input input_idx 0.
         * 
         * @param input_idx Unnecessary (TODO: handle unary and binary ops separately).
         * 
         * @param operands Operands (point at which the derivative is evaluated).
         * 
         * @return Partial derivative of the rectified linear unit with respect to input input_idx.
         * 
         * @pre operands.size() == 1
         * 
         **/
        T backward(const size_t input_idx, const std::vector<T> operands) const;

    };

    /**
     * @struct ScalarSin
     * 
     * @brief Scalar sine operation with derivative.
     * 
     * Functor-style scalar sine operation differentiable with respect to inputs.
     * 
     * @tparam T Floating point data type for numerical computations.
     * 
     **/
    template <typename T>
    struct ScalarSin{

        ScalarSin();

        /**
         * 
         * Scalar sine.
         * 
         * @param operands Operand.
         * 
         * @return Sine of the operand.
         * 
         **/
        T operator()(const std::vector<T> operands) const;

        /**
         * 
         * Partial derivative of the sine with respect to input.
         * 
         * @param input_idx Unnecessary (TODO: handle unary and binary ops separately).
         * 
         * @param operands Operand (point at which the derivative is evaluated).
         * 
         * @return Partial derivative of the sine with respect to input input_idx.
         * 
         * @pre operands.size() == 1.
         * 
         **/
        T backward(const size_t input_idx, const std::vector<T> operands) const;

    };

    /**
     * @struct ScalarCos
     * 
     * @brief Scalar cosine operation with derivative.
     * 
     * Functor-style scalar cosine operation differentiable with respect to inputs.
     * 
     * @tparam T Floating point data type for numerical computations.
     * 
     **/
    template <typename T>
    struct ScalarCos{

        ScalarCos();

        /**
         * 
         * Scalar cosine.
         * 
         * @param operands Operand.
         * 
         * @return Cosine of the operand.
         * 
         **/
        T operator()(const std::vector<T> operands) const;

        /**
         * 
         * Partial derivative of the cosine with respect to input.
         * 
         * @param input_idx Unnecessary (TODO: handle unary and binary ops separately).
         * 
         * @param operands Operand (point at which the derivative is evaluated).
         * 
         * @return Partial derivative of the cosine with respect to input input_idx.
         * 
         * @pre operands.size() == 1
         * 
         **/
        T backward(const size_t input_idx, const std::vector<T> operands) const;

    };

    /**
     * @struct ScalarTan
     * 
     * @brief Scalar tangent operation with derivative.
     * 
     * Functor-style scalar tangent operation differentiable with respect to inputs.
     * 
     * @tparam T Floating point data type for numerical computations.
     * 
     **/
    template <typename T>
    struct ScalarTan{

        ScalarTan();

        /**
         * 
         * Scalar tangent.
         * 
         * @param operands Operand.
         * 
         * @return Tangent of the operands.
         * 
         **/
        T operator()(const std::vector<T> operands) const;

        /**
         * 
         * Partial derivative of the tanget with respect to input.
         * 
         * @param input_idx Unnecessary (TODO: handle unary and binary ops separately).
         * 
         * @param operands Operand (point at which the derivative is evaluated).
         * 
         * @return Partial derivative of the tangent with respect to input input_idx.
         * 
         * @pre operands.size() == 1.
         * 
         **/
        T backward(const size_t input_idx, const std::vector<T> operands) const;

    };

    // TODO: inverse trig

    /**
     * @struct ScalarSinH
     * 
     * @brief Scalar hyperbolic sine operation with derivative.
     * 
     * Functor-style scalar hyperbolic sine operation differentiable with respect to inputs.
     * 
     * @tparam T Floating point data type for numerical computations.
     * 
     **/
    template <typename T>
    struct ScalarSinH{

        ScalarSinH();

        /**
         * 
         * Scalar hyperbolic sine.
         * 
         * @param operands Operand.
         * 
         * @return Hyperbolic sine of the operand.
         * 
         **/
        T operator()(const std::vector<T> operands) const;

        /**
         * 
         * Partial derivative of the hyperbolic sine with respect to input.
         * 
         * @param input_idx Unnecessary (TODO: handle unary and binary ops separately).
         * 
         * @param operands Operand (point at which the derivative is evaluated).
         * 
         * @return Partial derivative of the hyperbolic sine with respect to input.
         * 
         * @pre operands.size() == 1.
         * 
         **/
        T backward(const size_t input_idx, const std::vector<T> operands) const;

    };

    /**
     * @struct ScalarCosH
     * 
     * @brief Scalar hyperbolic cosine operation with derivative.
     * 
     * Functor-style scalar hyperbolic cosine operation differentiable with respect to inputs.
     * 
     * @tparam T Floating point data type for numerical computations.
     * 
     **/
    template <typename T>
    struct ScalarCosH{

        ScalarCosH();

        /**
         * 
         * Scalar hyperbolic cosine.
         * 
         * @param operands Operand.
         * 
         * @return Hyperbolic cosine of the operand.
         * 
         **/
        T operator()(const std::vector<T> operands) const;

        /**
         * 
         * Partial derivative of the hyperbolic cosine with respect to input.
         * 
         * @param input_idx Unnecessary (TODO: handle unary and binary ops separately).
         * 
         * @param operands Operand (point at which the derivative is evaluated).
         * 
         * @return Partial derivative of the hyperbolic cosine with respect to input input_idx.
         * 
         * @pre operands.size() == 1
         * 
         **/
        T backward(const size_t input_idx, const std::vector<T> operands) const;

    };

    /**
     * @struct ScalarTanH
     * 
     * @brief Scalar hyperbolic tangent operation with derivative.
     * 
     * Functor-style scalar hyperbolic tangent operation differentiable with respect to inputs.
     * 
     * @tparam T Floating point data type for numerical computations.
     * 
     **/
    template <typename T>
    struct ScalarTanH{

        ScalarTanH();

        /**
         * 
         * Scalar hyperbolic tangent.
         * 
         * @param operands Operand.
         * 
         * @return Hyperbolic tangent of the operands.
         * 
         **/
        T operator()(const std::vector<T> operands) const;

        /**
         * 
         * Partial derivative of the hyperbolic tanget with respect to input.
         * 
         * @param input_idx Unnecessary (TODO: handle unary and binary ops separately).
         * 
         * @param operands Operand (point at which the derivative is evaluated).
         * 
         * @return Partial derivative of the hyperbolic tangent with respect to input input_idx.
         * 
         * @pre operands.size() == 1.
         * 
         **/
        T backward(const size_t input_idx, const std::vector<T> operands) const;

    };

    // TODO: inverse hyperbolic trig

    /**
     * @struct ScalarSgn
     * 
     * @brief Scalar sign operation with derivative.
     * 
     * Functor-style scalar sign operation differentiable with respect to inputs.
     * 
     * @tparam T Floating point data type for numerical computations.
     * 
     **/
    template <typename T>
    struct ScalarSgn{

        ScalarSgn();

        /**
         * 
         * Scalar sign.
         * 
         * @param operands Operand.
         * 
         * @return Sign of the operand.
         * 
         **/
        T operator()(const std::vector<T> operands) const;

        /**
         * 
         * Partial derivative of the sign with respect to input input_idx 0.
         * 
         * @param input_idx Unnecessary (TODO: handle unary and binary ops separately).
         * 
         * @param operands Operand (point at which the derivative is evaluated).
         * 
         * @return Partial derivative of the sign with respect to input input_idx.
         * 
         * @pre operands.size() == 1
         * 
         **/
        T backward(const size_t input_idx, const std::vector<T> operands) const;

    };

    /**
     * @struct ScalarAbs
     * 
     * @brief Scalar absolute value operation with derivative.
     * 
     * Functor-style scalar absolute value operation differentiable with respect to inputs.
     * 
     * @tparam T Floating point data type for numerical computations.
     * 
     **/
    template <typename T>
    struct ScalarAbs{

        ScalarAbs();

        /**
         * 
         * Scalar absolute value.
         * 
         * @param operands Operand.
         * 
         * @return Absolute value of the operands.
         * 
         **/
        T operator()(const std::vector<T> operands) const;

        /**
         * 
         * Partial derivative of the absolute value with respect to input input_idx 0.
         * 
         * @param input_idx Unnecessary (TODO: handle unary and binary ops separately).
         * 
         * @param operands Operand (point at which the derivative is evaluated).
         * 
         * @return Partial derivative of the absolute value with respect to input input_idx.
         * 
         * @pre operands.size() == 1
         * 
         **/
        T backward(const size_t input_idx, const std::vector<T> operands) const;

    };

    /**
     * @struct Scalar???
     * 
     * @brief Scalar ??? operation with derivative.
     * 
     * Functor-style scalar ??? operation differentiable with respect to inputs.
     * 
     * @tparam T Floating point data type for numerical computations.
     * 
     **/
    template <typename T>
    struct ScalarNeg{

        ScalarNeg();

        /**
         * 
         * Scalar negation.
         * 
         * @param operands Operand.
         * 
         * @return Negation of the operands.
         * 
         **/
        T operator()(const std::vector<T> operands) const;

        /**
         * 
         * Partial derivative of the negation with respect to input.
         * 
         * @param input_idx Unnecessary (TODO: handle unary and binary ops separately).
         * 
         * @param operands Operand (point at which the derivative is evaluated).
         * 
         * @return Partial derivative of the negation with respect to input input_idx.
         * 
         * @pre operands.size() == 1
         * 
         **/
        T backward(const size_t input_idx, const std::vector<T> operands) const;

    };

    /**
     * 
     * @todo Better comments and names in LaTeX.
     * 
     * @struct ScalarLp
     * 
     * @brief Scalar l^p operation with derivative.
     * 
     * Functor-style scalar l^p operation differentiable with respect to inputs.
     * 
     * @tparam T Floating point data type for numerical computations.
     * 
     **/
    template <typename T>
    struct ScalarLp{

        ScalarLp();

        /**
         * 
         * Scalar l^p.
         * 
         * @param operands Operand.
         * 
         * @return l^p operation of the operands.
         * 
         **/
        T operator()(const std::vector<T> operands) const;

        /**
         * 
         * Partial derivative of the l^p with respect to input input_idx 0.
         * 
         * @param input_idx Unnecessary (TODO: handle unary and binary ops separately).
         * 
         * @param operands Operand (point at which the derivative is evaluated).
         * 
         * @return Partial derivative of the l^p with respect to input input_idx.
         * 
         * @pre operands.size() == 2
         * 
         **/
        T backward(const size_t input_idx, const std::vector<T> operands) const;

    };

    /**
     * @struct ScalarSigmoid
     * 
     * @brief Scalar sigmoid operation with derivative.
     * 
     * Functor-style scalar sigmoid operation differentiable with respect to inputs.
     * 
     * @tparam T Floating point data type for numerical computations.
     * 
     **/
    template <typename T>
    struct ScalarSigmoid{

        ScalarSigmoid();

        /**
         * 
         * Scalar sigmoid.
         * 
         * @param operands Operand.
         * 
         * @return Sigmoid of the operands.
         * 
         **/
        T operator()(const std::vector<T> operands) const;

        /**
         * 
         * Partial derivative of the sigmoid with respect to input input_idx 0.
         * 
         * @param input_idx Unnecessary (TODO: handle unary and binary ops separately).
         * 
         * @param operands Operand (point at which the derivative is evaluated).
         * 
         * @return Partial derivative of the sigmoid with respect to input input_idx.
         * 
         * @pre operands.size() == 1
         * 
         **/
        T backward(const size_t input_idx, const std::vector<T> operands) const;

    };

    /**
     * @struct Scalar???
     * 
     * @brief Scalar ??? operation with derivative.
     * 
     * Functor-style scalar ??? operation differentiable with respect to inputs.
     * 
     * @tparam T Floating point data type for numerical computations.
     * 
     **/
    template <typename T>
    struct ScalarMean{

        ScalarMean();

        /**
         * 
         * Scalar mean.
         * 
         * @param operands Operand.
         * 
         * @return Mean of the operands.
         * 
         **/
        T operator()(const std::vector<T> operands) const;

        /**
         * 
         * Partial derivative of the mean with respect to input input_idx 0.
         * 
         * @param input_idx Unnecessary (TODO: handle unary and binary ops separately).
         * 
         * @param operands Operand (point at which the derivative is evaluated).
         * 
         * @return Partial derivative of the mean with respect to input input_idx.
         * 
         * @pre operands.size() == 1
         * 
         **/
        T backward(const size_t input_idx, const std::vector<T> operands) const;

    };

    /**
     * @struct ScalarCobbDouglas
     * 
     * @brief Scalar Cobb-Douglas operation with derivative.
     * 
     * Functor-style scalar Cobb-Douglas operation differentiable with respect to inputs.
     * 
     * @tparam T Floating point data type for numerical computations.
     * 
     **/
    template <typename T>
    struct ScalarCobbDouglas{

        ScalarCobbDouglas();

        /**
         * 
         * Scalar Cobb-Douglas.
         * 
         * @param operands Operands.
         * 
         * @return Cobb-Douglas of the operands.
         * 
         **/
        T operator()(const std::vector<T> operands) const;

        /**
         * 
         * Partial derivative of the Cobb-Douglas with respect to input input_idx 0.
         * 
         * @param input_idx Unnecessary (TODO: handle unary and binary ops separately).
         * 
         * @param operands Operand (point at which the derivative is evaluated).
         * 
         * @return Partial derivative of the Cobb-Douglas with respect to input input_idx.
         * 
         * @pre operands.size() == 1
         * 
         **/
        T backward(const size_t input_idx, const std::vector<T> operands) const;

    };

}  // namespace tinytorch

#include "../../src/scalar_operation/scalar_operation.tpp"