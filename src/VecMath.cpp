#include "VecMath.hpp"

#include <cmath>
#include <stdexcept>

namespace VecMath
{

    float dotProduct(const std::vector<float> &a, const std::vector<float> &b)
    {
        if (a.size() != b.size())
            throw std::invalid_argument("dotProduct: vektory mají různé délky");
        float sum = 0.0f;
        for (std::size_t i = 0; i < a.size(); ++i)
            sum += a[i] * b[i];
        return sum;
    }

    float magnitude(const std::vector<float> &v)
    {
        float sum = 0.0f;
        for (float x : v)
            sum += x * x;
        return std::sqrt(sum);
    }

    float cosineSimilarity(const std::vector<float> &a, const std::vector<float> &b)
    {
        float magA = magnitude(a);
        float magB = magnitude(b);
        if (magA < 1e-8f || magB < 1e-8f)
            return 0.0f;
        return dotProduct(a, b) / (magA * magB);
    }

    float euclideanDistance(const std::vector<float> &a, const std::vector<float> &b)
    {
        if (a.size() != b.size())
            throw std::invalid_argument("euclideanDistance: vektory mají různé délky");
        float sum = 0.0f;
        for (std::size_t i = 0; i < a.size(); ++i)
        {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }

    float dotProductRaw(const float *a, const float *b, std::size_t dim)
    {
        float sum = 0.0f;
        for (std::size_t i = 0; i < dim; ++i)
            sum += a[i] * b[i];
        return sum;
    }

    float magnitudeRaw(const float *v, std::size_t dim)
    {
        float sum = 0.0f;
        for (std::size_t i = 0; i < dim; ++i)
            sum += v[i] * v[i];
        return std::sqrt(sum);
    }

}
