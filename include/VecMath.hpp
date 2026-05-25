#pragma once

#include <cstddef>
#include <vector>

namespace VecMath
{

    float dotProduct(const std::vector<float> &a, const std::vector<float> &b);
    float magnitude(const std::vector<float> &v);
    float cosineSimilarity(const std::vector<float> &a, const std::vector<float> &b);
    float euclideanDistance(const std::vector<float> &a, const std::vector<float> &b);

    float dotProductRaw(const float *a, const float *b, std::size_t dim);
    float magnitudeRaw(const float *v, std::size_t dim);

}
