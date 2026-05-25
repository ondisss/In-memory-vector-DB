#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct VectorRecord
{
    uint32_t id;
    std::string text;
    std::vector<float> embedding;
};
