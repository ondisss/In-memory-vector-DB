#pragma once

#include "VectorRecord.hpp"

#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

struct HNSWParams {
    std::size_t maxNeighbors    = 16;
    std::size_t buildExploration  = 200;
    std::size_t searchExploration = 50;
};

struct HNSWResult {
    float       score;
    uint32_t    id;
    std::string text;
};

class HNSWIndex {
public:
    explicit HNSWIndex(std::size_t dimension, HNSWParams params = {});

    void build(const std::vector<VectorRecord>& records);

    std::vector<HNSWResult> search(const std::vector<float>& query,
                                   std::size_t               topK,
                                   std::size_t               explorationFactor = 0) const;

    bool        built()     const noexcept;
    std::size_t size()      const noexcept;
    std::size_t dimension() const noexcept;
    int         levels()    const noexcept;

private:
    struct Node {
        uint32_t                              id;
        std::string                           text;
        float                                 magnitude;
        int                                   maxLayer;
        std::vector<std::vector<std::size_t>> neighbors;
    };

    int randomLevel();

    std::vector<std::pair<float, std::size_t>>
    searchLayer(const std::vector<float>& query,
                float                     queryMagnitude,
                std::size_t               entryPoint,
                std::size_t               explorationFactor,
                int                       layer) const;

    std::vector<std::pair<float, std::size_t>>
    searchLayer(const std::vector<float>&                          query,
                float                                              queryMagnitude,
                const std::vector<std::pair<float, std::size_t>>& entryPoints,
                std::size_t                                        explorationFactor,
                int                                                layer) const;

    std::vector<std::size_t>
    selectNeighbors(std::vector<std::pair<float, std::size_t>> candidates,
                    std::size_t                                maxNeighbors) const;

    void pruneConnections(Node& node, std::size_t nodeIndex, int layer, std::size_t maxNeighbors);

    const float* emb(std::size_t idx) const noexcept {
        return embeddings_.data() + idx * dimension_;
    }

    std::size_t        dimension_;
    HNSWParams         params_;
    std::vector<Node>  nodes_;
    std::vector<float> embeddings_;
    std::size_t        entryPoint_        = 0;
    int                maxLevel_          = -1;
    double             levelNormFactor_;
    std::mt19937       rng_;

    mutable std::vector<uint64_t> visitStamp_;
    mutable uint64_t              visitGeneration_ = 0;
};
