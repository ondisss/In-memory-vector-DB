#include "HNSWIndex.hpp"
#include "VecMath.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <queue>
#include <stdexcept>

HNSWIndex::HNSWIndex(std::size_t dimension, HNSWParams params)
    : dimension_(dimension),
      params_(params),
      levelNormFactor_(1.0 / std::log(static_cast<double>(params.maxNeighbors))),
      rng_(std::random_device{}())
{
    if (dimension == 0)
        throw std::invalid_argument("HNSWIndex: dimenze musí být > 0");
    if (params.maxNeighbors < 2)
        throw std::invalid_argument("HNSWIndex: maxNeighbors musí být >= 2");
}

int HNSWIndex::randomLevel()
{
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double r = dist(rng_);
    if (r == 0.0)
        r = 1e-15;
    return static_cast<int>(-std::log(r) * levelNormFactor_);
}

std::vector<std::pair<float, std::size_t>> HNSWIndex::searchLayer(
    const std::vector<float> &query,
    float queryMagnitude,
    std::size_t entryPoint,
    std::size_t explorationFactor,
    int layer) const
{
    using Pair = std::pair<float, std::size_t>;
    const float denom = queryMagnitude * nodes_[entryPoint].magnitude;
    float score = (denom < 1e-8f) ? 0.0f
                                  : VecMath::dotProductRaw(query.data(), emb(entryPoint), dimension_) / denom;
    return searchLayer(query, queryMagnitude, std::vector<Pair>{{score, entryPoint}}, explorationFactor, layer);
}

std::vector<std::pair<float, std::size_t>>
HNSWIndex::searchLayer(const std::vector<float> &query,
                       float queryMagnitude,
                       const std::vector<std::pair<float, std::size_t>> &entryPoints,
                       std::size_t explorationFactor,
                       int layer) const
{
    using Pair = std::pair<float, std::size_t>;

    ++visitGeneration_;
    if (visitStamp_.size() < nodes_.size())
        visitStamp_.resize(nodes_.size(), 0);

    auto markVisited = [&](std::size_t idx)
    { visitStamp_[idx] = visitGeneration_; };
    auto isVisited = [&](std::size_t idx)
    { return visitStamp_[idx] == visitGeneration_; };

    const float *queryRaw = query.data();

    std::priority_queue<Pair> candidates;
    std::priority_queue<Pair, std::vector<Pair>, std::greater<Pair>> results;

    for (const auto &[score, idx] : entryPoints)
    {
        if (isVisited(idx))
            continue;
        markVisited(idx);
        candidates.push({score, idx});
        results.push({score, idx});
    }

    while (!candidates.empty())
    {
        auto [candidateScore, candidateIndex] = candidates.top();
        candidates.pop();

        if (results.size() >= explorationFactor && candidateScore < results.top().first)
            break;

        for (std::size_t neighborIndex : nodes_[candidateIndex].neighbors[static_cast<std::size_t>(layer)])
        {
            if (isVisited(neighborIndex))
                continue;
            markVisited(neighborIndex);

            const float denom_nb = queryMagnitude * nodes_[neighborIndex].magnitude;
            float neighborScore = (denom_nb < 1e-8f) ? 0.0f
                                                     : VecMath::dotProductRaw(queryRaw, emb(neighborIndex), dimension_) / denom_nb;

            if (results.size() < explorationFactor || neighborScore > results.top().first)
            {
                candidates.push({neighborScore, neighborIndex});
                results.push({neighborScore, neighborIndex});
                if (results.size() > explorationFactor)
                    results.pop();
            }
        }
    }

    std::vector<Pair> sortedResults;
    sortedResults.reserve(results.size());
    while (!results.empty())
    {
        sortedResults.push_back(results.top());
        results.pop();
    }
    std::reverse(sortedResults.begin(), sortedResults.end());
    return sortedResults;
}

std::vector<std::size_t>
HNSWIndex::selectNeighbors(std::vector<std::pair<float, std::size_t>> candidates,
                           std::size_t maxNeighbors) const
{
    std::sort(candidates.begin(), candidates.end(),
              [](const auto &a, const auto &b)
              { return a.first > b.first; });

    std::size_t count = std::min(maxNeighbors, candidates.size());
    std::vector<std::size_t> result;
    result.reserve(count);
    for (std::size_t i = 0; i < count; ++i)
        result.push_back(candidates[i].second);
    return result;
}

void HNSWIndex::pruneConnections(Node &node, std::size_t nodeIndex, int layer, std::size_t maxNeighbors)
{
    auto &neighbors = node.neighbors[static_cast<std::size_t>(layer)];
    if (neighbors.size() <= maxNeighbors)
        return;

    const float *nodeEmb = emb(nodeIndex);
    std::vector<std::pair<float, std::size_t>> scored;
    scored.reserve(neighbors.size());
    for (std::size_t neighborIndex : neighbors)
    {
        const float denom = node.magnitude * nodes_[neighborIndex].magnitude;
        float score = (denom < 1e-8f) ? 0.0f
                                      : VecMath::dotProductRaw(nodeEmb, emb(neighborIndex), dimension_) / denom;
        scored.push_back({score, neighborIndex});
    }

    std::partial_sort(scored.begin(),
                      scored.begin() + static_cast<std::ptrdiff_t>(maxNeighbors),
                      scored.end(),
                      [](const auto &a, const auto &b)
                      { return a.first > b.first; });

    neighbors.resize(maxNeighbors);
    for (std::size_t i = 0; i < maxNeighbors; ++i)
        neighbors[i] = scored[i].second;
}

void HNSWIndex::build(const std::vector<VectorRecord> &records)
{
    nodes_.clear();
    embeddings_.clear();
    nodes_.reserve(records.size());
    embeddings_.reserve(records.size() * dimension_);
    maxLevel_ = -1;

    for (const VectorRecord &rec : records)
    {
        if (rec.embedding.size() != dimension_)
            continue;

        embeddings_.insert(embeddings_.end(), rec.embedding.begin(), rec.embedding.end());

        Node node;
        node.id = rec.id;
        node.text = rec.text;
        node.magnitude = VecMath::magnitudeRaw(rec.embedding.data(), dimension_);
        node.maxLayer = randomLevel();
        node.neighbors.resize(static_cast<std::size_t>(node.maxLayer) + 1);

        std::size_t newNodeIndex = nodes_.size();
        nodes_.push_back(std::move(node));

        if (newNodeIndex == 0)
        {
            entryPoint_ = 0;
            maxLevel_ = nodes_[0].maxLayer;
            continue;
        }

        std::size_t entryPoint = entryPoint_;
        const float newNodeMagnitude = nodes_[newNodeIndex].magnitude;

        for (int currentLayer = maxLevel_; currentLayer > nodes_[newNodeIndex].maxLayer; --currentLayer)
        {
            auto nearest = searchLayer(rec.embedding, newNodeMagnitude, entryPoint, 1, currentLayer);
            if (!nearest.empty())
                entryPoint = nearest[0].second;
        }

        std::vector<std::pair<float, std::size_t>> candidateSet;
        {
            const float d0 = newNodeMagnitude * nodes_[entryPoint].magnitude;
            float s0 = (d0 < 1e-8f) ? 0.0f
                                    : VecMath::dotProductRaw(rec.embedding.data(), emb(entryPoint), dimension_) / d0;
            candidateSet.push_back({s0, entryPoint});
        }

        for (int currentLayer = std::min(nodes_[newNodeIndex].maxLayer, maxLevel_); currentLayer >= 0; --currentLayer)
        {
            std::size_t maxNeighborsForLayer = (currentLayer == 0) ? 2 * params_.maxNeighbors : params_.maxNeighbors;

            auto candidates = searchLayer(rec.embedding, newNodeMagnitude, candidateSet, params_.buildExploration, currentLayer);
            auto neighbors = selectNeighbors(candidates, maxNeighborsForLayer);

            nodes_[newNodeIndex].neighbors[static_cast<std::size_t>(currentLayer)] = neighbors;
            for (std::size_t neighborIndex : neighbors)
            {
                nodes_[neighborIndex].neighbors[static_cast<std::size_t>(currentLayer)].push_back(newNodeIndex);
                pruneConnections(nodes_[neighborIndex], neighborIndex, currentLayer, maxNeighborsForLayer);
            }

            candidateSet = candidates;
        }

        if (nodes_[newNodeIndex].maxLayer > maxLevel_)
        {
            maxLevel_ = nodes_[newNodeIndex].maxLayer;
            entryPoint_ = newNodeIndex;
        }
    }
}

std::vector<HNSWResult>
HNSWIndex::search(const std::vector<float> &query,
                  std::size_t topK,
                  std::size_t explorationFactor) const
{
    if (!built() || topK == 0)
        return {};
    if (query.size() != dimension_)
        throw std::invalid_argument("HNSWIndex::search: špatná dimenze dotazu");

    if (explorationFactor == 0)
        explorationFactor = params_.searchExploration;
    if (explorationFactor < topK)
        explorationFactor = topK;

    const float queryMagnitude = VecMath::magnitudeRaw(query.data(), dimension_);

    std::size_t entryPoint = entryPoint_;

    for (int currentLayer = maxLevel_; currentLayer > 0; --currentLayer)
    {
        auto nearest = searchLayer(query, queryMagnitude, entryPoint, 1, currentLayer);
        if (!nearest.empty())
            entryPoint = nearest[0].second;
    }

    auto candidates = searchLayer(query, queryMagnitude, entryPoint, explorationFactor, 0);

    std::vector<HNSWResult> results;
    results.reserve(topK);
    for (std::size_t i = 0; i < candidates.size() && results.size() < topK; ++i)
    {
        std::size_t nodeIdx = candidates[i].second;
        bool seen = false;
        for (const auto &r : results)
            if (r.id == nodes_[nodeIdx].id) { seen = true; break; }
        if (seen) continue;
        const Node &n = nodes_[nodeIdx];
        results.push_back({candidates[i].first, n.id, n.text});
    }
    return results;
}

bool HNSWIndex::built() const noexcept { return maxLevel_ >= 0; }
std::size_t HNSWIndex::size() const noexcept { return nodes_.size(); }
std::size_t HNSWIndex::dimension() const noexcept { return dimension_; }
int HNSWIndex::levels() const noexcept { return maxLevel_ + 1; }
