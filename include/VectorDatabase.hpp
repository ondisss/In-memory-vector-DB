#pragma once

#include "VectorRecord.hpp"

#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <variant>
#include <vector>

struct SearchResult
{
    float score;
    VectorRecord record;
};

struct LoadStats
{
    std::size_t loaded = 0;
    std::size_t skipped = 0;
    double elapsedMs = 0.0;
};

using ProgressCallback = std::function<void(std::size_t current, std::size_t total)>;

enum class Metric
{
    Cosine,
    Euclidean
};

class VectorDatabase
{
public:
    explicit VectorDatabase(std::size_t dimension);

    bool addRecord(VectorRecord record);

    void reserve(std::size_t n);

    bool deleteById(uint32_t id);

    void clear() noexcept;

    std::variant<LoadStats, std::string> loadFromJSON(const std::string &filePath,
                                                      ProgressCallback onProgress = nullptr);

    std::variant<LoadStats, std::string> loadFromCSV(const std::string &filePath,
                                                     ProgressCallback onProgress = nullptr);

    std::optional<std::string> saveToBinary(const std::string &filePath) const;

    std::variant<LoadStats, std::string> loadFromBinary(const std::string &filePath,
                                                        ProgressCallback onProgress = nullptr);

    std::vector<SearchResult> search(const std::vector<float> &query, std::size_t topK,
                                     Metric metric = Metric::Cosine) const;

    std::vector<SearchResult> searchParallel(const std::vector<float> &query, std::size_t topK,
                                             std::size_t numThreads = 0, Metric metric = Metric::Cosine) const;

    std::size_t size() const noexcept;
    std::size_t dimension() const noexcept;
    std::size_t estimatedMemoryBytes() const noexcept;

    const std::vector<VectorRecord> &records() const noexcept;

private:
    std::vector<SearchResult> searchChunk(const std::vector<float> &query,
                                          std::size_t start,
                                          std::size_t end,
                                          std::size_t topK,
                                          Metric metric) const;

    std::size_t dimension_;
    std::vector<VectorRecord> records_;
};
