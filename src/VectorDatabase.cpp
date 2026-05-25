#include "VectorDatabase.hpp"
#include "VecMath.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <future>
#include <sstream>
#include <stdexcept>
#include <thread>

#include <nlohmann/json.hpp>

static constexpr char        BINARY_MAGIC[4]    = {'V', 'D', 'B', 'I'};
static constexpr uint32_t    BINARY_VERSION     = 1;
static constexpr std::size_t PROGRESS_INTERVAL  = 5000;

static double msSince(std::chrono::high_resolution_clock::time_point t0)
{
    return std::chrono::duration<double, std::milli>(
               std::chrono::high_resolution_clock::now() - t0).count();
}

static bool betterFirst(const SearchResult &a, const SearchResult &b, Metric metric)
{
    return (metric == Metric::Cosine) ? a.score > b.score : a.score < b.score;
}

VectorDatabase::VectorDatabase(std::size_t dimension) : dimension_(dimension)
{
    if (dimension == 0)
        throw std::invalid_argument("VectorDatabase: dimenze musí být větší než 0");
}

bool VectorDatabase::addRecord(VectorRecord record)
{
    if (record.embedding.size() != dimension_)
        return false;
    records_.push_back(std::move(record));
    return true;
}

void VectorDatabase::reserve(std::size_t n)
{
    records_.reserve(records_.size() + n);
}

bool VectorDatabase::deleteById(uint32_t id)
{
    auto it = std::remove_if(records_.begin(), records_.end(),
                             [id](const VectorRecord &r)
                             { return r.id == id; });
    if (it == records_.end())
        return false;
    records_.erase(it, records_.end());
    return true;
}

void VectorDatabase::clear() noexcept { records_.clear(); }

std::variant<LoadStats, std::string>
VectorDatabase::loadFromJSON(const std::string &filePath, ProgressCallback onProgress)
{
    std::ifstream file(filePath);
    if (!file.is_open())
        return std::string("Nelze otevřít soubor: " + filePath);

    nlohmann::json json;
    try
    {
        file >> json;
    }
    catch (const nlohmann::json::parse_error &e)
    {
        return std::string("Chyba parsování JSON: ") + e.what();
    }
    if (!json.is_array())
        return std::string("JSON musí být pole objektů (array na nejvyšší úrovni)");

    LoadStats stats;
    std::size_t total = json.size();
    records_.reserve(records_.size() + total);
    auto t0 = std::chrono::high_resolution_clock::now();

    for (std::size_t i = 0; i < total; ++i)
    {
        try
        {
            VectorRecord rec;
            rec.id = json[i].at("id").get<uint32_t>();
            rec.text = json[i].at("text").get<std::string>();
            rec.embedding = json[i].at("embedding").get<std::vector<float>>();
            addRecord(std::move(rec)) ? ++stats.loaded : ++stats.skipped;
        }
        catch (const nlohmann::json::exception &)
        {
            ++stats.skipped;
        }
        if (onProgress && (i % PROGRESS_INTERVAL == 0 || i + 1 == total))
            onProgress(i + 1, total);
    }

    stats.elapsedMs = msSince(t0);
    return stats;
}

std::variant<LoadStats, std::string>
VectorDatabase::loadFromCSV(const std::string &filePath, ProgressCallback onProgress)
{
    std::size_t totalLines = 0;
    {
        std::ifstream ctr(filePath);
        if (!ctr.is_open())
            return std::string("Nelze otevřít soubor: " + filePath);
        std::string ln;
        while (std::getline(ctr, ln))
            if (!ln.empty() && ln[0] != '#')
                ++totalLines;
    }

    std::ifstream file(filePath);
    if (!file.is_open())
        return std::string("Nelze otevřít soubor: " + filePath);

    LoadStats stats;
    records_.reserve(records_.size() + totalLines);
    auto t0 = std::chrono::high_resolution_clock::now();
    std::string line;
    std::size_t lineNum = 0;

    while (std::getline(file, line))
    {
        if (line.empty() || line[0] == '#')
            continue;
        ++lineNum;
        std::istringstream ss(line);
        std::string token;
        VectorRecord rec;

        if (!std::getline(ss, token, ','))
        {
            ++stats.skipped;
            continue;
        }
        try
        {
            rec.id = static_cast<uint32_t>(std::stoul(token));
        }
        catch (const std::exception &)
        {
            ++stats.skipped;
            continue;
        }

        if (!std::getline(ss, token, ','))
        {
            ++stats.skipped;
            continue;
        }
        rec.text = token;

        rec.embedding.reserve(dimension_);
        while (std::getline(ss, token, ','))
        {
            try
            {
                rec.embedding.push_back(std::stof(token));
            }
            catch (const std::exception &)
            {
                break;
            }
        }

        addRecord(std::move(rec)) ? ++stats.loaded : ++stats.skipped;

        if (onProgress && (lineNum % PROGRESS_INTERVAL == 0 || lineNum == totalLines))
            onProgress(lineNum, totalLines);
    }

    stats.elapsedMs = msSince(t0);
    return stats;
}

std::optional<std::string>
VectorDatabase::saveToBinary(const std::string &filePath) const
{
    std::ofstream file(filePath, std::ios::binary);
    if (!file.is_open())
        return std::string("Nelze vytvořit soubor: " + filePath);

    file.write(BINARY_MAGIC, 4);
    file.write(reinterpret_cast<const char *>(&BINARY_VERSION), 4);
    uint32_t dim32 = static_cast<uint32_t>(dimension_);
    uint64_t count = static_cast<uint64_t>(records_.size());
    file.write(reinterpret_cast<const char *>(&dim32), 4);
    file.write(reinterpret_cast<const char *>(&count), 8);

    for (const VectorRecord &rec : records_)
    {
        file.write(reinterpret_cast<const char *>(&rec.id), 4);
        uint32_t tlen = static_cast<uint32_t>(rec.text.size());
        file.write(reinterpret_cast<const char *>(&tlen), 4);
        file.write(rec.text.data(), tlen);
        file.write(reinterpret_cast<const char *>(rec.embedding.data()),
                   static_cast<std::streamsize>(dimension_ * sizeof(float)));
    }
    if (!file)
        return std::string("Chyba zápisu: " + filePath);
    return std::nullopt;
}

std::variant<LoadStats, std::string>
VectorDatabase::loadFromBinary(const std::string &filePath, ProgressCallback onProgress)
{
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open())
        return std::string("Nelze otevřít soubor: " + filePath);

    char magic[4];
    file.read(magic, 4);
    if (std::memcmp(magic, BINARY_MAGIC, 4) != 0)
        return std::string("Neplatný formát souboru (špatné magic bytes)");

    uint32_t version, dim32;
    uint64_t count;
    file.read(reinterpret_cast<char *>(&version), 4);
    if (version != BINARY_VERSION)
        return std::string("Nepodporovaná verze: " + std::to_string(version));
    file.read(reinterpret_cast<char *>(&dim32), 4);
    if (dim32 != dimension_)
        return std::string("Dimenze souboru (" + std::to_string(dim32) +
                           ") ≠ databáze (" + std::to_string(dimension_) + ")");
    file.read(reinterpret_cast<char *>(&count), 8);

    LoadStats stats;
    records_.reserve(records_.size() + static_cast<std::size_t>(count));
    auto t0 = std::chrono::high_resolution_clock::now();

    for (uint64_t i = 0; i < count; ++i)
    {
        VectorRecord rec;
        file.read(reinterpret_cast<char *>(&rec.id), 4);
        uint32_t tlen;
        file.read(reinterpret_cast<char *>(&tlen), 4);
        rec.text.resize(tlen);
        file.read(rec.text.data(), tlen);
        rec.embedding.resize(dimension_);
        file.read(reinterpret_cast<char *>(rec.embedding.data()),
                  static_cast<std::streamsize>(dimension_ * sizeof(float)));
        if (!file)
            return std::string("Neočekávaný konec souboru u záznamu " + std::to_string(i));
        records_.push_back(std::move(rec));
        ++stats.loaded;
        if (onProgress && (i % PROGRESS_INTERVAL == 0 || i + 1 == count))
            onProgress(i + 1, static_cast<std::size_t>(count));
    }

    stats.elapsedMs = msSince(t0);
    return stats;
}

std::vector<SearchResult> VectorDatabase::searchChunk(
    const std::vector<float> &query,
    std::size_t start, std::size_t end,
    std::size_t topK, Metric metric) const
{
    std::vector<SearchResult> results;
    results.reserve(end - start);

    for (std::size_t i = start; i < end; ++i)
    {
        float score = (metric == Metric::Cosine)
                          ? VecMath::cosineSimilarity(query, records_[i].embedding)
                          : VecMath::euclideanDistance(query, records_[i].embedding);
        results.push_back({score, records_[i]});
    }

    auto cmp = [metric](const SearchResult &a, const SearchResult &b)
    { return betterFirst(a, b, metric); };

    std::size_t k = std::min(topK, results.size());
    std::partial_sort(results.begin(),
                      results.begin() + static_cast<std::ptrdiff_t>(k),
                      results.end(), cmp);
    results.erase(results.begin() + static_cast<std::ptrdiff_t>(k), results.end());
    return results;
}

std::vector<SearchResult> VectorDatabase::search(
    const std::vector<float> &query, std::size_t topK, Metric metric) const
{
    if (query.size() != dimension_)
        throw std::invalid_argument("search: dotaz má špatný počet dimenzí");
    if (records_.empty() || topK == 0)
        return {};
    return searchChunk(query, 0, records_.size(), topK, metric);
}

std::vector<SearchResult> VectorDatabase::searchParallel(
    const std::vector<float> &query,
    std::size_t topK, std::size_t numThreads, Metric metric) const
{
    if (query.size() != dimension_)
        throw std::invalid_argument("searchParallel: dotaz má špatný počet dimenzí");
    if (records_.empty() || topK == 0)
        return {};

    if (numThreads == 0)
    {
        numThreads = std::thread::hardware_concurrency();
        if (numThreads == 0)
            numThreads = 4;
    }
    numThreads = std::min(numThreads, records_.size());
    if (numThreads == 1)
        return search(query, topK, metric);

    std::size_t chunkSize = records_.size() / numThreads;

    std::vector<std::future<std::vector<SearchResult>>> futures;
    futures.reserve(numThreads);
    for (std::size_t t = 0; t < numThreads; ++t)
    {
        std::size_t start = t * chunkSize;
        std::size_t end = (t + 1 == numThreads) ? records_.size() : start + chunkSize;
        futures.push_back(
            std::async(std::launch::async,
                       &VectorDatabase::searchChunk, this,
                       std::cref(query), start, end, topK, metric));
    }

    std::vector<SearchResult> merged;
    merged.reserve(numThreads * topK);
    for (auto &f : futures)
    {
        auto partial = f.get();
        merged.insert(merged.end(), partial.begin(), partial.end());
    }

    auto cmp = [metric](const SearchResult &a, const SearchResult &b)
    { return betterFirst(a, b, metric); };
    std::size_t k = std::min(topK, merged.size());
    std::partial_sort(merged.begin(),
                      merged.begin() + static_cast<std::ptrdiff_t>(k),
                      merged.end(), cmp);
    merged.erase(merged.begin() + static_cast<std::ptrdiff_t>(k), merged.end());
    return merged;
}

std::size_t VectorDatabase::size() const noexcept { return records_.size(); }
std::size_t VectorDatabase::dimension() const noexcept { return dimension_; }

const std::vector<VectorRecord> &VectorDatabase::records() const noexcept { return records_; }

std::size_t VectorDatabase::estimatedMemoryBytes() const noexcept
{
    std::size_t total = 0;
    for (const auto &rec : records_)
    {
        total += sizeof(VectorRecord);
        total += rec.embedding.size() * sizeof(float);
        total += rec.text.capacity();
    }
    return total;
}
