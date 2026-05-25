#include "commands.hpp"
#include "VectorRecord.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>

static std::string wrapText(const std::string &raw, int width, int indent)
{
    std::string flat;
    flat.reserve(raw.size());
    bool prevSpace = false;
    for (char c : raw)
    {
        if (c == '\n' || c == '\r' || c == '\t')
            c = ' ';
        if (c == ' ')
        {
            if (!prevSpace)
            {
                flat += c;
                prevSpace = true;
            }
        }
        else
        {
            flat += c;
            prevSpace = false;
        }
    }

    std::string pad(static_cast<std::size_t>(indent), ' ');
    std::string result = pad;
    int lineLen = indent;
    bool firstWord = true;
    std::istringstream iss(flat);
    std::string word;
    while (iss >> word)
    {
        int needed = (firstWord ? 0 : 1) + static_cast<int>(word.size());
        if (!firstWord && lineLen + needed > width)
        {
            result += '\n' + pad;
            lineLen = indent;
            firstWord = true;
        }
        if (!firstWord)
        {
            result += ' ';
            ++lineLen;
        }
        result += word;
        lineLen += static_cast<int>(word.size());
        firstWord = false;
    }
    return result;
}

std::string formatBytes(std::size_t bytes)
{
    if (bytes < 1024)
        return std::to_string(bytes) + " B";
    if (bytes < 1024 * 1024)
        return std::to_string(bytes / 1024) + " KB";
    return std::to_string(bytes / (1024 * 1024)) + " MB";
}

void progressBar(std::size_t current, std::size_t total)
{
    constexpr int WIDTH = 40;
    float frac = (total > 0) ? static_cast<float>(current) / static_cast<float>(total) : 0.0f;
    int filled = static_cast<int>(frac * WIDTH);
    std::cout << "\r  [";
    for (int i = 0; i < WIDTH; ++i)
        std::cout << (i < filled ? '#' : '.');
    std::cout << "] " << static_cast<int>(frac * 100.0f) << "%"
              << "  " << current << "/" << total << "   ";
    if (current >= total)
        std::cout << "\n";
    std::cout.flush();
}

double elapsedMs(std::chrono::high_resolution_clock::time_point t0)
{
    return std::chrono::duration<double, std::milli>(
               std::chrono::high_resolution_clock::now() - t0)
        .count();
}

static void printOneResult(std::size_t rank, float score, uint32_t id, const std::string &text)
{
    std::string source, body;
    auto sep = text.find(" | ");
    if (sep != std::string::npos) { source = text.substr(0, sep); body = text.substr(sep + 3); }
    else                            body = text;

    std::cout << "\n " << Color::bold() << rank << "." << Color::reset()
              << "  " << Color::green() << std::fixed << std::setprecision(4) << score << Color::reset()
              << "  " << Color::gray() << source << "  id=" << id << Color::reset() << "\n"
              << wrapText(body, 76, 5) << "\n";
}

void printResults(const std::vector<SearchResult> &results,
                  double ms, std::size_t total, Metric metric)
{
    const char *metricStr = (metric == Metric::Cosine) ? "cosine" : "euclidean";
    std::cout << Color::cyan() << Color::bold()
              << "── Top " << results.size()
              << "  │  " << std::fixed << std::setprecision(1) << ms << " ms"
              << "  │  " << total << " záznamů"
              << "  │  " << metricStr
              << Color::reset() << "\n";
    for (std::size_t i = 0; i < results.size(); ++i)
        printOneResult(i + 1, results[i].score, results[i].record.id, results[i].record.text);
    std::cout << "\n";
}

void printHNSWResults(const std::vector<HNSWResult> &results, double ms)
{
    std::cout << Color::cyan() << Color::bold()
              << "── HNSW Top " << results.size()
              << "  │  " << std::fixed << std::setprecision(1) << ms << " ms"
              << Color::reset() << "\n";
    for (std::size_t i = 0; i < results.size(); ++i)
        printOneResult(i + 1, results[i].score, results[i].id, results[i].text);
    std::cout << "\n";
}

std::vector<float> parseFloats(std::istringstream &iss)
{
    std::vector<float> v;
    float x;
    while (iss >> x)
        v.push_back(x);
    return v;
}

bool tryParseMetric(std::istringstream &iss, Metric &out)
{
    auto pos = iss.tellg();
    std::string tok;
    if (!(iss >> tok))
        return false;
    if (tok == "cosine")
    {
        out = Metric::Cosine;
        return true;
    }
    if (tok == "euclidean")
    {
        out = Metric::Euclidean;
        return true;
    }
    iss.clear();
    iss.seekg(pos);
    return false;
}
