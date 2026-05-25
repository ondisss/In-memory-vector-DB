#pragma once

#include "EmbedClient.hpp"
#include "HNSWIndex.hpp"
#include "VectorDatabase.hpp"

#include <chrono>
#include <optional>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

namespace Color
{
    inline bool on = false;

    inline void init() { on = isatty(STDOUT_FILENO) != 0; }
    inline const char *reset() { return on ? "\033[0m" : ""; }
    inline const char *bold() { return on ? "\033[1m" : ""; }
    inline const char *green() { return on ? "\033[32m" : ""; }
    inline const char *cyan() { return on ? "\033[36m" : ""; }
    inline const char *gray() { return on ? "\033[90m" : ""; }
    inline const char *red() { return on ? "\033[31m" : ""; }
}

std::string formatBytes(std::size_t bytes);
void progressBar(std::size_t current, std::size_t total);
double elapsedMs(std::chrono::high_resolution_clock::time_point t0);

void printResults(const std::vector<SearchResult> &results,
                  double ms, std::size_t total, Metric metric);
void printHNSWResults(const std::vector<HNSWResult> &results, double ms);

std::vector<float> parseFloats(std::istringstream &iss);
bool tryParseMetric(std::istringstream &iss, Metric &out);

void doLoad(VectorDatabase &db, const std::string &path);
void askModeAfterLoad(EmbedClient &embedder);

void cmdInfo(VectorDatabase &db, const std::optional<HNSWIndex> &index,
             const EmbedClient &embedder);

void cmdLoad(std::istringstream &iss, VectorDatabase &db,
             std::optional<HNSWIndex> &index, EmbedClient &embedder);

void cmdSave(std::istringstream &iss, VectorDatabase &db);

void cmdAdd(std::istringstream &iss, VectorDatabase &db,
            std::optional<HNSWIndex> &index, std::size_t dim);

void cmdDel(std::istringstream &iss, VectorDatabase &db,
            std::optional<HNSWIndex> &index);

void cmdClear(VectorDatabase &db, std::optional<HNSWIndex> &index);

void cmdSearch(std::istringstream &iss, VectorDatabase &db, std::size_t dim);

void cmdPsearch(std::istringstream &iss, VectorDatabase &db, std::size_t dim);

void cmdBuild(std::istringstream &iss, VectorDatabase &db,
              std::optional<HNSWIndex> &index, std::size_t dim);

void cmdIsearch(std::istringstream &iss, const std::optional<HNSWIndex> &index,
                std::size_t dim);

void cmdBench(std::istringstream &iss, VectorDatabase &db,
              const std::optional<HNSWIndex> &index, std::size_t dim);

void cmdGenerate(std::istringstream &iss, VectorDatabase &db,
                 std::optional<HNSWIndex> &index, std::size_t dim);

void cmdEmbed(std::istringstream &iss, EmbedClient &embedder);

void cmdTsearch(std::istringstream &iss, VectorDatabase &db,
                EmbedClient &embedder, std::size_t dim);

void cmdTpsearch(std::istringstream &iss, VectorDatabase &db,
                 EmbedClient &embedder, std::size_t dim);

void cmdTisearch(std::istringstream &iss, const std::optional<HNSWIndex> &index,
                 EmbedClient &embedder, std::size_t dim);
