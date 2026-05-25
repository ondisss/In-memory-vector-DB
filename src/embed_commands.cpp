#include "commands.hpp"

#include <iostream>
#include <thread>

static std::vector<float> embedText(const std::string &text, EmbedClient &embedder, std::size_t dim)
{
    auto query = embedder.embed(text);
    if (query.empty())
    {
        std::cerr << "Chyba: embedding selhal.\n";
        return {};
    }
    if (query.size() != dim)
    {
        std::cerr << "Chyba: embedding má " << query.size()
                  << " dimenzí, databáze vyžaduje " << dim
                  << ". Spusť s --dim " << query.size() << ".\n";
        return {};
    }
    return query;
}

static bool parseTextSearchArgs(std::istringstream &iss, EmbedClient &embedder,
                                std::size_t dim, const char *usage,
                                std::size_t &topK, std::vector<float> &query)
{
    if (!embedder.running())
    {
        std::cerr << "Embedder není spuštěn. Spusť 'embed start'.\n";
        return false;
    }
    if (!(iss >> topK) || topK == 0)
    {
        std::cerr << "Použití: " << usage << "\n";
        return false;
    }
    std::string text;
    std::getline(iss >> std::ws, text);
    if (text.empty())
    {
        std::cerr << "Použití: " << usage << "\n";
        return false;
    }
    query = embedText(text, embedder, dim);
    return !query.empty();
}

void cmdEmbed(std::istringstream &iss, EmbedClient &embedder)
{
    std::string subcmd;
    if (!(iss >> subcmd))
    {
        std::cerr << "Použití: embed start [skript] | embed stop\n";
        return;
    }

    if (subcmd == "start")
    {
        if (embedder.running())
        {
            std::cout << "Embedder již běží.\n";
            return;
        }
        std::string pythonBin = "python3", script, tok;
        while (iss >> tok)
        {
            if (tok == "--python" || tok == "-p")
                iss >> pythonBin;
            else
                script = tok;
        }
        if (script.empty())
            script = "data/embed_server.py";
        std::cout << "Spouštím embedding server\n"
                  << "  python : " << pythonBin << "\n"
                  << "  skript : " << script << "\n"
                  << "(první spuštění může stáhnout model ~120 MB)\n";
        auto err = embedder.start(script, pythonBin);
        if (!err.empty())
            std::cerr << "Chyba: " << err << "\n";
        else
            std::cout << "Embedder spuštěn. Model: paraphrase-multilingual-MiniLM-L12-v2 (dim=384)\n";
    }
    else if (subcmd == "stop")
    {
        embedder.stop();
        std::cout << "Embedder zastaven.\n";
    }
    else
        std::cerr << "Neznámý podpříkaz: embed " << subcmd << "\n";
}

void cmdTsearch(std::istringstream &iss, VectorDatabase &db,
                EmbedClient &embedder, std::size_t dim)
{
    std::size_t topK = 0;
    std::vector<float> query;
    if (!parseTextSearchArgs(iss, embedder, dim, "tsearch <k> <text...>", topK, query))
        return;
    auto t0 = std::chrono::high_resolution_clock::now();
    printResults(db.search(query, topK, Metric::Cosine), elapsedMs(t0), db.size(), Metric::Cosine);
}

void cmdTpsearch(std::istringstream &iss, VectorDatabase &db,
                 EmbedClient &embedder, std::size_t dim)
{
    std::size_t topK = 0;
    std::vector<float> query;
    if (!parseTextSearchArgs(iss, embedder, dim, "tpsearch <k> <text...>", topK, query))
        return;
    std::size_t cores = std::thread::hardware_concurrency();
    auto t0 = std::chrono::high_resolution_clock::now();
    std::cout << "[" << (cores ? cores : 4) << " vláken]  ";
    printResults(db.searchParallel(query, topK, 0, Metric::Cosine), elapsedMs(t0), db.size(), Metric::Cosine);
}

void cmdTisearch(std::istringstream &iss, const std::optional<HNSWIndex> &index,
                 EmbedClient &embedder, std::size_t dim)
{
    if (!index || !index->built())
    {
        std::cerr << "Index není postaven. Spusť 'build'.\n";
        return;
    }
    std::size_t topK = 0;
    std::vector<float> query;
    if (!parseTextSearchArgs(iss, embedder, dim, "tisearch <k> <text...>", topK, query))
        return;
    auto t0 = std::chrono::high_resolution_clock::now();
    printHNSWResults(index->search(query, topK), elapsedMs(t0));
}
