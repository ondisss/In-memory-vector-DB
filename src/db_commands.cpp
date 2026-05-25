#include "commands.hpp"
#include "VectorRecord.hpp"

#include <cmath>
#include <iostream>
#include <random>
#include <thread>
#include <variant>

static std::size_t tryReadOptional(std::istringstream &iss, std::size_t maxVal)
{
    auto pos = iss.tellg();
    std::size_t value = 0;
    if (iss >> value && value > 0 && value <= maxVal)
        return value;
    iss.clear();
    iss.seekg(pos);
    return 0;
}

static std::vector<float> parseQuery(std::istringstream &iss, std::size_t dim)
{
    auto v = parseFloats(iss);
    if (v.size() != dim)
    {
        std::cerr << "Chyba: špatná dimenze dotazu (očekáváno " << dim << ").\n";
        return {};
    }
    return v;
}

void doLoad(VectorDatabase &db, const std::string &path)
{
    std::string ext;
    auto dot = path.rfind('.');
    if (dot != std::string::npos)
        ext = path.substr(dot);

    std::cout << "Načítám: " << path << " ...\n";

    auto printStats = [](const std::variant<LoadStats, std::string> &result)
    {
        if (std::holds_alternative<std::string>(result))
            std::cerr << "Chyba: " << std::get<std::string>(result) << "\n";
        else
        {
            const auto &s = std::get<LoadStats>(result);
            std::cout << "Hotovo: načteno " << s.loaded << " záznamů"
                      << (s.skipped > 0 ? ", přeskočeno " + std::to_string(s.skipped) : "")
                      << "  (" << s.elapsedMs << " ms)\n";
        }
    };

    if (ext == ".vdb")
        printStats(db.loadFromBinary(path, progressBar));
    else if (ext == ".csv")
        printStats(db.loadFromCSV(path, progressBar));
    else
        printStats(db.loadFromJSON(path, progressBar));
}

void askModeAfterLoad(EmbedClient &embedder)
{
    if (!isatty(STDIN_FILENO))
        return;

    std::cout << "\n"
              << Color::bold() << "Zvolte režim hledání:" << Color::reset() << "\n"
              << "  " << Color::cyan() << Color::bold() << "[1]" << Color::reset()
              << " Vektorové   – search, psearch, isearch\n"
              << "  " << Color::cyan() << Color::bold() << "[2]" << Color::reset()
              << " Text+vektor – tsearch, tisearch  (spustí Python embed server)\n"
              << Color::cyan() << "› " << Color::reset();

    std::string choice;
    if (!std::getline(std::cin, choice) || choice != "2")
        return;
    if (embedder.running())
    {
        std::cout << "Embedder již běží.\n";
        return;
    }

    std::cout << "Python interpreter [python3]: ";
    std::string pythonBin;
    std::getline(std::cin, pythonBin);
    if (pythonBin.empty())
        pythonBin = "python3";

    std::cout << "Spouštím embedding server ...\n";
    auto err = embedder.start("data/embed_server.py", pythonBin);
    if (!err.empty())
        std::cerr << Color::red() << "Chyba: " << err << Color::reset() << "\n";
    else
        std::cout << Color::green() << "Embedder spuštěn." << Color::reset() << "\n";
}

void cmdInfo(VectorDatabase &db, const std::optional<HNSWIndex> &index, const EmbedClient &embedder)
{
    std::size_t cores = std::thread::hardware_concurrency();
    std::cout << "Záznamy : " << db.size() << "\n"
              << "Dimenze : " << db.dimension() << "\n"
              << "RAM est.: " << formatBytes(db.estimatedMemoryBytes()) << "\n"
              << "Jádra   : " << (cores ? cores : 4) << "\n"
              << "HNSW    : " << (index && index->built() ? "postaven (" + std::to_string(index->size()) + " uzlů)" : "nepostaven – spusť 'build'") << "\n"
              << "Embedder: " << (embedder.running() ? "spuštěn" : "zastaven – spusť 'embed start'") << "\n";
}

void cmdLoad(std::istringstream &iss, VectorDatabase &db,
             std::optional<HNSWIndex> &index, EmbedClient &embedder)
{
    std::string path;
    if (!(iss >> path))
    {
        std::cerr << "Použití: load <soubor>\n";
        return;
    }
    doLoad(db, path);
    index.reset();
    if (db.size() > 0)
        askModeAfterLoad(embedder);
}

void cmdSave(std::istringstream &iss, VectorDatabase &db)
{
    std::string path;
    if (!(iss >> path))
    {
        std::cerr << "Použití: save <soubor.vdb>\n";
        return;
    }
    std::cout << "Ukládám " << db.size() << " záznamů do " << path << " ...\n";
    auto err = db.saveToBinary(path);
    if (err)
        std::cerr << "Chyba: " << *err << "\n";
    else
        std::cout << "Uloženo.\n";
}

void cmdAdd(std::istringstream &iss, VectorDatabase &db,
            std::optional<HNSWIndex> &index, std::size_t dim)
{
    uint32_t id;
    std::string text;
    if (!(iss >> id >> text))
    {
        std::cerr << "Použití: add <id> <text> <v...>\n";
        return;
    }
    auto emb = parseFloats(iss);
    if (!db.addRecord({id, text, std::move(emb)}))
        std::cerr << "Chyba: embedding musí mít " << dim << " hodnot.\n";
    else
    {
        std::cout << "Přidáno. Celkem: " << db.size() << "\n";
        index.reset();
    }
}

void cmdDel(std::istringstream &iss, VectorDatabase &db, std::optional<HNSWIndex> &index)
{
    uint32_t id;
    if (!(iss >> id))
    {
        std::cerr << "Použití: del <id>\n";
        return;
    }
    if (db.deleteById(id))
    {
        std::cout << "Záznam id=" << id << " smazán. Celkem: " << db.size() << "\n";
        index.reset();
    }
    else
        std::cerr << "Záznam id=" << id << " nenalezen.\n";
}

void cmdClear(VectorDatabase &db, std::optional<HNSWIndex> &index)
{
    std::size_t prev = db.size();
    db.clear();
    index.reset();
    std::cout << "Smazáno " << prev << " záznamů.\n";
}

void cmdSearch(std::istringstream &iss, VectorDatabase &db, std::size_t dim)
{
    std::size_t topK = 0;
    if (!(iss >> topK) || topK == 0)
    {
        std::cerr << "Použití: search <k> [cosine|euclidean] <v...>\n";
        return;
    }
    Metric metric = Metric::Cosine;
    tryParseMetric(iss, metric);
    auto query = parseQuery(iss, dim);
    if (query.empty())
        return;
    auto t0 = std::chrono::high_resolution_clock::now();
    printResults(db.search(query, topK, metric), elapsedMs(t0), db.size(), metric);
}

void cmdPsearch(std::istringstream &iss, VectorDatabase &db, std::size_t dim)
{
    std::size_t topK = 0;
    if (!(iss >> topK) || topK == 0)
    {
        std::cerr << "Použití: psearch <k> [T] [cosine|euclidean] <v...>\n";
        return;
    }
    std::size_t numThreads = tryReadOptional(iss, 512);
    Metric metric = Metric::Cosine;
    tryParseMetric(iss, metric);
    auto query = parseQuery(iss, dim);
    if (query.empty())
        return;
    std::size_t used = numThreads ? numThreads
                                  : std::max(std::size_t(1), (std::size_t)std::thread::hardware_concurrency());
    auto t0 = std::chrono::high_resolution_clock::now();
    std::cout << "[" << used << " vláken]  ";
    printResults(db.searchParallel(query, topK, numThreads, metric), elapsedMs(t0), db.size(), metric);
}

void cmdBuild(std::istringstream &iss, VectorDatabase &db,
              std::optional<HNSWIndex> &index, std::size_t dim)
{
    HNSWParams p;
    std::size_t tmp;
    if (iss >> tmp)
    {
        p.maxNeighbors = tmp;
        if (iss >> tmp)
        {
            p.buildExploration = tmp;
            if (iss >> tmp)
                p.searchExploration = tmp;
        }
    }

    if (db.size() == 0)
    {
        std::cerr << "Databáze je prázdná.\n";
        return;
    }

    std::cout << "Stavím HNSW index (maxNeighbors=" << p.maxNeighbors
              << " buildExploration=" << p.buildExploration
              << " searchExploration=" << p.searchExploration
              << ", záznamů: " << db.size() << ") ...\n";

    auto t0 = std::chrono::high_resolution_clock::now();
    index.emplace(dim, p);
    index->build(db.records());
    std::cout << "Index postaven za " << elapsedMs(t0) << " ms"
              << "  (vrstev: " << (index->built() ? std::to_string(index->levels()) : "chyba") << ")\n";
}

void cmdIsearch(std::istringstream &iss, const std::optional<HNSWIndex> &index, std::size_t dim)
{
    if (!index || !index->built())
    {
        std::cerr << "Index není postaven. Spusť příkaz 'build'.\n";
        return;
    }
    std::size_t topK = 0;
    if (!(iss >> topK) || topK == 0)
    {
        std::cerr << "Použití: isearch <k> [ef] <v...>\n";
        return;
    }
    std::size_t ef = tryReadOptional(iss, 100000);
    auto query = parseQuery(iss, dim);
    if (query.empty())
        return;
    auto t0 = std::chrono::high_resolution_clock::now();
    printHNSWResults(index->search(query, topK, ef), elapsedMs(t0));
}

void cmdBench(std::istringstream &iss, VectorDatabase &db,
              const std::optional<HNSWIndex> &index, std::size_t dim)
{
    std::size_t topK = 0;
    if (!(iss >> topK) || topK == 0)
    {
        std::cerr << "Použití: bench <k> [cosine|euclidean] <v...>\n";
        return;
    }
    Metric metric = Metric::Cosine;
    tryParseMetric(iss, metric);
    auto query = parseQuery(iss, dim);
    if (query.empty())
        return;

    auto t0    = std::chrono::high_resolution_clock::now();
    auto brute = db.search(query, topK, metric);
    double msBrute = elapsedMs(t0);

    auto t1       = std::chrono::high_resolution_clock::now();
    auto parallel = db.searchParallel(query, topK, 0, metric);
    double msParallel = elapsedMs(t1);

    std::size_t cores = std::thread::hardware_concurrency();
    std::cout << "--- 1 vlákno  : " << msBrute << " ms\n"
              << "--- " << (cores ? cores : 4) << " vláken : " << msParallel << " ms\n"
              << "--- Zrychlení bruteforce/parallel : " << (msBrute / msParallel) << "x\n";

    if (index && index->built() && metric == Metric::Cosine)
    {
        auto t2   = std::chrono::high_resolution_clock::now();
        auto hnsw = index->search(query, topK);
        double msHnsw = elapsedMs(t2);

        std::size_t hits = 0;
        for (const auto &r : hnsw)
            for (const auto &b : brute)
                if (b.record.id == r.id) { ++hits; break; }

        double recall = brute.empty() ? 0.0
                                      : 100.0 * static_cast<double>(hits) / static_cast<double>(brute.size());
        std::cout << "--- HNSW         : " << msHnsw << " ms  recall@" << topK << "=" << recall << "%\n"
                  << "--- Zrychlení bruteforce/HNSW  : " << (msBrute / msHnsw) << "x\n";
    }
    else if (metric != Metric::Cosine)
        std::cout << "--- HNSW         : (pouze kosinová metrika)\n";
    else
        std::cout << "--- HNSW         : (index není postaven, spusť 'build')\n";

    std::cout << "--- Top-1 ID  : "
              << (!brute.empty()    ? std::to_string(brute[0].record.id)    : "–")
              << " vs "
              << (!parallel.empty() ? std::to_string(parallel[0].record.id) : "–")
              << ((!brute.empty() && !parallel.empty() && brute[0].record.id == parallel[0].record.id)
                      ? "  ✓ shoda" : "  ✗ NESHODA")
              << "\n";
}

void cmdGenerate(std::istringstream &iss, VectorDatabase &db,
                 std::optional<HNSWIndex> &index, std::size_t dim)
{
    std::size_t count = 0;
    if (!(iss >> count) || count == 0)
    {
        std::cerr << "Použití: generate <počet>\n";
        return;
    }

    std::size_t estBytes = count * (sizeof(VectorRecord) + dim * sizeof(float) + 16);
    if (estBytes > 2ULL * 1024 * 1024 * 1024)
    {
        std::cout << "Varování: " << count << " záznamů zabere ~"
                  << formatBytes(estBytes) << " RAM.\nPokračovat? (ano/ne): ";
        std::string confirm;
        std::getline(std::cin, confirm);
        if (confirm != "ano")
            return;
    }

    db.reserve(count);
    std::mt19937 rng(std::random_device{}());
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::size_t base = db.size();
    auto t0 = std::chrono::high_resolution_clock::now();

    for (std::size_t i = 0; i < count; ++i)
    {
        std::vector<float> emb(dim);
        float norm = 0.0f;
        for (float &x : emb)
        {
            x = dist(rng);
            norm += x * x;
        }
        norm = std::sqrt(norm);
        if (norm > 1e-8f)
            for (float &x : emb)
                x /= norm;
        (void)db.addRecord({static_cast<uint32_t>(base + i + 1),
                            "gen_" + std::to_string(base + i + 1), std::move(emb)});
        if (i % 100000 == 0 || i + 1 == count)
            progressBar(i + 1, count);
    }

    std::cout << "Vygenerováno " << count << " záznamů za " << elapsedMs(t0) << " ms"
              << "  (RAM: " << formatBytes(db.estimatedMemoryBytes()) << ")\n";
    index.reset();
}
