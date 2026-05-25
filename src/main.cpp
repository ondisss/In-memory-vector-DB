#include "commands.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <unistd.h>

#ifdef HAVE_READLINE
#include <readline/readline.h>
#include <readline/history.h>
#include <cstring>
#endif

#include <nlohmann/json.hpp>

struct Config
{
    std::size_t dimension = 0;
    std::string defaultLoad;
};

static Config loadConfig(const std::string &path = "vecdb.json")
{
    Config cfg;
    std::ifstream f(path);
    if (!f.is_open())
        return cfg;
    try
    {
        nlohmann::json j;
        f >> j;
        if (j.contains("dimension"))
            cfg.dimension = j["dimension"].get<std::size_t>();
        if (j.contains("default_load"))
            cfg.defaultLoad = j["default_load"].get<std::string>();
    }
    catch (const std::exception &e)
    {
        std::cerr << "[config] Chyba při čtení " << path << ": " << e.what() << " – ignoruji.\n";
    }
    return cfg;
}

#ifdef HAVE_READLINE
static const char *const kCmds[] = {
    "load", "save", "add", "del", "clear",
    "search", "psearch", "build", "isearch",
    "embed", "tsearch", "tpsearch", "tisearch",
    "bench", "generate", "info", "help", "quit", nullptr};

static char *cmdGenerator(const char *text, int state)
{
    static int idx, len;
    if (!state)
    {
        idx = 0;
        len = static_cast<int>(std::strlen(text));
    }
    while (kCmds[idx])
    {
        const char *c = kCmds[idx++];
        if (std::strncmp(c, text, static_cast<std::size_t>(len)) == 0)
            return strdup(c);
    }
    return nullptr;
}

static char **vecdbComplete(const char *text, int start, int)
{
    if (start == 0)
    {
        rl_attempted_completion_over = 1;
        return rl_completion_matches(text, cmdGenerator);
    }
    return nullptr;
}
#endif

static void printUsage(const char *prog)
{
    std::cerr << "Použití: " << prog << " [--dim N] [--load soubor]\n"
              << "         Parametry lze také nastavit v souboru vecdb.json.\n";
}

static void printHelp(std::size_t dim)
{
    static const char *cmds[][2] = {
        {"load <soubor>",                             "Načte .json / .csv / .vdb"},
        {"save <soubor.vdb>",                         "Uloží databázi do binárního souboru"},
        {"add <id> <text> <v...>",                    "Přidá jeden záznam"},
        {"del <id>",                                  "Smaže záznam podle ID"},
        {"clear",                                     "Smaže všechny záznamy"},
        {"search <k> [cosine|euclidean] <v...>",      "Hledá Top-K brute-force (1 vlákno)"},
        {"psearch <k> [T] [cosine|euclidean] <v...>", "Hledá Top-K paralelně"},
        {"build [M] [efC] [efS]",                     "Postaví HNSW index (výchozí: 16 200 50)"},
        {"isearch <k> [ef] <v...>",                   "Přibližné Top-K přes HNSW index"},
        {"embed start [--python /cesta] [skript]",    "Spustí Python embedding server"},
        {"embed stop",                                "Zastaví embedding server"},
        {"tsearch <k> <text...>",                     "Textové hledání brute-force"},
        {"tpsearch <k> <text...>",                    "Textové hledání paralelní"},
        {"tisearch <k> <text...>",                    "Textové hledání přes HNSW"},
        {"bench <k> [cosine|euclidean] <v...>",       "Porovná brute / parallel / HNSW"},
        {"generate <n>",                              "Vygeneruje n náhodných záznamů"},
        {"info",                                      "Stav databáze a indexu"},
        {"help",                                      "Tato nápověda"},
        {"quit",                                      "Ukončení"},
    };
    std::size_t cores = std::thread::hardware_concurrency();
    std::cout << "\nPříkazy  (dimenze: " << dim << ", jádra: " << (cores ? cores : 4) << "):\n";
    for (const auto &row : cmds)
        std::cout << "  " << std::left << std::setw(44) << row[0] << row[1] << "\n";
    std::cout << "\n";
}

int main(int argc, char *argv[])
{
    Color::init();
    Config cfg = loadConfig();

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--dim" && i + 1 < argc)
        {
            try
            {
                int v = std::stoi(argv[++i]);
                if (v <= 0)
                    throw std::invalid_argument("");
                cfg.dimension = static_cast<std::size_t>(v);
            }
            catch (const std::exception &)
            {
                std::cerr << "Chyba: '--dim' musí být kladné celé číslo.\n";
                return 1;
            }
        }
        else if (arg == "--load" && i + 1 < argc)
            cfg.defaultLoad = argv[++i];
        else
        {
            printUsage(argv[0]);
            return 1;
        }
    }

    if (cfg.dimension == 0)
    {
        printUsage(argv[0]);
        return 1;
    }

    auto db = std::make_unique<VectorDatabase>(cfg.dimension);
    std::optional<HNSWIndex> index;
    EmbedClient embedder;

    std::cout << Color::cyan() << Color::bold() << "=== VectorDB ===" << Color::reset() << "\n";
    printHelp(cfg.dimension);

    if (!cfg.defaultLoad.empty())
    {
        doLoad(*db, cfg.defaultLoad);
        index.reset();
        if (db->size() > 0)
            askModeAfterLoad(embedder);
    }

#ifdef HAVE_READLINE
    rl_attempted_completion_function = vecdbComplete;
    using_history();
    const char *rlPrompt = Color::on ? "\001\033[36m\002> \001\033[0m\002" : "> ";
#endif

    std::string line;
    for (;;)
    {
#ifdef HAVE_READLINE
        char *raw = readline(rlPrompt);
        if (!raw)
        {
            std::cout << '\n';
            break;
        }
        line = raw;
        if (!line.empty())
            add_history(raw);
        free(raw);
#else
        std::cout << Color::cyan() << "> " << Color::reset();
        if (!std::getline(std::cin, line))
            break;
#endif
        if (line.empty())
            continue;

        std::istringstream iss(line);
        std::string cmd;
        iss >> cmd;

        if (cmd == "quit" || cmd == "q")
            break;
        else if (cmd == "help")
            printHelp(cfg.dimension);
        else if (cmd == "info")
            cmdInfo(*db, index, embedder);
        else if (cmd == "load")
            cmdLoad(iss, *db, index, embedder);
        else if (cmd == "save")
            cmdSave(iss, *db);
        else if (cmd == "add")
            cmdAdd(iss, *db, index, cfg.dimension);
        else if (cmd == "del")
            cmdDel(iss, *db, index);
        else if (cmd == "clear")
            cmdClear(*db, index);
        else if (cmd == "search")
            cmdSearch(iss, *db, cfg.dimension);
        else if (cmd == "psearch")
            cmdPsearch(iss, *db, cfg.dimension);
        else if (cmd == "build")
            cmdBuild(iss, *db, index, cfg.dimension);
        else if (cmd == "isearch")
            cmdIsearch(iss, index, cfg.dimension);
        else if (cmd == "bench")
            cmdBench(iss, *db, index, cfg.dimension);
        else if (cmd == "generate")
            cmdGenerate(iss, *db, index, cfg.dimension);
        else if (cmd == "embed")
            cmdEmbed(iss, embedder);
        else if (cmd == "tsearch")
            cmdTsearch(iss, *db, embedder, cfg.dimension);
        else if (cmd == "tpsearch")
            cmdTpsearch(iss, *db, embedder, cfg.dimension);
        else if (cmd == "tisearch")
            cmdTisearch(iss, index, embedder, cfg.dimension);
        else
            std::cerr << "Neznámý příkaz: '" << cmd << "'. Napište 'help'.\n";
    }

    std::cout << "Ukončuji.\n";
    return 0;
}
