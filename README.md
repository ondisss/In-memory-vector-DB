# VectorDB – in-memory vektorová databáze

Semestrální projekt do předmětu Programování v C++.

## Co to dělá

Program ukládá záznamy ve formě číselných vektorů (embeddingů) a umožňuje v nich hledat pomocí kosinové a euklidovské podobnosti.

Databáze podporuje tři režimy hledání:

- **brute-force** (`search`) – projde všechny záznamy, vždy přesné výsledky, O(n)
- **paralelní brute-force** (`psearch`) – stejné, ale rozdělí práci na více vláken přes `std::async`
- **HNSW** (`isearch`, `tisearch`) – aproximativní graf, řádově rychlejší na velkých datech

## Architektura

```
include/
  VectorRecord.hpp    – struktura záznamu: id, text, embedding
  VecMath.hpp         – dotProduct, magnitude, cosineSimilarity, euclideanDistance (header-only)
  VectorDatabase.hpp  – deklarace VectorDatabase, SearchResult, LoadStats, Metric
  HNSWIndex.hpp       – deklarace HNSWIndex, HNSWParams, HNSWResult
  EmbedClient.hpp     – deklarace EmbedClient (komunikace s Python procesem)
src/
  VectorDatabase.cpp  – load/save, brute-force search, parallel search
  HNSWIndex.cpp       – stavba a prohledávání HNSW grafu
  EmbedClient.cpp     – fork + pipe + execlp komunikace s Python skriptem
  main.cpp            – interaktivní CLI, příkazy, readline, konfigurace
  tests.cpp           – unit testy (assert-based, bez externího frameworku)
data/
  embed_server.py     – Python subprocess: přečte text, vrátí embedding vektor
  kronika.vdb         – binární databáze kroniky Holešova
vecdb.json            – konfigurace (dimenze, výchozí soubor)
CMakeLists.txt
```

### Klíčové třídy

**VectorDatabase** spravuje záznamy jako `std::vector<VectorRecord>`. Záznamy lze načíst z JSON, CSV nebo vlastního binárního `.vdb` formátu (magic `VDBI`, přímý dump dat). Binary formát je vlastní – načítání 28 000 záznamů je díky tomu téměř okamžité.

**HNSWIndex** implementuje Hierarchical Navigable Small World graf. Každý uzel je propojen s nejbližšími sousedy, graf má více vrstev. Hledání probíhá greedy sestupem od nejvyšší vrstvy dolů do vrstvy 0. Prohledávání používá `visitStamp_` mechanismus, aby bylo možné označit navštívené uzly bez alokací.

**EmbedClient** spustí Python skript jako subprocess přes `fork()` + `pipe()` + `execlp()`. C++ mu zapíše text na stdin přes pipu, Python vrátí čísla na stdout.

## Sestavení

Vyžaduje: CMake ≥ 3.16, GCC nebo Clang s C++17, internet při prvním cmake (stáhne `nlohmann/json`).

Pro textové vyhledávání (`tsearch`, `tisearch`) je navíc potřeba Python 3 s balíčkem `sentence-transformers`. Je nutné ho nainstalovat ručně před prvním spuštěním:

```bash
pip install sentence-transformers
```

Samotný jazykový model (~120 MB) se pak stáhne automaticky při prvním volání `embed start` uvnitř programu.

```bash
mkdir build && cd build
cmake ..
make -j
```

## Spuštění a testování

### Unit testy

```bash
./build/vecdb_tests
```

Pokrývají: dotProduct, magnitude, cosine (totožné/kolmé/opačné/nulové vektory), euclidean, addRecord, deleteById, clear, search (cosine i euclidean), shodu searchParallel se search.

### Kompletní walkthrough od spuštění po testování

```bash
# 1. Spustit program (konfigurace se načte z vecdb.json: dim=384, soubor data/kronika.vdb)
./build/vecdb
```

Program se spustí, načte databázi a zeptá se na režim hledání:

```
Zvolte režim hledání:
  [1] Vektorové   – search, psearch, isearch
  [2] Text+vektor – tsearch, tisearch  (spustí Python embed server)
› 2
```

Zadej `2` pro textové hledání. Program se zeptá na Python interpreter:

```
Python interpreter [python3]:
```

Stačí stisknout **Enter** – použije se výchozí `python3`. Při prvním spuštění se stáhne jazykový model (~120 MB), pak program oznámí `Embedder spuštěn.`

Dále v interaktivní smyčce:

```
# zobrazit stav databáze
> info

# brute-force hledání (prochází všechny záznamy)
> tsearch 5 jak se žilo v Holešově

# sestavit HNSW index (aproximativní, rychlý)
> build

# hledání přes HNSW index
> tisearch 5 jak se žilo v Holešově

# hledání klíčového slova
> tisearch 3 kostel

# ukončení
> quit
```

> **Poznámka k `bench`:** příkaz bench bere jako vstup přímo embedding vektor (čísla),
> ne text. Pro textový benchmark stačí porovnat časy `tsearch` a `tisearch` ručně.

### Přehled příkazů

| Příkaz | Popis |
|--------|-------|
| `info` | Stav databáze: počet záznamů, dimenze, RAM, HNSW index, embedder |
| `load <soubor>` | Načte `.json` / `.csv` / `.vdb` |
| `save <soubor.vdb>` | Uloží databázi do binárního souboru |
| `add <id> <text> <v...>` | Přidá jeden záznam ručně |
| `del <id>` | Smaže záznam podle ID |
| `clear` | Smaže všechny záznamy |
| `search <k> [cosine\|euclidean] <v...>` | Brute-force Top-K (1 vlákno) |
| `psearch <k> [T] [cosine\|euclidean] <v...>` | Brute-force Top-K paralelně (T vláken, výchozí: všechna jádra) |
| `build [M] [efC] [efS]` | Postaví HNSW index (výchozí: `16 200 50`) |
| `isearch <k> [ef] <v...>` | HNSW Top-K podle číselného vektoru |
| `embed start [--python /cesta]` | Spustí Python embedding server |
| `embed stop` | Zastaví Python embedding server |
| `tsearch <k> <text...>` | Brute-force textové hledání (1 vlákno) |
| `tpsearch <k> <text...>` | Brute-force textové hledání (paralelní) |
| `tisearch <k> <text...>` | HNSW textové hledání |
| `bench <k> <v...>` | Porovná časy brute-force / parallel / HNSW |
| `generate <n>` | Vygeneruje n náhodných záznamů pro testování |
| `help` | Vypíše nápovědu přímo v programu |
| `quit` | Ukončí program |

### Benchmark výsledky

Měřeno na WSL2, 28 jader, reálná data (kronika, 28 239 záznamů):

| Metoda          | Čas    | Přesnost |
| --------------- | ------ | -------- |
| brute-force 1T  | 271 ms | 100 %    |
| brute-force 28T | ~90 ms | 100 %    |
| HNSW            | 1.9 ms | 100 %    |

HNSW je tedy ~143× rychlejší než brute-force při zachování plné přesnosti na reálných textových datech.

## Návrhová rozhodnutí

**Proč `float` místo `double`:** každý embedding má 384 hodnot – uložené jako `float` (4 bajty) místo `double` (8 bajtů) zabírají vektory polovinu RAM. Pro kosinovou podobnost přesnost `float` plně dostačuje.

**Proč vlastní binární formát:** JSON načítání 28 000 záznamů trvá sekundy (parsování textu). Binary načtení je téměř okamžité – jde o přímý dump floatů z paměti.

**Proč `std::async` pro paralelizaci:** automatická správa životnosti vláken, `std::future` pro výsledky, bez ručního `join()`. Každé vlákno dostane svůj chunk databáze, výsledky se mergují přes `std::partial_sort`.

**Proč HNSW místo jiných algoritmů:** HNSW je relativně pochopitelný (hierarchický graf + greedy search) a dá se implementovat čistě v C++ bez externích závislostí.

**Proč fork+pipe pro Python:** `system()` by nestačil pro obousměrnou komunikaci. Přímá unixová IPC (`fork`, `dup2`, `execlp`, `select`) je nízkoúrovňová, ale přesně vidím co se děje a nemám žádnou extra závislost.

## Co bylo zajímavé / co mi dalo práci

**HNSW bug:** Po implementaci algoritmu jsem zjistil, že v určitých případech se stejný uzel vrátil ve výsledcích dvakrát (duplicitní výsledek se shodným id i skóre). Příčina je v edge case při graph traversal – uzel se stejným skóre projde visitStamp filtrem. Opravil jsem to deduplicí ve výstupní funkci `search()`.

**WSL2:** Stavba HNSW indexu trvá na WSL2 výrazně déle než na nativním Linuxu. Graph traversal dělá hodně náhodných přístupů do paměti a WSL2 má v tomto ohledu vyšší latenci. Na reálném Linuxu by build byl výrazně rychlejší.

**Fork + pipe:** Implementace obousměrné komunikace s Pythonem na čistém Unixovém API (bez knihovny) byla zajímavá. Musel jsem správně zavřít nevyužité konce pipu, ošetřit `SIGPIPE` a přidat `select()` timeout aby program nezávisl při chybě v Pythonu.

## Co bych udělal lépe

- Přidat perzistenci HNSW grafu (dnes se po každém spuštění musí znovu buildovat, což trvá minuty)
- HNSW `build` je single-threaded, šlo by paralelizovat po vrstvách
