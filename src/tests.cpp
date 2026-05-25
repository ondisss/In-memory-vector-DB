#include "VecMath.hpp"
#include "VectorDatabase.hpp"
#include "VectorRecord.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>

// ---------------------------------------------------------------------------
// Makra – aktivní i v Release módu (nezávisí na NDEBUG)
// ---------------------------------------------------------------------------

#define RUN(name) std::cout << "  " << (name) << " ... "
#define OK()      std::cout << "OK\n"

// CHECK vždy ukončí program s chybou, pokud podmínka nesplněna.
#define CHECK(cond) \
    do { if (!(cond)) { \
        std::cerr << "\nCHECK selhal: " #cond "\n  " << __FILE__ << ":" << __LINE__ << "\n"; \
        std::exit(1); \
    }} while(0)

static bool nearlyEqual(float a, float b, float eps = 1e-5f) {
    return std::fabs(a - b) < eps;
}

// ---------------------------------------------------------------------------
// VecMath testy
// ---------------------------------------------------------------------------

static void testDotProduct() {
    RUN("dotProduct základní");
    CHECK(nearlyEqual(VecMath::dotProduct({1,2,3}, {4,5,6}), 32.0f));
    OK();

    RUN("dotProduct nulový vektor");
    CHECK(nearlyEqual(VecMath::dotProduct({0,0,0}, {1,2,3}), 0.0f));
    OK();
}

static void testMagnitude() {
    RUN("magnitude 3-4-5");
    CHECK(nearlyEqual(VecMath::magnitude({3,4}), 5.0f));
    OK();

    RUN("magnitude jednotkový");
    CHECK(nearlyEqual(VecMath::magnitude({1,0,0}), 1.0f));
    OK();
}

static void testCosineSimilarity() {
    RUN("cosine totožné vektory → 1");
    CHECK(nearlyEqual(VecMath::cosineSimilarity({1,0,0}, {1,0,0}), 1.0f));
    OK();

    RUN("cosine kolmé vektory → 0");
    CHECK(nearlyEqual(VecMath::cosineSimilarity({1,0,0}, {0,1,0}), 0.0f));
    OK();

    RUN("cosine opačné vektory → -1");
    CHECK(nearlyEqual(VecMath::cosineSimilarity({1,0,0}, {-1,0,0}), -1.0f));
    OK();

    RUN("cosine nulový vektor → 0 (ochrana dělení nulou)");
    CHECK(nearlyEqual(VecMath::cosineSimilarity({0,0,0}, {1,2,3}), 0.0f));
    OK();
}

static void testEuclideanDistance() {
    RUN("euclidean vzdálenost 3-4-5 trojúhelník");
    // body (0,0) a (3,4) → vzdálenost = 5
    CHECK(nearlyEqual(VecMath::euclideanDistance({0,0}, {3,4}), 5.0f));
    OK();

    RUN("euclidean stejný bod → 0");
    CHECK(nearlyEqual(VecMath::euclideanDistance({1,2,3}, {1,2,3}), 0.0f));
    OK();
}

// ---------------------------------------------------------------------------
// VectorDatabase testy
// ---------------------------------------------------------------------------

static void testAddAndSize() {
    RUN("addRecord a size()");
    VectorDatabase db(3);
    CHECK(db.size() == 0);
    CHECK(db.addRecord({1, "a", {1,0,0}}));
    CHECK(db.addRecord({2, "b", {0,1,0}}));
    CHECK(db.size() == 2);
    OK();

    RUN("addRecord špatná dimenze → false");
    CHECK(!db.addRecord({3, "bad", {1,2}}));
    CHECK(db.size() == 2);
    OK();
}

static void testDeleteById() {
    RUN("deleteById existující");
    VectorDatabase db(2);
    (void)db.addRecord({10, "x", {1,0}});
    (void)db.addRecord({20, "y", {0,1}});
    CHECK(db.deleteById(10));
    CHECK(db.size() == 1);
    OK();

    RUN("deleteById neexistující → false");
    CHECK(!db.deleteById(99));
    OK();
}

static void testClear() {
    RUN("clear");
    VectorDatabase db(2);
    (void)db.addRecord({1, "a", {1,0}});
    (void)db.addRecord({2, "b", {0,1}});
    db.clear();
    CHECK(db.size() == 0);
    OK();
}

static void testSearchCosine() {
    RUN("search cosine – Top-1 je nejpodobnější");
    VectorDatabase db(3);
    (void)db.addRecord({1, "x_axis",    {1,0,0}});
    (void)db.addRecord({2, "y_axis",    {0,1,0}});
    (void)db.addRecord({3, "z_axis",    {0,0,1}});
    (void)db.addRecord({4, "diag",      {1,1,0}});

    auto results = db.search({1,0,0}, 1, Metric::Cosine);
    CHECK(results.size() == 1);
    CHECK(results[0].record.id == 1);
    OK();

    RUN("search cosine – Top-K vrátí K výsledků");
    auto top3 = db.search({1,0,0}, 3, Metric::Cosine);
    CHECK(top3.size() == 3);
    OK();
}

static void testSearchEuclidean() {
    RUN("search euclidean – Top-1 je nejbližší bod");
    VectorDatabase db(2);
    (void)db.addRecord({1, "blizko", {1,0}});
    (void)db.addRecord({2, "daleko", {5,5}});

    auto results = db.search({1.1f, 0.0f}, 1, Metric::Euclidean);
    CHECK(results.size() == 1);
    CHECK(results[0].record.id == 1);
    OK();
}

static void testSearchParallelMatchesSearch() {
    RUN("searchParallel shoduje se se search (cosine)");
    // Vektory mají různé úhly → jednoznačné pořadí bez remíz.
    // id=1: (1,0,0,0) → kosinus dotazu (1,0,0,0) = 1.0  (nejvyšší)
    // id=2: (1,1,0,0) po normalizaci → ~0.707
    // id=3: (0,1,0,0) → 0.0
    VectorDatabase db(4);
    for (uint32_t i = 1; i <= 100; ++i) {
        float fi = static_cast<float>(i);
        // každý záznam: (1, i, 0, 0) → kosinus k (1,0,0,0) = 1/sqrt(1+i²)
        // menší i → vyšší kosinus; id=1 má kosinus 1/sqrt(2) ≈ 0.707; id=100 je nejnižší
        (void)db.addRecord({i, "v", {1.0f, fi, 0.0f, 0.0f}});
    }
    // id=101: vektor (1,0,0,0) → kosinus = 1.0 (jasný vítěz)
    (void)db.addRecord({101, "winner", {1,0,0,0}});

    std::vector<float> q = {1,0,0,0};
    auto seq = db.search(q, 3, Metric::Cosine);
    auto par = db.searchParallel(q, 3, 4, Metric::Cosine);

    CHECK(seq.size() == par.size());
    CHECK(seq[0].record.id == 101);
    CHECK(par[0].record.id == 101);
    OK();
}

static void testConstructorThrowsOnZeroDim() {
    RUN("konstruktor hází při dimension=0");
    try {
        VectorDatabase db(0);
        CHECK(false && "měla se hodit výjimka");
    } catch (const std::invalid_argument&) {}
    OK();
}

// ---------------------------------------------------------------------------
// Spuštění
// ---------------------------------------------------------------------------

int main() {
    std::cout << "=== VectorDB unit testy ===\n\n";

    std::cout << "[ VecMath ]\n";
    testDotProduct();
    testMagnitude();
    testCosineSimilarity();
    testEuclideanDistance();

    std::cout << "\n[ VectorDatabase ]\n";
    testConstructorThrowsOnZeroDim();
    testAddAndSize();
    testDeleteById();
    testClear();
    testSearchCosine();
    testSearchEuclidean();
    testSearchParallelMatchesSearch();

    std::cout << "\nVšechny testy prošly.\n";
    return 0;
}
