// Unit tests for the semiring abstraction.

#include "test_framework.hpp"
#include "hpc/hpc.hpp"

#include <vector>
#include <numeric>
#include <limits>

using namespace hpc;

// ---------------------------------------------------------------------------
// Built-in semiring definitions
// ---------------------------------------------------------------------------

TEST(Semiring, PlusTimesIdentity) {
    double zero = -1.0, one = -1.0;
    PlusTimesDouble sem;
    sem.identity_add(&zero);
    sem.identity_multiply(&one);

    EXPECT_NEAR(zero, 0.0, 1e-15);
    EXPECT_NEAR(one, 1.0, 1e-15);
    EXPECT_STREQ(sem.name(), "");   // traits_.name is unset by default
    EXPECT_EQ(sem.element_size(), sizeof(double));
}

TEST(Semiring, MinPlusIdentity) {
    float zero = -1.0f, one = -1.0f;
    MinPlusFloat sem;
    sem.identity_add(&zero);
    sem.identity_multiply(&one);

    // min-identity is +inf, plus-identity is 0
    EXPECT_TRUE(std::isinf(zero));
    EXPECT_GT(zero, 0.0f);
    EXPECT_NEAR(one, 0.0f, 1e-7f);
}

TEST(Semiring, MaxPlusIdentity) {
    double zero = 1.0, one = 1.0;
    MaxPlusDouble sem;
    sem.identity_add(&zero);
    sem.identity_multiply(&one);

    EXPECT_TRUE(std::isinf(zero));
    EXPECT_LT(zero, 0.0);
    EXPECT_NEAR(one, 0.0, 1e-15);
}

TEST(Semiring, PlusTimesOperations) {
    PlusTimesDouble sem;
    double r = 0.0;
    double a = 3.5, b = 2.25;

    sem.add(&a, &b, &r);
    EXPECT_NEAR(r, 5.75, 1e-12);

    sem.multiply(&a, &b, &r);
    EXPECT_NEAR(r, 7.875, 1e-12);

    // 1 * x == x
    double one = 1.0;
    sem.multiply(&one, &b, &r);
    EXPECT_NEAR(r, b, 1e-15);
}

TEST(Semiring, MinPlusOperations) {
    MinPlusDouble sem;
    double r = 0.0;

    double a = 4.0, b = 7.0;
    sem.add(&a, &b, &r);
    EXPECT_NEAR(r, 4.0, 1e-12);        // min

    double x = 3.0, y = 5.0;
    sem.multiply(&x, &y, &r);
    EXPECT_NEAR(r, 8.0, 1e-12);        // plus
}

TEST(Semiring, MaxPlusOperations) {
    MaxPlusDouble sem;
    double r = 0.0;

    double a = 4.0, b = 7.0;
    sem.add(&a, &b, &r);
    EXPECT_NEAR(r, 7.0, 1e-12);

    double x = 3.0, y = 5.0;
    sem.multiply(&x, &y, &r);
    EXPECT_NEAR(r, 8.0, 1e-12);
}

TEST(Semiring, BooleanOperations) {
    LorLandBool sem;
    bool r = false;

    bool t = true, f = false;
    sem.add(&t, &f, &r);
    EXPECT_TRUE(r);

    sem.multiply(&t, &f, &r);
    EXPECT_FALSE(r);

    bool ident_add = true, ident_mul = false;
    sem.identity_add(&ident_add);
    sem.identity_multiply(&ident_mul);
    EXPECT_FALSE(ident_add);   // false OR x == x
    EXPECT_TRUE(ident_mul);    // true AND x == x
}

TEST(Semiring, Int32Operations) {
    PlusTimesInt32 sem;
    int32_t r = 0;

    int32_t a = 100000, b = 40000;
    sem.add(&a, &b, &r);
    EXPECT_EQ(r, 140000);

    sem.multiply(&a, &b, &r);
    EXPECT_EQ(r, static_cast<int32_t>(4000000000LL));   // wraps, documented
}

TEST(Semiring, Int64Operations) {
    PlusTimesInt64 sem;
    int64_t r = 0;

    int64_t a = 3000000000LL, b = 4000000000LL;
    sem.add(&a, &b, &r);
    EXPECT_EQ(r, 7000000000LL);

    sem.multiply(&a, &b, &r);
    EXPECT_EQ(r, 12000000000000000000LL - (1LL << 63) * 2);  // wrapped
}

TEST(Semiring, EqualsAndCopy) {
    PlusTimesDouble sem;
    double a = 2.5, b = 2.5, c = 3.5, dst = 0.0;

    EXPECT_TRUE(sem.equals(&a, &b));
    EXPECT_FALSE(sem.equals(&a, &c));

    sem.copy(&c, &dst);
    EXPECT_NEAR(dst, 3.5, 1e-15);
    EXPECT_TRUE(sem.equals(&dst, &c));
}

TEST(Semiring, FirstAndSecondOperators) {
    PlusFirstInt32 sem;
    int32_t r = 0;
    int32_t a = 7, b = 99;

    sem.add(&a, &b, &r);
    EXPECT_EQ(r, 106);

    sem.multiply(&a, &b, &r);
    EXPECT_EQ(r, 7);    // first(x, y) == x

    PlusSecondInt32 sem2;
    sem2.multiply(&a, &b, &r);
    EXPECT_EQ(r, 99);   // second(x, y) == y
}

// ---------------------------------------------------------------------------
// Algebraic laws: each semiring must satisfy the SpGEMM preconditions.
// ---------------------------------------------------------------------------

namespace {

template<typename Sem, typename T>
void check_algebraic_laws(Sem sem) {
    T a{}, b{}, c{}, r{};

    // add-identity: 0 + x == x
    sem.identity_add(&a);
    sem.add(&a, &b, &r);
    EXPECT_TRUE(sem.equals(&r, &b));

    // multiply-identity: 1 * x == x
    sem.identity_multiply(&c);
    sem.multiply(&c, &b, &r);
    EXPECT_TRUE(sem.equals(&r, &b));

    // add associativity: (a + b) + c == a + (b + c)
    sem.identity_add(&a);
    sem.identity_multiply(&c);
    T t1{}, t2{}, left{}, right{};
    sem.add(&b, &c, &t1);
    sem.add(&a, &t1, &left);
    sem.add(&c, &a, &t2);
    sem.add(&b, &t2, &right);
    EXPECT_TRUE(sem.equals(&left, &right));

    // add commutativity
    sem.add(&b, &c, &t1);
    sem.add(&c, &b, &t2);
    EXPECT_TRUE(sem.equals(&t1, &t2));
}

} // namespace

TEST(SemiringLaws, PlusTimesDouble) { check_algebraic_laws<PlusTimesDouble>(PlusTimesDouble{}); }
TEST(SemiringLaws, PlusTimesFloat)  { check_algebraic_laws<PlusTimesFloat>(PlusTimesFloat{}); }
TEST(SemiringLaws, PlusTimesInt32)  { check_algebraic_laws<PlusTimesInt32>(PlusTimesInt32{}); }
TEST(SemiringLaws, PlusTimesInt64)  { check_algebraic_laws<PlusTimesInt64>(PlusTimesInt64{}); }
TEST(SemiringLaws, MinPlusDouble)   { check_algebraic_laws<MinPlusDouble>(MinPlusDouble{}); }
TEST(SemiringLaws, MinPlusFloat)    { check_algebraic_laws<MinPlusFloat>(MinPlusFloat{}); }
TEST(SemiringLaws, MaxPlusDouble)   { check_algebraic_laws<MaxPlusDouble>(MaxPlusDouble{}); }
TEST(SemiringLaws, LorLandBool)     { check_algebraic_laws<LorLandBool>(LorLandBool{}); }

TEST(SemiringLaws, MinPlusDistributivity) {
    MinPlusDouble sem;
    // min(x, y + z) == min(x + z, y + z)
    double x = 2.0, y = 5.0, z = 3.0;
    double ypz{}, xpz{}, lhs{}, rhs{};
    sem.add(&y, &z, &ypz);
    sem.multiply(&x, &z, &xpz);   // x + z
    sem.multiply(&x, &y, &lhs);   // min(x, y)
    sem.add(&lhs, &z, &lhs);      // min(x, y) + z
    sem.multiply(&x, &ypz, &rhs); // min(x, y + z)
    EXPECT_NEAR(lhs, rhs, 1e-12);
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

TEST(SemiringRegistry, BuiltinSemiringsAreRegistered) {
    EXPECT_EQ(SemiringRegistry::instance().get_id_by_name("plus_times_double"),
              SemiringType::PLUS_TIMES_DOUBLE);
    EXPECT_EQ(SemiringRegistry::instance().get_id_by_name("min_plus_int32"),
              SemiringType::MIN_PLUS_INT32);
    EXPECT_EQ(SemiringRegistry::instance().get_id_by_name("max_plus_float"),
              SemiringType::MAX_PLUS_FLOAT);
    EXPECT_EQ(SemiringRegistry::instance().get_id_by_name("lor_land_bool"),
              SemiringType::LOR_LAND_BOOL);
}

TEST(SemiringRegistry, LookupUnknownNameFails) {
    EXPECT_EQ(SemiringRegistry::instance().get_id_by_name("no_such_semiring"),
              SEMIRING_ID_INVALID);
}

TEST(SemiringRegistry, DescriptorIsUsable) {
    const auto* d = SemiringRegistry::instance().get_descriptor(
        SemiringType::PLUS_TIMES_DOUBLE);
    ASSERT_TRUE(d != nullptr);
    EXPECT_EQ(d->element_size, sizeof(double));

    double a = 2.0, b = 3.0, r = 0.0;
    d->add_op(&a, &b, &r);
    EXPECT_NEAR(r, 5.0, 1e-15);

    d->mul_op(&a, &b, &r);
    EXPECT_NEAR(r, 6.0, 1e-15);

    double zero = 0.0;
    d->add_identity(&zero);
    EXPECT_NEAR(zero, 0.0, 1e-15);
}

TEST(SemiringManager, LookupById) {
    const auto* d = SemiringManager::instance().get_semiring(
        SemiringManager::min_plus_double());
    ASSERT_TRUE(d != nullptr);

    double a = 8.0, b = 3.0, r = 0.0;
    d->add_op(&a, &b, &r);
    EXPECT_NEAR(r, 3.0, 1e-15);
}

TEST(SemiringManager, CustomSemiringRegistration) {
    auto& mgr = SemiringManager::instance();
    const SemiringID id = mgr.register_semiring<PlusTimesDouble>("test_custom_semiring");
    EXPECT_NE(id, SEMIRING_ID_INVALID);
    EXPECT_EQ(mgr.get_semiring_id("test_custom_semiring"), id);

    // Re-registering the same name must be idempotent.
    const SemiringID id2 = mgr.register_semiring<PlusTimesDouble>("test_custom_semiring");
    EXPECT_EQ(id, id2);
}

// ---------------------------------------------------------------------------
// Semiring application: Floyd-Warshall on a dense graph encoded as min-plus.
// ---------------------------------------------------------------------------

TEST(SemiringApplication, ShortestPathViaMinPlusPower) {
    // 4-node line graph: 0-1-2-3 with unit weights, plus a 0->3 shortcut.
    const int n = 4;
    const int INF = 1'000'000;

    std::vector<int32_t> row_ptr{0, 2, 4, 6, 7};
    std::vector<int32_t> col_idx{1, 3, 0, 2, 1, 3, 2};
    std::vector<int32_t> values{1, 10, 1, 1, 1, 1, 1};

    MinPlusInt32 sem;

    // W^1 = W
    // W^2 = W * W  (paths with 2 edges)
    std::vector<int32_t> c_row_ptr(n + 1, 0);
    std::vector<int32_t> c_col_idx(64, 0);
    std::vector<int32_t> c_values(64, 0);

    cpu::merge_path_spgemm<int32_t, int32_t, MinPlusInt32>(
        row_ptr.data(), col_idx.data(), values.data(),
        row_ptr.data(), col_idx.data(), values.data(),
        c_row_ptr.data(), c_col_idx.data(), c_values.data(),
        n, n, n, sem);

    // From node 0: 0->1 = 1, 0->3 = 10 (or 1+1+1 = 3 via 2 nodes)
    const int base = c_row_ptr[0];
    int best = INF;
    for (int p = base; p < c_row_ptr[1]; ++p) {
        if (c_col_idx[p] == 3) best = std::min(best, c_values[p]);
    }
    // W^2 only allows paths of length exactly 2 edges, so 0->3 is still 10.
    EXPECT_EQ(best, 10);

    // W^3 should find 0->2->3 style paths of 3 edges, giving 0->3 = 3.
    std::vector<int32_t> d_row_ptr(n + 1, 0);
    std::vector<int32_t> d_col_idx(256, 0);
    std::vector<int32_t> d_values(256, 0);
    cpu::merge_path_spgemm<int32_t, int32_t, MinPlusInt32>(
        c_row_ptr.data(), c_col_idx.data(), c_values.data(),
        row_ptr.data(), col_idx.data(), values.data(),
        d_row_ptr.data(), d_col_idx.data(), d_values.data(),
        n, n, n, sem);

    best = INF;
    for (int p = d_row_ptr[0]; p < d_row_ptr[1]; ++p) {
        if (d_col_idx[p] == 3) best = std::min(best, d_values[p]);
    }
    EXPECT_EQ(best, 3);
}

TEST(SemiringApplication, ReachabilityViaBooleanSpGEMM) {
    // W is the adjacency matrix of a path 0-1-2. (W)^2 should mark 0->2.
    const int n = 3;
    std::vector<int32_t> row_ptr{0, 1, 2, 3};
    std::vector<int32_t> col_idx{1, 2, 0};
    std::vector<int32_t> values{1, 1, 1};

    LorLandInt32 sem;

    std::vector<int32_t> c_row_ptr(n + 1, 0);
    std::vector<int32_t> c_col_idx(32, 0);
    std::vector<int32_t> c_values(32, 0);

    cpu::gustavson_spgemm<int32_t, int32_t, LorLandInt32>(
        row_ptr.data(), col_idx.data(), values.data(),
        row_ptr.data(), col_idx.data(), values.data(),
        c_row_ptr.data(), c_col_idx.data(), c_values.data(),
        n, n, n, sem);

    // Row 0 must contain column 2 (0 -> 1 -> 2)
    bool found = false;
    for (int p = c_row_ptr[0]; p < c_row_ptr[1]; ++p) {
        if (c_col_idx[p] == 2 && c_values[p] != 0) found = true;
    }
    EXPECT_TRUE(found);
}

HPC_TEST_MAIN()
