// Minimal, dependency-free test framework.
//
// The project prefers GoogleTest when it is available, but CI machines and
// HPC login nodes frequently lack it. This header provides just enough of the
// familiar interface (TEST, EXPECT_*, ASSERT_*) that the suite compiles and
// runs anywhere, including inside a container without a package manager.

#pragma once

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <functional>
#include <sstream>
#include <algorithm>
#include <chrono>

namespace hptest {

struct TestCase {
    std::string suite;
    std::string name;
    std::function<void()> fn;
};

inline std::vector<TestCase>& registry() {
    static std::vector<TestCase> tests;
    return tests;
}

struct FailureRecord {
    bool failed = false;
    std::string message;
};

inline FailureRecord& current_failure() {
    static FailureRecord rec;
    return rec;
}

// Counters for the currently running test.
inline int& assertion_count() { static int n = 0; return n; }
inline int& failure_count()   { static int n = 0; return n; }

inline void report_failure(const char* file, int line, const std::string& what) {
    current_failure().failed = true;
    ++failure_count();
    std::ostringstream ss;
    ss << file << ":" << line << ": " << what;
    current_failure().message += ss.str() + "\n";
    fprintf(stderr, "    FAIL  %s\n", ss.str().c_str());
}

struct Registrar {
    Registrar(const char* suite, const char* name, std::function<void()> fn) {
        registry().push_back(TestCase{suite, name, std::move(fn)});
    }
};

inline bool nearly_equal(double a, double b, double tol) {
    if (std::isnan(a) || std::isnan(b)) return false;
    const double diff = std::abs(a - b);
    if (diff <= tol) return true;
    const double scale = std::max(std::abs(a), std::abs(b));
    return diff <= tol * std::max(1.0, scale);
}

inline int run_all(int argc, char** argv) {
    std::string filter;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg.rfind("--filter=", 0) == 0) {
            filter = arg.substr(9);
        }
    }

    printf("\n");
    printf("================================================================\n");
    printf("  HPC Research Project - test suite\n");
    printf("================================================================\n\n");

    int run = 0, passed = 0, failed = 0, skipped = 0;
    std::vector<std::string> failed_names;

    const auto t_start = std::chrono::steady_clock::now();

    for (auto& t : registry()) {
        const std::string full = t.suite + "." + t.name;
        if (!filter.empty() && full.find(filter) == std::string::npos) continue;

        ++run;
        current_failure() = FailureRecord{};
        const int before = assertion_count();

        printf("  [ RUN      ] %s\n", full.c_str());
        fflush(stdout);

        const auto t0 = std::chrono::steady_clock::now();
        try {
            t.fn();
        } catch (const std::exception& e) {
            report_failure(__FILE__, __LINE__,
                           std::string("unexpected exception: ") + e.what());
        } catch (...) {
            report_failure(__FILE__, __LINE__, "unexpected non-standard exception");
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        const int checks = assertion_count() - before;

        if (current_failure().failed) {
            ++failed;
            failed_names.push_back(full);
            printf("  [   FAILED ] %s  (%d checks, %.1f ms)\n", full.c_str(), checks, ms);
        } else {
            ++passed;
            printf("  [       OK ] %s  (%d checks, %.1f ms)\n", full.c_str(), checks, ms);
        }
        fflush(stdout);
    }

    const auto t_end = std::chrono::steady_clock::now();
    const double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    printf("\n================================================================\n");
    printf("  tests run   : %d\n", run);
    printf("  passed      : %d\n", passed);
    printf("  failed      : %d\n", failed);
    printf("  skipped     : %d\n", skipped);
    printf("  assertions  : %d\n", assertion_count());
    printf("  duration    : %.1f ms\n", total_ms);
    printf("================================================================\n");

    if (!failed_names.empty()) {
        printf("\n  Failed tests:\n");
        for (const auto& n : failed_names) printf("    - %s\n", n.c_str());
        printf("\n");
    }

    return failed == 0 ? 0 : 1;
}

// Silence the "no tests registered" case for translation units with no tests.
inline void ensure_registry_used() { (void)registry(); }

} // namespace hptest

#define HPC_TEST_CONCAT_INNER(a, b) a##b
#define HPC_TEST_CONCAT(a, b) HPC_TEST_CONCAT_INNER(a, b)

#define TEST(suite, name)                                                       \
    static void HPC_TEST_CONCAT(hptest_fn_, __LINE__)();                        \
    static ::hptest::Registrar HPC_TEST_CONCAT(hptest_reg_, __LINE__)(          \
        #suite, #name, &HPC_TEST_CONCAT(hptest_fn_, __LINE__));                 \
    static void HPC_TEST_CONCAT(hptest_fn_, __LINE__)()

#define HPC_REPORT(msg) ::hptest::report_failure(__FILE__, __LINE__, (msg))

#define EXPECT_TRUE(cond)                                                       \
    do {                                                                        \
        ++::hptest::assertion_count();                                          \
        if (!(cond)) {                                                          \
            HPC_REPORT(std::string("expected true: ") + #cond);                 \
        }                                                                       \
    } while (0)

#define EXPECT_FALSE(cond)                                                      \
    do {                                                                        \
        ++::hptest::assertion_count();                                          \
        if ((cond)) {                                                           \
            HPC_REPORT(std::string("expected false: ") + #cond);                \
        }                                                                       \
    } while (0)

#define EXPECT_EQ(a, b)                                                         \
    do {                                                                        \
        ++::hptest::assertion_count();                                          \
        const auto hpc_a = (a);                                                 \
        const auto hpc_b = (b);                                                 \
        if (!(hpc_a == hpc_b)) {                                                \
            std::ostringstream hpc_ss;                                          \
            hpc_ss << "expected " << #a << " == " << #b << " ("                  \
                   << hpc_a << " vs " << hpc_b << ")";                          \
            HPC_REPORT(hpc_ss.str());                                           \
        }                                                                       \
    } while (0)

#define EXPECT_NE(a, b)                                                         \
    do {                                                                        \
        ++::hptest::assertion_count();                                          \
        if ((a) == (b)) {                                                       \
            HPC_REPORT(std::string("expected ") + #a + " != " + #b);            \
        }                                                                       \
    } while (0)

#define EXPECT_LT(a, b)                                                         \
    do {                                                                        \
        ++::hptest::assertion_count();                                          \
        if (!((a) < (b))) HPC_REPORT(std::string("expected ") + #a + " < " + #b); \
    } while (0)

#define EXPECT_LE(a, b)                                                         \
    do {                                                                        \
        ++::hptest::assertion_count();                                          \
        if (!((a) <= (b))) HPC_REPORT(std::string("expected ") + #a + " <= " + #b); \
    } while (0)

#define EXPECT_GT(a, b)                                                         \
    do {                                                                        \
        ++::hptest::assertion_count();                                          \
        if (!((a) > (b))) HPC_REPORT(std::string("expected ") + #a + " > " + #b); \
    } while (0)

#define EXPECT_GE(a, b)                                                         \
    do {                                                                        \
        ++::hptest::assertion_count();                                          \
        if (!((a) >= (b))) HPC_REPORT(std::string("expected ") + #a + " >= " + #b); \
    } while (0)

#define EXPECT_NEAR(a, b, tol)                                                 \
    do {                                                                        \
        ++::hptest::assertion_count();                                          \
        const double hpc_a = static_cast<double>(a);                            \
        const double hpc_b = static_cast<double>(b);                            \
        if (!::hptest::nearly_equal(hpc_a, hpc_b, static_cast<double>(tol))) {  \
            std::ostringstream hpc_ss;                                          \
            hpc_ss << "expected " << #a << " ~= " << #b << " ("                  \
                   << hpc_a << " vs " << hpc_b << ", tol " << (tol) << ")";      \
            HPC_REPORT(hpc_ss.str());                                           \
        }                                                                       \
    } while (0)

#define EXPECT_STREQ(a, b)                                                      \
    do {                                                                        \
        ++::hptest::assertion_count();                                          \
        if (std::strcmp((a), (b)) != 0) {                                       \
            HPC_REPORT(std::string("expected \"") + (a) + "\" == \"" + (b) + "\""); \
        }                                                                       \
    } while (0)

#define ASSERT_TRUE(cond)                                                       \
    do {                                                                        \
        ++::hptest::assertion_count();                                          \
        if (!(cond)) {                                                          \
            HPC_REPORT(std::string("assert failed: ") + #cond);                 \
            return;                                                             \
        }                                                                       \
    } while (0)

#define ASSERT_EQ(a, b)                                                         \
    do {                                                                        \
        ++::hptest::assertion_count();                                          \
        const auto hpc_a = (a);                                                 \
        const auto hpc_b = (b);                                                 \
        if (!(hpc_a == hpc_b)) {                                                \
            std::ostringstream hpc_ss;                                          \
            hpc_ss << "assert failed: " << #a << " == " << #b << " ("           \
                   << hpc_a << " vs " << hpc_b << ")";                          \
            HPC_REPORT(hpc_ss.str());                                           \
            return;                                                             \
        }                                                                       \
    } while (0)

#define HPC_TEST_MAIN()                                                         \
    int main(int argc, char** argv) { return ::hptest::run_all(argc, argv); }
