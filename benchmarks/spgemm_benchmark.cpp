// spgemm_benchmark - performance sweep across matrix sizes, algorithms and semirings.
//
// Usage:
//   ./spgemm_benchmark [--n=10000] [--density=1e-4] [--reps=10]
//                      [--algo=auto|hashmap|heap|merge_path|galatic]
//                      [--semiring=plus_times_double|min_plus_double|...]
//                      [--threads=N] [--gpus=N] [--json=out.json] [--csv=out.csv]
//
// References: McFarland, Bellavita & Guidi, ICPE 2025 (kernel selection and
// GPU acceleration); Buluç & Gilbert (Sparse SUMMA cost model).

#include "hpc/hpc.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <iomanip>
#include <algorithm>

using namespace hpc;

namespace {

struct Options {
    uint64_t n = 10000;
    double density = 1e-4;
    int reps = 10;
    int warmup = 3;
    int threads = 0;
    int gpus = 0;
    std::string algo = "auto";
    std::string semiring = "plus_times_double";
    std::string json;
    std::string csv;
    bool sweep_sizes = false;
};

double get_flag_double(const char* s, const char* key, double fallback) {
    const size_t kl = std::strlen(key);
    if (std::strncmp(s, key, kl) == 0) return std::atof(s + kl);
    return fallback;
}

const char* get_flag_str(const char* s, const char* key) {
    const size_t kl = std::strlen(key);
    if (std::strncmp(s, key, kl) == 0) return s + kl;
    return nullptr;
}

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        double d;
        if ((d = get_flag_double(a, "--n=", 0)) > 0) o.n = static_cast<uint64_t>(d);
        else if ((d = get_flag_double(a, "--density=", 0)) > 0) o.density = d;
        else if ((d = get_flag_double(a, "--reps=", 0)) > 0) o.reps = static_cast<int>(d);
        else if ((d = get_flag_double(a, "--warmup=", 0)) > 0) o.warmup = static_cast<int>(d);
        else if ((d = get_flag_double(a, "--threads=", -1)) >= 0) o.threads = static_cast<int>(d);
        else if ((d = get_flag_double(a, "--gpus=", -1)) >= 0) o.gpus = static_cast<int>(d);
        else if (std::strcmp(a, "--sweep") == 0) o.sweep_sizes = true;

        if (const char* v = get_flag_str(a, "--algo=")) o.algo = v;
        else if (const char* v = get_flag_str(a, "--semiring=")) o.semiring = v;
        else if (const char* v = get_flag_str(a, "--json=")) o.json = v;
        else if (const char* v = get_flag_str(a, "--csv=")) o.csv = v;
        else if (std::strcmp(a, "--help") == 0 || std::strcmp(a, "-h") == 0) {
            printf("Usage: %s [--n=N] [--density=P] [--reps=R] [--warmup=W]\n"
                   "          [--threads=T] [--gpus=G] [--algo=A] [--semiring=S]\n"
                   "          [--json=FILE] [--csv=FILE] [--sweep]\n", argv[0]);
            std::exit(0);
        }
    }
    return o;
}

SpGEMMAlgorithm parse_algo(const std::string& s) {
    if (s == "hashmap")    return SpGEMMAlgorithm::HASHMAP;
    if (s == "heap")       return SpGEMMAlgorithm::HEAP;
    if (s == "merge_path") return SpGEMMAlgorithm::MERGE_PATH;
    if (s == "galatic")    return SpGEMMAlgorithm::GALATIC;
    return SpGEMMAlgorithm::AUTO;
}

SemiringID parse_semiring(const std::string& s) {
    if (s == "plus_times_double") return SemiringType::PLUS_TIMES_DOUBLE;
    if (s == "plus_times_float")  return SemiringType::PLUS_TIMES_FLOAT;
    if (s == "plus_times_int32")  return SemiringType::PLUS_TIMES_INT32;
    if (s == "min_plus_double")   return SemiringType::MIN_PLUS_DOUBLE;
    if (s == "min_plus_float")    return SemiringType::MIN_PLUS_FLOAT;
    if (s == "max_plus_double")   return SemiringType::MAX_PLUS_DOUBLE;
    if (s == "lor_land_int32")    return SemiringType::LOR_LAND_INT32;
    return SemiringType::PLUS_TIMES_DOUBLE;
}

struct Row {
    uint64_t n;
    uint64_t nnz_a;
    uint64_t nnz_c;
    std::string algo;
    double time_ms;
    double gflops;
    double bandwidth_gb_s;
    double intensity;
    double symbolic_ms;
    double numeric_ms;
};

double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

void print_header() {
    printf("\n%-9s %11s %11s %10s %10s %10s %10s\n",
           "n", "nnz(A)", "nnz(C)", "time[ms]", "GFLOP/s", "GB/s", "intensity");
    printf("-------------------------------------------------------------------------\n");
}

void print_row(const Row& r) {
    printf("%-9llu %11llu %11llu %10.3f %10.2f %10.2f %10.4f\n",
           static_cast<unsigned long long>(r.n),
           static_cast<unsigned long long>(r.nnz_a),
           static_cast<unsigned long long>(r.nnz_c),
           r.time_ms, r.gflops, r.bandwidth_gb_s, r.intensity);
}

} // namespace

int main(int argc, char** argv) {
    MPI_Comm dummy_world;
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    const Options opt = parse(argc, argv);
    printf("HPC Research Project - SpGEMM benchmark\n");
    printf("  n=%llu  density=%.3g  reps=%d  warmup=%d  threads=%d  gpus=%d\n",
           static_cast<unsigned long long>(opt.n), opt.density, opt.reps, opt.warmup,
           opt.threads, opt.gpus);
    printf("  algorithm=%s  semiring=%s\n", opt.algo.c_str(), opt.semiring.c_str());

    print_hardware_info(detect_hardware());

    std::vector<uint64_t> sizes;
    if (opt.sweep_sizes) {
        sizes = {1000, 2000, 5000, 10000, 20000, 50000, 100000};
    } else {
        sizes = {opt.n};
    }

    std::vector<Row> rows;
    print_header();

    for (uint64_t n : sizes) {
        auto A = generate_random_sparse(n, n, opt.density, MatrixFormat::COO);
        auto B = generate_random_sparse(n, n, opt.density, MatrixFormat::COO);

        SpGEMMDescriptor d;
        d.m = d.n = d.k = n;
        d.nnz_a = A->num_nonzeros();
        d.nnz_b = B->num_nonzeros();
        d.density_a = d.density_b = opt.density;
        d.num_gpus = opt.gpus;

        SpGEMMConfig cfg = SpGEMM::recommend_config(d);
        cfg.algorithm = parse_algo(opt.algo);
        cfg.openmp_threads = opt.threads;
        cfg.use_cuda = opt.gpus > 0;

        auto s = create_spgemm(cfg);
        if (s->set_semiring(parse_semiring(opt.semiring)) != 0) {
            fprintf(stderr, "unknown semiring\n");
            MPI_Finalize();
            return 1;
        }
        if (s->set_matrix_a<>(A.get()) != 0 || s->set_matrix_b<>(B.get()) != 0) {
            fprintf(stderr, "failed to set matrices\n");
            MPI_Finalize();
            return 1;
        }

        // Warmup
        for (int i = 0; i < opt.warmup; ++i) s->compute();

        BenchmarkConfig bc;
        bc.warmup_iterations = 0;
        bc.measurement_iterations = opt.reps;
        bc.min_time_ms = 0;
        bc.verbose = false;

        std::vector<double> times;
        run_benchmark([&]() { s->compute(); }, bc, times);

        const auto& st = s->stats();

        Row r;
        r.n = n;
        r.nnz_a = st.input_nnz_a;
        r.nnz_c = st.output_nnz;
        r.algo = opt.algo;
        r.time_ms = median(times);
        r.gflops = st.gflops;
        r.bandwidth_gb_s = st.memory_bandwidth_gb_s;
        r.intensity = st.arithmetic_intensity;
        r.symbolic_ms = st.symbolic_time_ms;
        r.numeric_ms = st.numeric_time_ms;
        rows.push_back(r);
        print_row(r);
    }

    printf("-------------------------------------------------------------------------\n");
    for (const auto& r : rows) {
        printf("n=%-8llu symbolic=%.3f ms  numeric=%.3f ms\n",
               static_cast<unsigned long long>(r.n), r.symbolic_ms, r.numeric_ms);
    }

    if (!opt.csv.empty()) {
        std::ofstream f(opt.csv);
        f << "n,nnz_a,nnz_c,algorithm,time_ms,gflops,bandwidth_gb_s,"
             "arithmetic_intensity,symbolic_ms,numeric_ms\n";
        for (const auto& r : rows) {
            f << r.n << ',' << r.nnz_a << ',' << r.nnz_c << ',' << r.algo << ','
              << r.time_ms << ',' << r.gflops << ',' << r.bandwidth_gb_s << ','
              << r.intensity << ',' << r.symbolic_ms << ',' << r.numeric_ms << '\n';
        }
        printf("\nwrote %s\n", opt.csv.c_str());
    }

    if (!opt.json.empty()) {
        std::ofstream f(opt.json);
        f << std::setprecision(10);
        f << "{\n  \"benchmark\": \"spgemm\",\n  \"density\": " << opt.density
          << ",\n  \"results\": [\n";
        for (size_t i = 0; i < rows.size(); ++i) {
            const auto& r = rows[i];
            f << "    {\"n\": " << r.n << ", \"nnz_a\": " << r.nnz_a
              << ", \"nnz_c\": " << r.nnz_c << ", \"time_ms\": " << r.time_ms
              << ", \"gflops\": " << r.gflops << ", \"bandwidth_gb_s\": "
              << r.bandwidth_gb_s << "}" << (i + 1 < rows.size() ? "," : "") << "\n";
        }
        f << "  ]\n}\n";
        printf("wrote %s\n", opt.json.c_str());
    }

    MPI_Finalize();
    return 0;
}
