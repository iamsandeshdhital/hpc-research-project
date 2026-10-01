#include "hpc/matrix.hpp"
#include <algorithm>
#include <fstream>
#include <sstream>
#include <random>
#include <unordered_map>
#include <tuple>

HPC_NAMESPACE_BEGIN

// Matrix implementation
template<typename T, MatrixFormat Format>
void Matrix<T, Format>::resize(uint64_t rows, uint64_t cols, uint64_t nnz) {
    rows_ = rows;
    cols_ = cols;
    nnz_ = nnz;

    switch (Format) {
        case MatrixFormat::COO:
            coo_rows_.resize(nnz);
            coo_cols_.resize(nnz);
            coo_values_.resize(nnz);
            break;
        case MatrixFormat::CSR:
            csr_row_ptr_.resize(rows + 1);
            csr_col_idx_.resize(nnz);
            csr_values_.resize(nnz);
            break;
        case MatrixFormat::CSC:
            csc_col_ptr_.resize(cols + 1);
            csc_row_idx_.resize(nnz);
            csc_values_.resize(nnz);
            break;
        case MatrixFormat::DCSC:
            dcsc_col_ptr_.reserve(cols + 1);
            dcsc_row_idx_.reserve(nnz);
            dcsc_values_.reserve(nnz);
            dcsc_super_ptr_.reserve(cols + 1);
            dcsc_super_idx_.reserve(cols + 1);
            break;
        case MatrixFormat::DENSE:
            dense_values_.resize(rows * cols);
            break;
        case MatrixFormat::ELL:
            ell_col_idx_.resize(rows * nnz);  // nnz here is max per row
            ell_values_.resize(rows * nnz);
            ell_max_nnz_per_row_ = nnz;
            break;
        default:
            break;
    }
}

template<typename T, MatrixFormat Format>
void Matrix<T, Format>::clear() {
    rows_ = 0;
    cols_ = 0;
    nnz_ = 0;

    coo_rows_.clear();
    coo_cols_.clear();
    coo_values_.clear();

    csr_row_ptr_.clear();
    csr_col_idx_.clear();
    csr_values_.clear();

    csc_col_ptr_.clear();
    csc_row_idx_.clear();
    csc_values_.clear();

    dcsc_col_ptr_.clear();
    dcsc_row_idx_.clear();
    dcsc_values_.clear();
    dcsc_super_ptr_.clear();
    dcsc_super_idx_.clear();

    dense_values_.clear();

    ell_col_idx_.clear();
    ell_values_.clear();
    ell_max_nnz_per_row_ = 0;

    properties_ = MatrixProperties{};
}

template<typename T, MatrixFormat Format>
size_t Matrix<T, Format>::memory_usage() const {
    size_t bytes = 0;

    bytes += coo_rows_.size() * sizeof(uint64_t);
    bytes += coo_cols_.size() * sizeof(uint64_t);
    bytes += coo_values_.size() * sizeof(T);

    bytes += csr_row_ptr_.size() * sizeof(uint64_t);
    bytes += csr_col_idx_.size() * sizeof(uint64_t);
    bytes += csr_values_.size() * sizeof(T);

    bytes += csc_col_ptr_.size() * sizeof(uint64_t);
    bytes += csc_row_idx_.size() * sizeof(uint64_t);
    bytes += csc_values_.size() * sizeof(T);

    bytes += dcsc_col_ptr_.size() * sizeof(uint64_t);
    bytes += dcsc_row_idx_.size() * sizeof(uint64_t);
    bytes += dcsc_values_.size() * sizeof(T);
    bytes += dcsc_super_ptr_.size() * sizeof(uint64_t);
    bytes += dcsc_super_idx_.size() * sizeof(uint64_t);

    bytes += dense_values_.size() * sizeof(T);

    bytes += ell_col_idx_.size() * sizeof(uint64_t);
    bytes += ell_values_.size() * sizeof(T);

    return bytes;
}

template<typename T, MatrixFormat Format>
void Matrix<T, Format>::transpose() {
    // Convert to COO if not already
    if (Format != MatrixFormat::COO) {
        // This would need a full conversion
        // For now, just swap rows/cols
        std::swap(rows_, cols_);
        properties_.symmetric = false;
        return;
    }

    // Transpose COO
    for (uint64_t i = 0; i < nnz_; ++i) {
        std::swap(coo_rows_[i], coo_cols_[i]);
    }
    std::swap(rows_, cols_);
    properties_.symmetric = false;
}

template<typename T, MatrixFormat Format>
void Matrix<T, Format>::convert_to(MatrixFormat target_format) {
    if (target_format == Format) return;

    // Convert via COO as intermediate
    // This is a simplified implementation
    // Full implementation would convert each format pair directly
    HPC_UNUSED(target_format);
}

// Explicit instantiations for common types and formats
#define INSTANTIATE_MATRIX(T, FMT) \
    template class Matrix<T, MatrixFormat::FMT>;

INSTANTIATE_MATRIX(int32_t, COO)
INSTANTIATE_MATRIX(int32_t, CSR)
INSTANTIATE_MATRIX(int32_t, CSC)
INSTANTIATE_MATRIX(int32_t, DCSC)
INSTANTIATE_MATRIX(int32_t, DENSE)
INSTANTIATE_MATRIX(int32_t, ELL)

INSTANTIATE_MATRIX(int64_t, COO)
INSTANTIATE_MATRIX(int64_t, CSR)
INSTANTIATE_MATRIX(int64_t, CSC)
INSTANTIATE_MATRIX(int64_t, DCSC)
INSTANTIATE_MATRIX(int64_t, DENSE)
INSTANTIATE_MATRIX(int64_t, ELL)

INSTANTIATE_MATRIX(float, COO)
INSTANTIATE_MATRIX(float, CSR)
INSTANTIATE_MATRIX(float, CSC)
INSTANTIATE_MATRIX(float, DCSC)
INSTANTIATE_MATRIX(float, DENSE)
INSTANTIATE_MATRIX(float, ELL)

INSTANTIATE_MATRIX(double, COO)
INSTANTIATE_MATRIX(double, CSR)
INSTANTIATE_MATRIX(double, CSC)
INSTANTIATE_MATRIX(double, DCSC)
INSTANTIATE_MATRIX(double, DENSE)
INSTANTIATE_MATRIX(double, ELL)

INSTANTIATE_MATRIX(bool, COO)
INSTANTIATE_MATRIX(bool, CSR)
INSTANTIATE_MATRIX(bool, CSC)
INSTANTIATE_MATRIX(bool, DCSC)
INSTANTIATE_MATRIX(bool, DENSE)
INSTANTIATE_MATRIX(bool, ELL)

#undef INSTANTIATE_MATRIX

// Matrix Market I/O
bool read_matrix_market(const char* filename, MatrixBase** matrix) {
    std::ifstream file(filename);
    if (!file.is_open()) return false;

    std::string line;
    // Read header
    std::getline(file, line);
    if (line.substr(0, 14) != "%%MatrixMarket") return false;

    // Skip comments
    while (std::getline(file, line)) {
        if (line[0] != '%') break;
    }

    // Parse dimensions
    std::istringstream iss(line);
    uint64_t rows, cols, nnz;
    iss >> rows >> cols >> nnz;

    // Determine format from header
    bool is_symmetric = (line.find("symmetric") != std::string::npos ||
                         line.find("hermitian") != std::string::npos);

    // Create matrix (default to CSR double)
    auto mat = std::make_unique<Matrix<double, MatrixFormat::COO>>(rows, cols);
    mat->properties_.symmetric = is_symmetric;

    // Read entries
    for (uint64_t i = 0; i < nnz && std::getline(file, line); ++i) {
        std::istringstream ess(line);
        uint64_t row, col;
        double val;
        ess >> row >> col >> val;
        // Matrix Market is 1-indexed
        mat->coo_rows_[i] = row - 1;
        mat->coo_cols_[i] = col - 1;
        mat->coo_values_[i] = val;
    }

    mat->nnz_ = nnz;
    *matrix = mat.release();
    return true;
}

bool write_matrix_market(const char* filename, const MatrixBase* matrix) {
    std::ofstream file(filename);
    if (!file.is_open()) return false;

    // Write header
    file << "%%MatrixMarket matrix coordinate real general\n";
    file << "% Generated by HPC Research Project\n";
    file << matrix->num_rows() << " " << matrix->num_cols() << " " << matrix->num_nonzeros() << "\n";

    // Write entries (COO format)
    // This is simplified - would need to convert if not COO
    return true;
}

// Binary I/O
bool read_binary_matrix(const char* filename, MatrixBase** matrix) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) return false;

    uint64_t rows, cols, nnz, format;
    file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    file.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    file.read(reinterpret_cast<char*>(&nnz), sizeof(nnz));
    file.read(reinterpret_cast<char*>(&format), sizeof(format));

    // Create matrix based on format
    // Simplified - only COO double for now
    auto mat = std::make_unique<Matrix<double, MatrixFormat::COO>>(rows, cols);
    mat->coo_rows_.resize(nnz);
    mat->coo_cols_.resize(nnz);
    mat->coo_values_.resize(nnz);

    file.read(reinterpret_cast<char*>(mat->coo_rows_.data()), nnz * sizeof(uint64_t));
    file.read(reinterpret_cast<char*>(mat->coo_cols_.data()), nnz * sizeof(uint64_t));
    file.read(reinterpret_cast<char*>(mat->coo_values_.data()), nnz * sizeof(double));

    mat->nnz_ = nnz;
    *matrix = mat.release();
    return true;
}

bool write_binary_matrix(const char* filename, const MatrixBase* matrix) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) return false;

    uint64_t rows = matrix->num_rows();
    uint64_t cols = matrix->num_cols();
    uint64_t nnz = matrix->num_nonzeros();
    uint64_t format = static_cast<uint64_t>(matrix->format());

    file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    file.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
    file.write(reinterpret_cast<const char*>(&nnz), sizeof(nnz));
    file.write(reinterpret_cast<const char*>(&format), sizeof(format));

    // Write data (simplified)
    return true;
}

// Matrix generation
std::unique_ptr<MatrixBase> generate_erdos_renyi(uint64_t n, double p, MatrixFormat fmt) {
    auto mat = std::make_unique<Matrix<double, MatrixFormat::COO>>(n, n);

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    std::vector<std::tuple<uint64_t, uint64_t, double>> edges;
    edges.reserve(static_cast<size_t>(n * n * p * 1.1));

    for (uint64_t i = 0; i < n; ++i) {
        for (uint64_t j = 0; j < n; ++j) {
            if (dist(rng) < p) {
                edges.emplace_back(i, j, dist(rng));
            }
        }
    }

    mat->nnz_ = edges.size();
    mat->coo_rows_.resize(mat->nnz_);
    mat->coo_cols_.resize(mat->nnz_);
    mat->coo_values_.resize(mat->nnz_);

    for (size_t i = 0; i < edges.size(); ++i) {
        mat->coo_rows_[i] = std::get<0>(edges[i]);
        mat->coo_cols_[i] = std::get<1>(edges[i]);
        mat->coo_values_[i] = std::get<2>(edges[i]);
    }

    // Convert to target format if needed
    if (fmt != MatrixFormat::COO) {
        mat->convert_to(fmt);
    }

    return mat;
}

std::unique_ptr<MatrixBase> generate_rmat(uint64_t scale, uint64_t edge_factor, MatrixFormat fmt) {
    uint64_t n = 1ULL << scale;
    uint64_t num_edges = n * edge_factor;

    auto mat = std::make_unique<Matrix<double, MatrixFormat::COO>>(n, n);

    // RMAT parameters (Graph500 defaults)
    double a = 0.57, b = 0.19, c = 0.19, d = 0.05;

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    std::vector<std::tuple<uint64_t, uint64_t, double>> edges;
    edges.reserve(num_edges);

    for (uint64_t e = 0; e < num_edges; ++e) {
        uint64_t row = 0, col = 0;
        for (uint64_t bit = 0; bit < scale; ++bit) {
            double r = dist(rng);
            uint64_t row_bit = 0, col_bit = 0;

            if (r < a) {
                row_bit = 0; col_bit = 0;
            } else if (r < a + b) {
                row_bit = 0; col_bit = 1;
            } else if (r < a + b + c) {
                row_bit = 1; col_bit = 0;
            } else {
                row_bit = 1; col_bit = 1;
            }

            row = (row << 1) | row_bit;
            col = (col << 1) | col_bit;
        }

        edges.emplace_back(row, col, dist(rng));
    }

    mat->nnz_ = edges.size();
    mat->coo_rows_.resize(mat->nnz_);
    mat->coo_cols_.resize(mat->nnz_);
    mat->coo_values_.resize(mat->nnz_);

    for (size_t i = 0; i < edges.size(); ++i) {
        mat->coo_rows_[i] = std::get<0>(edges[i]);
        mat->coo_cols_[i] = std::get<1>(edges[i]);
        mat->coo_values_[i] = std::get<2>(edges[i]);
    }

    if (fmt != MatrixFormat::COO) {
        mat->convert_to(fmt);
    }

    return mat;
}

std::unique_ptr<MatrixBase> generate_grid_2d(uint64_t n, MatrixFormat fmt) {
    auto mat = std::make_unique<Matrix<double, MatrixFormat::COO>>(n * n, n * n);

    std::vector<std::tuple<uint64_t, uint64_t, double>> edges;
    edges.reserve(n * n * 4);

    for (uint64_t i = 0; i < n; ++i) {
        for (uint64_t j = 0; j < n; ++j) {
            uint64_t u = i * n + j;
            if (i > 0) edges.emplace_back(u, (i - 1) * n + j, 1.0);
            if (i < n - 1) edges.emplace_back(u, (i + 1) * n + j, 1.0);
            if (j > 0) edges.emplace_back(u, i * n + (j - 1), 1.0);
            if (j < n - 1) edges.emplace_back(u, i * n + (j + 1), 1.0);
        }
    }

    mat->nnz_ = edges.size();
    mat->coo_rows_.resize(mat->nnz_);
    mat->coo_cols_.resize(mat->nnz_);
    mat->coo_values_.resize(mat->nnz_);

    for (size_t i = 0; i < edges.size(); ++i) {
        mat->coo_rows_[i] = std::get<0>(edges[i]);
        mat->coo_cols_[i] = std::get<1>(edges[i]);
        mat->coo_values_[i] = std::get<2>(edges[i]);
    }

    if (fmt != MatrixFormat::COO) {
        mat->convert_to(fmt);
    }

    return mat;
}

std::unique_ptr<MatrixBase> generate_grid_3d(uint64_t n, MatrixFormat fmt) {
    auto mat = std::make_unique<Matrix<double, MatrixFormat::COO>>(n * n * n, n * n * n);

    std::vector<std::tuple<uint64_t, uint64_t, double>> edges;
    edges.reserve(n * n * n * 6);

    for (uint64_t i = 0; i < n; ++i) {
        for (uint64_t j = 0; j < n; ++j) {
            for (uint64_t k = 0; k < n; ++k) {
                uint64_t u = (i * n + j) * n + k;
                if (i > 0) edges.emplace_back(u, ((i - 1) * n + j) * n + k, 1.0);
                if (i < n - 1) edges.emplace_back(u, ((i + 1) * n + j) * n + k, 1.0);
                if (j > 0) edges.emplace_back(u, (i * n + (j - 1)) * n + k, 1.0);
                if (j < n - 1) edges.emplace_back(u, (i * n + (j + 1)) * n + k, 1.0);
                if (k > 0) edges.emplace_back(u, (i * n + j) * n + (k - 1), 1.0);
                if (k < n - 1) edges.emplace_back(u, (i * n + j) * n + (k + 1), 1.0);
            }
        }
    }

    mat->nnz_ = edges.size();
    mat->coo_rows_.resize(mat->nnz_);
    mat->coo_cols_.resize(mat->nnz_);
    mat->coo_values_.resize(mat->nnz_);

    for (size_t i = 0; i < edges.size(); ++i) {
        mat->coo_rows_[i] = std::get<0>(edges[i]);
        mat->coo_cols_[i] = std::get<1>(edges[i]);
        mat->coo_values_[i] = std::get<2>(edges[i]);
    }

    if (fmt != MatrixFormat::COO) {
        mat->convert_to(fmt);
    }

    return mat;
}

std::unique_ptr<MatrixBase> generate_random_sparse(uint64_t rows, uint64_t cols, double density, MatrixFormat fmt) {
    uint64_t nnz = static_cast<uint64_t>(rows * cols * density);

    auto mat = std::make_unique<Matrix<double, MatrixFormat::COO>>(rows, cols);

    std::mt19937 rng(42);
    std::uniform_int_distribution<uint64_t> row_dist(0, rows - 1);
    std::uniform_int_distribution<uint64_t> col_dist(0, cols - 1);
    std::uniform_real_distribution<double> val_dist(-1.0, 1.0);

    // Use a set to avoid duplicates (simplified)
    std::unordered_map<uint64_t, double> entries;
    entries.reserve(nnz * 2);

    while (entries.size() < nnz) {
        uint64_t row = row_dist(rng);
        uint64_t col = col_dist(rng);
        uint64_t key = row * cols + col;
        entries[key] = val_dist(rng);
    }

    mat->nnz_ = entries.size();
    mat->coo_rows_.resize(mat->nnz_);
    mat->coo_cols_.resize(mat->nnz_);
    mat->coo_values_.resize(mat->nnz_);

    size_t i = 0;
    for (const auto& [key, val] : entries) {
        mat->coo_rows_[i] = key / cols;
        mat->coo_cols_[i] = key % cols;
        mat->coo_values_[i] = val;
        ++i;
    }

    if (fmt != MatrixFormat::COO) {
        mat->convert_to(fmt);
    }

    return mat;
}

// Matrix operations
void matrix_transpose(const MatrixBase* input, MatrixBase* output) {
    // Implementation would depend on formats
    HPC_UNUSED(input);
    HPC_UNUSED(output);
}

void matrix_convert(const MatrixBase* input, MatrixFormat target_format, MatrixBase** output) {
    *output = input->clone();
    (*output)->convert_to(target_format);
}

void matrix_scale(MatrixBase* matrix, const void* alpha) {
    // Scale all values by alpha
    HPC_UNUSED(matrix);
    HPC_UNUSED(alpha);
}

void matrix_add(const MatrixBase* a, const MatrixBase* b, MatrixBase* c) {
    HPC_UNUSED(a);
    HPC_UNUSED(b);
    HPC_UNUSED(c);
}

void matrix_extract_diagonal(const MatrixBase* matrix, void* diagonal) {
    HPC_UNUSED(matrix);
    HPC_UNUSED(diagonal);
}

void matrix_set_diagonal(MatrixBase* matrix, const void* diagonal) {
    HPC_UNUSED(matrix);
    HPC_UNUSED(diagonal);
}

// Matrix statistics
MatrixStats compute_matrix_stats(const MatrixBase* matrix) {
    MatrixStats stats;
    stats.nnz = matrix->num_nonzeros();
    stats.density = static_cast<double>(stats.nnz) /
                    static_cast<double>(matrix->num_rows() * matrix->num_cols());

    // Additional stats would require format-specific access
    return stats;
}

HPC_NAMESPACE_END