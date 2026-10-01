#pragma once

#include "config.hpp"
#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>
#include <memory>
#include <utility>

HPC_NAMESPACE_BEGIN

// Matrix format types
enum class MatrixFormat : uint8_t {
    COO = 0,      // Coordinate format
    CSR = 1,      // Compressed Sparse Row
    CSC = 2,      // Compressed Sparse Column
    DCSC = 3,     // Doubly Compressed Sparse Column
    DENSE = 4,    // Dense matrix
    ELL = 5,      // ELLPACK format
    HYB = 6       // Hybrid format
};

// Matrix properties
struct HPC_API MatrixProperties {
    bool symmetric = false;
    bool hermitian = false;
    bool positive_definite = false;
    bool diagonal_dominant = false;
    bool sorted_indices = false;
    uint32_t num_block_rows = 1;
    uint32_t num_block_cols = 1;
};

// Base matrix descriptor (type-erased)
class HPC_API MatrixBase {
public:
    virtual ~MatrixBase() = default;
    virtual MatrixFormat format() const = 0;
    virtual uint64_t num_rows() const = 0;
    virtual uint64_t num_cols() const = 0;
    virtual uint64_t num_nonzeros() const = 0;
    virtual size_t element_size() const = 0;
    virtual const void* values() const = 0;
    virtual void* mutable_values() = 0;
    virtual const MatrixProperties& properties() const = 0;
    virtual MatrixProperties& mutable_properties() = 0;
    virtual std::unique_ptr<MatrixBase> clone() const = 0;
    virtual void convert_to(MatrixFormat target_format) = 0;
    virtual void transpose() = 0;
    virtual void clear() = 0;
    virtual size_t memory_usage() const = 0;
};

// Templated matrix class
template<typename T, MatrixFormat Format>
class Matrix : public MatrixBase {
public:
    using value_type = T;
    static constexpr MatrixFormat format_value = Format;

    Matrix() = default;
    Matrix(uint64_t rows, uint64_t cols) : rows_(rows), cols_(cols) {}

    // Format-specific data structures
    // COO format
    std::vector<uint64_t> coo_rows_;
    std::vector<uint64_t> coo_cols_;
    std::vector<T> coo_values_;

    // CSR format
    std::vector<uint64_t> csr_row_ptr_;
    std::vector<uint64_t> csr_col_idx_;
    std::vector<T> csr_values_;

    // CSC format
    std::vector<uint64_t> csc_col_ptr_;
    std::vector<uint64_t> csc_row_idx_;
    std::vector<T> csc_values_;

    // DCSC format
    std::vector<uint64_t> dcsc_col_ptr_;
    std::vector<uint64_t> dcsc_row_idx_;
    std::vector<T> dcsc_values_;
    std::vector<uint64_t> dcsc_super_ptr_;
    std::vector<uint64_t> dcsc_super_idx_;

    // Dense format
    std::vector<T> dense_values_;

    // ELL format
    std::vector<uint64_t> ell_col_idx_;
    std::vector<T> ell_values_;
    uint64_t ell_max_nnz_per_row_ = 0;

    // Properties
    MatrixProperties properties_;

    HPC_HOST MatrixFormat format() const override { return Format; }
    HPC_HOST uint64_t num_rows() const override { return rows_; }
    HPC_HOST uint64_t num_cols() const override { return cols_; }
    HPC_HOST uint64_t num_nonzeros() const override { return nnz_; }
    HPC_HOST size_t element_size() const override { return sizeof(T); }
    HPC_HOST const void* values() const override { return get_values_ptr(); }
    HPC_HOST void* mutable_values() override { return get_values_ptr(); }
    HPC_HOST const MatrixProperties& properties() const override { return properties_; }
    HPC_HOST MatrixProperties& mutable_properties() override { return properties_; }

    virtual std::unique_ptr<MatrixBase> clone() const override {
        auto copy = std::make_unique<Matrix<T, Format>>(*this);
        return copy;
    }

    virtual void convert_to(MatrixFormat target_format) override;
    virtual void transpose() override;
    virtual void clear() override;
    virtual size_t memory_usage() const override;

    // Resize
    void resize(uint64_t rows, uint64_t cols, uint64_t nnz = 0);

    // Accessors for specific formats
    HPC_HOST_DEVICE const T* values_ptr() const { return get_values_ptr(); }
    HPC_HOST_DEVICE T* values_ptr() { return get_values_ptr(); }

    // Format-specific access
    // CSR
    HPC_HOST_DEVICE const uint64_t* csr_row_ptr() const { return csr_row_ptr_.data(); }
    HPC_HOST_DEVICE const uint64_t* csr_col_idx() const { return csr_col_idx_.data(); }
    HPC_HOST_DEVICE const T* csr_values() const { return csr_values_.data(); }
    HPC_HOST_DEVICE uint64_t* csr_row_ptr() { return csr_row_ptr_.data(); }
    HPC_HOST_DEVICE uint64_t* csr_col_idx() { return csr_col_idx_.data(); }
    HPC_HOST_DEVICE T* csr_values() { return csr_values_.data(); }

    // CSC
    HPC_HOST_DEVICE const uint64_t* csc_col_ptr() const { return csc_col_ptr_.data(); }
    HPC_HOST_DEVICE const uint64_t* csc_row_idx() const { return csc_row_idx_.data(); }
    HPC_HOST_DEVICE const T* csc_values() const { return csc_values_.data(); }

    // COO
    HPC_HOST_DEVICE const uint64_t* coo_rows() const { return coo_rows_.data(); }
    HPC_HOST_DEVICE const uint64_t* coo_cols() const { return coo_cols_.data(); }
    HPC_HOST_DEVICE const T* coo_values() const { return coo_values_.data(); }

    // DCSC
    HPC_HOST_DEVICE const uint64_t* dcsc_col_ptr() const { return dcsc_col_ptr_.data(); }
    HPC_HOST_DEVICE const uint64_t* dcsc_row_idx() const { return dcsc_row_idx_.data(); }
    HPC_HOST_DEVICE const T* dcsc_values() const { return dcsc_values_.data(); }
    HPC_HOST_DEVICE const uint64_t* dcsc_super_ptr() const { return dcsc_super_ptr_.data(); }
    HPC_HOST_DEVICE const uint64_t* dcsc_super_idx() const { return dcsc_super_idx_.data(); }

protected:
    uint64_t rows_ = 0;
    uint64_t cols_ = 0;
    uint64_t nnz_ = 0;

    const T* get_values_ptr() const {
        switch (Format) {
            case MatrixFormat::COO: return coo_values_.data();
            case MatrixFormat::CSR: return csr_values_.data();
            case MatrixFormat::CSC: return csc_values_.data();
            case MatrixFormat::DCSC: return dcsc_values_.data();
            case MatrixFormat::DENSE: return dense_values_.data();
            case MatrixFormat::ELL: return ell_values_.data();
            default: return nullptr;
        }
    }

    T* get_values_ptr() {
        return const_cast<T*>(const_cast<const Matrix*>(this)->get_values_ptr());
    }
};

// Type aliases for common matrices
template<typename T> using MatrixCOO = Matrix<T, MatrixFormat::COO>;
template<typename T> using MatrixCSR = Matrix<T, MatrixFormat::CSR>;
template<typename T> using MatrixCSC = Matrix<T, MatrixFormat::CSC>;
template<typename T> using MatrixDCSC = Matrix<T, MatrixFormat::DCSC>;
template<typename T> using MatrixDense = Matrix<T, MatrixFormat::DENSE>;
template<typename T> using MatrixELL = Matrix<T, MatrixFormat::ELL>;

// Matrix builder for fluent construction
template<typename T>
class MatrixBuilder {
public:
    MatrixBuilder& rows(uint64_t r) { rows_ = r; return *this; }
    MatrixBuilder& cols(uint64_t c) { cols_ = c; return *this; }
    MatrixBuilder& format(MatrixFormat f) { format_ = f; return *this; }
    MatrixBuilder& symmetric(bool s) { symmetric_ = s; return *this; }
    MatrixBuilder& reserve_nnz(uint64_t n) { reserve_nnz_ = n; return *this; }

    template<MatrixFormat Fmt>
    std::unique_ptr<Matrix<T, Fmt>> build() const {
        auto mat = std::make_unique<Matrix<T, Fmt>>(rows_, cols_);
        mat->properties_.symmetric = symmetric_;
        if (reserve_nnz_ > 0) {
            mat->resize(rows_, cols_, reserve_nnz_);
        }
        return mat;
    }

private:
    uint64_t rows_ = 0;
    uint64_t cols_ = 0;
    MatrixFormat format_ = MatrixFormat::CSR;
    bool symmetric_ = false;
    uint64_t reserve_nnz_ = 0;
};

// Matrix I/O functions
HPC_API bool read_matrix_market(const char* filename, MatrixBase** matrix);
HPC_API bool write_matrix_market(const char* filename, const MatrixBase* matrix);
HPC_API bool read_binary_matrix(const char* filename, MatrixBase** matrix);
HPC_API bool write_binary_matrix(const char* filename, const MatrixBase* matrix);

// Matrix generation functions
HPC_API std::unique_ptr<MatrixBase> generate_erdos_renyi(uint64_t n, double p, MatrixFormat fmt = MatrixFormat::CSR);
HPC_API std::unique_ptr<MatrixBase> generate_rmat(uint64_t scale, uint64_t edge_factor, MatrixFormat fmt = MatrixFormat::CSR);
HPC_API std::unique_ptr<MatrixBase> generate_grid_2d(uint64_t n, MatrixFormat fmt = MatrixFormat::CSR);
HPC_API std::unique_ptr<MatrixBase> generate_grid_3d(uint64_t n, MatrixFormat fmt = MatrixFormat::CSR);
HPC_API std::unique_ptr<MatrixBase> generate_random_sparse(uint64_t rows, uint64_t cols, double density, MatrixFormat fmt = MatrixFormat::CSR);

// Matrix operations
HPC_API void matrix_transpose(const MatrixBase* input, MatrixBase* output);
HPC_API void matrix_convert(const MatrixBase* input, MatrixFormat target_format, MatrixBase** output);
HPC_API void matrix_scale(MatrixBase* matrix, const void* alpha);
HPC_API void matrix_add(const MatrixBase* a, const MatrixBase* b, MatrixBase* c);
HPC_API void matrix_extract_diagonal(const MatrixBase* matrix, void* diagonal);
HPC_API void matrix_set_diagonal(MatrixBase* matrix, const void* diagonal);

// Matrix statistics
struct HPC_API MatrixStats {
    uint64_t nnz = 0;
    double density = 0.0;
    double avg_nnz_per_row = 0.0;
    uint64_t max_nnz_per_row = 0;
    uint64_t min_nnz_per_row = 0;
    double nnz_per_row_stddev = 0.0;
    bool is_symmetric = false;
    double symmetry_error = 0.0;
};

HPC_API MatrixStats compute_matrix_stats(const MatrixBase* matrix);

// Distributed matrix descriptor
struct HPC_API DistributedMatrixDescriptor {
    uint64_t global_rows = 0;
    uint64_t global_cols = 0;
    uint64_t local_rows = 0;
    uint64_t local_cols = 0;
    uint64_t row_offset = 0;
    uint64_t col_offset = 0;
    int rank = 0;
    int num_ranks = 1;
    int grid_rows = 1;
    int grid_cols = 1;
    int grid_layer = 0;  // For 2.5D
    int num_layers = 1;  // For 2.5D
};

HPC_NAMESPACE_END