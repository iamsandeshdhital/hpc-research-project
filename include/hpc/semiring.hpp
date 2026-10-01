#pragma once

#include "config.hpp"
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <functional>
#include <string>
#include <unordered_map>

HPC_NAMESPACE_BEGIN

// Forward declarations
template<typename T> struct SemiringTraits;

// Semiring base class for type erasure
class HPC_API SemiringBase {
public:
    virtual ~SemiringBase() = default;
    virtual const char* name() const = 0;
    virtual size_t element_size() const = 0;
    virtual void* add(const void* a, const void* b, void* result) const = 0;
    virtual void* multiply(const void* a, const void* b, void* result) const = 0;
    virtual void* identity_add(void* result) const = 0;
    virtual void* identity_multiply(void* result) const = 0;
    virtual bool equals(const void* a, const void* b) const = 0;
    virtual void copy(const void* src, void* dst) const = 0;
    virtual void destroy(void* ptr) const = 0;
};

// Semiring ID type
using SemiringID = uint32_t;
constexpr SemiringID SEMIRING_ID_INVALID = 0xFFFFFFFF;

// Predefined semiring IDs
enum class SemiringType : SemiringID {
    PLUS_TIMES_INT32     = 1,
    PLUS_TIMES_INT64     = 2,
    PLUS_TIMES_FLOAT     = 3,
    PLUS_TIMES_DOUBLE    = 4,
    MIN_PLUS_INT32       = 5,
    MIN_PLUS_INT64       = 6,
    MIN_PLUS_FLOAT       = 7,
    MIN_PLUS_DOUBLE      = 8,
    MAX_PLUS_INT32       = 9,
    MAX_PLUS_INT64       = 10,
    MAX_PLUS_FLOAT       = 11,
    MAX_PLUS_DOUBLE      = 12,
    LOR_LAND_BOOL        = 13,
    LOR_LAND_INT32       = 14,
    PLUS_FIRST_INT32     = 15,
    PLUS_SECOND_INT32    = 16,
    ANY_ALL_BOOL         = 17,
    CUSTOM               = 0xFFFFFFF0
};

// Semiring descriptor for registration
struct HPC_API SemiringDescriptor {
    SemiringID id;
    const char* name;
    size_t element_size;
    std::function<void*(const void*, const void*, void*)> add_op;
    std::function<void*(const void*, const void*, void*)> mul_op;
    std::function<void*(void*)> add_identity;
    std::function<void*(void*)> mul_identity;
    std::function<bool(const void*, const void*)> equals_op;
    std::function<void(const void*, void*)> copy_op;
    std::function<void(void*)> destroy_op;
};

// Global semiring registry
class HPC_API SemiringRegistry {
public:
    static SemiringRegistry& instance() {
        static SemiringRegistry registry;
        return registry;
    }

    SemiringID register_semiring(const SemiringDescriptor& desc);
    const SemiringDescriptor* get_descriptor(SemiringID id) const;
    SemiringID get_id_by_name(const char* name) const;
    void initialize_builtin_semirings();

private:
    SemiringRegistry() = default;
    std::unordered_map<SemiringID, SemiringDescriptor> descriptors_;
    std::unordered_map<std::string, SemiringID> name_to_id_;
    SemiringID next_id_ = static_cast<SemiringID>(SemiringType::CUSTOM) + 1;
};

// Templated semiring implementation
template<typename T, typename AddOp, typename MulOp, T AddIdentity, T MulIdentity>
class Semiring : public SemiringBase {
public:
    using value_type = T;
    using add_op_type = AddOp;
    using mul_op_type = MulOp;

    static constexpr T add_identity = AddIdentity;
    static constexpr T mul_identity = MulIdentity;

    HPC_HOST_DEVICE Semiring() = default;

    HPC_HOST_DEVICE const char* name() const override {
        return traits_.name;
    }

    HPC_HOST_DEVICE size_t element_size() const override {
        return sizeof(T);
    }

    HPC_HOST_DEVICE void* add(const void* a, const void* b, void* result) const override {
        const T* lhs = static_cast<const T*>(a);
        const T* rhs = static_cast<const T*>(b);
        T* res = static_cast<T*>(result);
        *res = AddOp{}(*lhs, *rhs);
        return result;
    }

    HPC_HOST_DEVICE void* multiply(const void* a, const void* b, void* result) const override {
        const T* lhs = static_cast<const T*>(a);
        const T* rhs = static_cast<const T*>(b);
        T* res = static_cast<T*>(result);
        *res = MulOp{}(*lhs, *rhs);
        return result;
    }

    HPC_HOST_DEVICE void* identity_add(void* result) const override {
        T* res = static_cast<T*>(result);
        *res = AddIdentity;
        return result;
    }

    HPC_HOST_DEVICE void* identity_multiply(void* result) const override {
        T* res = static_cast<T*>(result);
        *res = MulIdentity;
        return result;
    }

    HPC_HOST_DEVICE bool equals(const void* a, const void* b) const override {
        const T* lhs = static_cast<const T*>(a);
        const T* rhs = static_cast<const T*>(b);
        return *lhs == *rhs;
    }

    HPC_HOST_DEVICE void copy(const void* src, void* dst) const override {
        const T* s = static_cast<const T*>(src);
        T* d = static_cast<T*>(dst);
        *d = *s;
    }

    HPC_HOST_DEVICE void destroy(void* ptr) const override {
        // No-op for trivially destructible types
        (void)ptr;
    }

    struct Traits {
        const char* name;
    } traits_;
};

// Built-in semiring operators
struct Plus {
    template<typename T>
    HPC_HOST_DEVICE T operator()(T a, T b) const { return a + b; }
};

struct Times {
    template<typename T>
    HPC_HOST_DEVICE T operator()(T a, T b) const { return a * b; }
};

struct Min {
    template<typename T>
    HPC_HOST_DEVICE T operator()(T a, T b) const { return a < b ? a : b; }
};

struct Max {
    template<typename T>
    HPC_HOST_DEVICE T operator()(T a, T b) const { return a > b ? a : b; }
};

struct LogicalOr {
    template<typename T>
    HPC_HOST_DEVICE T operator()(T a, T b) const { return a || b; }
};

struct LogicalAnd {
    template<typename T>
    HPC_HOST_DEVICE T operator()(T a, T b) const { return a && b; }
};

struct First {
    template<typename T>
    HPC_HOST_DEVICE T operator()(T a, T b) const { return a; }
};

struct Second {
    template<typename T>
    HPC_HOST_DEVICE T operator()(T a, T b) const { return b; }
};

// Type traits for identity elements
template<typename T, typename Op>
struct IdentityElement;

template<typename T>
struct IdentityElement<T, Plus> {
    static constexpr T value = T(0);
};

template<typename T>
struct IdentityElement<T, Times> {
    static constexpr T value = T(1);
};

template<typename T>
struct IdentityElement<T, Min> {
    static constexpr T value = std::numeric_limits<T>::max();
};

template<typename T>
struct IdentityElement<T, Max> {
    static constexpr T value = std::numeric_limits<T>::lowest();
};

template<typename T>
struct IdentityElement<T, LogicalOr> {
    static constexpr T value = T(false);
};

template<typename T>
struct IdentityElement<T, LogicalAnd> {
    static constexpr T value = T(true);
};

template<typename T>
struct IdentityElement<T, First> {
    static constexpr T value = T(0);
};

template<typename T>
struct IdentityElement<T, Second> {
    static constexpr T value = T(0);
};

// Predefined semiring types
using PlusTimesInt32    = Semiring<int32_t, Plus, Times, IdentityElement<int32_t, Plus>::value, IdentityElement<int32_t, Times>::value>;
using PlusTimesInt64    = Semiring<int64_t, Plus, Times, IdentityElement<int64_t, Plus>::value, IdentityElement<int64_t, Times>::value>;
using PlusTimesFloat    = Semiring<float, Plus, Times, IdentityElement<float, Plus>::value, IdentityElement<float, Times>::value>;
using PlusTimesDouble   = Semiring<double, Plus, Times, IdentityElement<double, Plus>::value, IdentityElement<double, Times>::value>;

using MinPlusInt32      = Semiring<int32_t, Min, Plus, IdentityElement<int32_t, Min>::value, IdentityElement<int32_t, Plus>::value>;
using MinPlusInt64      = Semiring<int64_t, Min, Plus, IdentityElement<int64_t, Min>::value, IdentityElement<int64_t, Plus>::value>;
using MinPlusFloat      = Semiring<float, Min, Plus, IdentityElement<float, Min>::value, IdentityElement<float, Plus>::value>;
using MinPlusDouble     = Semiring<double, Min, Plus, IdentityElement<double, Min>::value, IdentityElement<double, Plus>::value>;

using MaxPlusInt32      = Semiring<int32_t, Max, Plus, IdentityElement<int32_t, Max>::value, IdentityElement<int32_t, Plus>::value>;
using MaxPlusInt64      = Semiring<int64_t, Max, Plus, IdentityElement<int64_t, Max>::value, IdentityElement<int64_t, Plus>::value>;
using MaxPlusFloat      = Semiring<float, Max, Plus, IdentityElement<float, Max>::value, IdentityElement<float, Plus>::value>;
using MaxPlusDouble     = Semiring<double, Max, Plus, IdentityElement<double, Max>::value, IdentityElement<double, Plus>::value>;

using LorLandBool       = Semiring<bool, LogicalOr, LogicalAnd, IdentityElement<bool, LogicalOr>::value, IdentityElement<bool, LogicalAnd>::value>;
using LorLandInt32      = Semiring<int32_t, LogicalOr, LogicalAnd, IdentityElement<int32_t, LogicalOr>::value, IdentityElement<int32_t, LogicalAnd>::value>;

using PlusFirstInt32    = Semiring<int32_t, Plus, First, IdentityElement<int32_t, Plus>::value, IdentityElement<int32_t, First>::value>;
using PlusSecondInt32   = Semiring<int32_t, Plus, Second, IdentityElement<int32_t, Plus>::value, IdentityElement<int32_t, Second>::value>;

using AnyAllBool        = Semiring<bool, LogicalOr, LogicalAnd, IdentityElement<bool, LogicalOr>::value, IdentityElement<bool, LogicalAnd>::value>;

// Semiring manager for runtime selection
class HPC_API SemiringManager {
public:
    static SemiringManager& instance();

    template<typename SemiringT>
    SemiringID register_semiring(const char* name) {
        SemiringT semiring;
        semiring.traits_.name = name;

        SemiringDescriptor desc;
        desc.id = static_cast<SemiringID>(SemiringType::CUSTOM) + next_custom_id_++;
        desc.name = name;
        desc.element_size = sizeof(typename SemiringT::value_type);
        desc.add_op = [](const void* a, const void* b, void* result) -> void* {
            return SemiringT{}.add(a, b, result);
        };
        desc.mul_op = [](const void* a, const void* b, void* result) -> void* {
            return SemiringT{}.multiply(a, b, result);
        };
        desc.add_identity = [](void* result) -> void* {
            return SemiringT{}.identity_add(result);
        };
        desc.mul_identity = [](void* result) -> void* {
            return SemiringT{}.identity_multiply(result);
        };
        desc.equals_op = [](const void* a, const void* b) -> bool {
            return SemiringT{}.equals(a, b);
        };
        desc.copy_op = [](const void* src, void* dst) -> void {
            SemiringT{}.copy(src, dst);
        };
        desc.destroy_op = [](void* ptr) -> void {
            SemiringT{}.destroy(ptr);
        };

        return SemiringRegistry::instance().register_semiring(desc);
    }

    const SemiringDescriptor* get_semiring(SemiringID id) const;
    SemiringID get_semiring_id(const char* name) const;

    // Get predefined semiring IDs
    static constexpr SemiringID plus_times_int32() { return static_cast<SemiringID>(SemiringType::PLUS_TIMES_INT32); }
    static constexpr SemiringID plus_times_int64() { return static_cast<SemiringID>(SemiringType::PLUS_TIMES_INT64); }
    static constexpr SemiringID plus_times_float() { return static_cast<SemiringID>(SemiringType::PLUS_TIMES_FLOAT); }
    static constexpr SemiringID plus_times_double() { return static_cast<SemiringID>(SemiringType::PLUS_TIMES_DOUBLE); }
    static constexpr SemiringID min_plus_int32() { return static_cast<SemiringID>(SemiringType::MIN_PLUS_INT32); }
    static constexpr SemiringID min_plus_int64() { return static_cast<SemiringID>(SemiringType::MIN_PLUS_INT64); }
    static constexpr SemiringID min_plus_float() { return static_cast<SemiringID>(SemiringType::MIN_PLUS_FLOAT); }
    static constexpr SemiringID min_plus_double() { return static_cast<SemiringID>(SemiringType::MIN_PLUS_DOUBLE); }
    static constexpr SemiringID max_plus_int32() { return static_cast<SemiringID>(SemiringType::MAX_PLUS_INT32); }
    static constexpr SemiringID max_plus_int64() { return static_cast<SemiringID>(SemiringType::MAX_PLUS_INT64); }
    static constexpr SemiringID max_plus_float() { return static_cast<SemiringID>(SemiringType::MAX_PLUS_FLOAT); }
    static constexpr SemiringID max_plus_double() { return static_cast<SemiringID>(SemiringType::MAX_PLUS_DOUBLE); }
    static constexpr SemiringID lor_land_bool() { return static_cast<SemiringID>(SemiringType::LOR_LAND_BOOL); }
    static constexpr SemiringID lor_land_int32() { return static_cast<SemiringID>(SemiringType::LOR_LAND_INT32); }
    static constexpr SemiringID plus_first_int32() { return static_cast<SemiringID>(SemiringType::PLUS_FIRST_INT32); }
    static constexpr SemiringID plus_second_int32() { return static_cast<SemiringID>(SemiringType::PLUS_SECOND_INT32); }
    static constexpr SemiringID any_all_bool() { return static_cast<SemiringID>(SemiringType::ANY_ALL_BOOL); }

private:
    SemiringManager() = default;
    uint32_t next_custom_id_ = 1;
};

// Convenience functions
HPC_API SemiringID semiring_plus_times_int32();
HPC_API SemiringID semiring_plus_times_int64();
HPC_API SemiringID semiring_plus_times_float();
HPC_API SemiringID semiring_plus_times_double();
HPC_API SemiringID semiring_min_plus_int32();
HPC_API SemiringID semiring_min_plus_int64();
HPC_API SemiringID semiring_min_plus_float();
HPC_API SemiringID semiring_min_plus_double();
HPC_API SemiringID semiring_max_plus_int32();
HPC_API SemiringID semiring_max_plus_int64();
HPC_API SemiringID semiring_max_plus_float();
HPC_API SemiringID semiring_max_plus_double();
HPC_API SemiringID semiring_lor_land_bool();
HPC_API SemiringID semiring_lor_land_int32();
HPC_API SemiringID semiring_plus_first_int32();
HPC_API SemiringID semiring_plus_second_int32();
HPC_API SemiringID semiring_any_all_bool();

HPC_NAMESPACE_END