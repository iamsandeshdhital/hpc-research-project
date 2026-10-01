#include "hpc/semiring.hpp"
#include <mutex>
#include <algorithm>

HPC_NAMESPACE_BEGIN

// SemiringRegistry implementation
SemiringID SemiringRegistry::register_semiring(const SemiringDescriptor& desc) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Check if name already exists
    auto it = name_to_id_.find(desc.name);
    if (it != name_to_id_.end()) {
        return it->second;
    }

    SemiringID id = next_id_++;
    descriptors_[id] = desc;
    name_to_id_[desc.name] = id;

    return id;
}

const SemiringDescriptor* SemiringRegistry::get_descriptor(SemiringID id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = descriptors_.find(id);
    if (it != descriptors_.end()) {
        return &it->second;
    }
    return nullptr;
}

SemiringID SemiringRegistry::get_id_by_name(const char* name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = name_to_id_.find(name);
    if (it != name_to_id_.end()) {
        return it->second;
    }
    return SEMIRING_ID_INVALID;
}

void SemiringRegistry::initialize_builtin_semirings() {
    // Built-in semirings are registered via the SemiringManager
}

std::mutex SemiringRegistry::mutex_;

// SemiringManager implementation
SemiringManager& SemiringManager::instance() {
    static SemiringManager manager;
    return manager;
}

const SemiringDescriptor* SemiringManager::get_semiring(SemiringID id) const {
    return SemiringRegistry::instance().get_descriptor(id);
}

SemiringID SemiringManager::get_semiring_id(const char* name) const {
    return SemiringRegistry::instance().get_id_by_name(name);
}

// Predefined semiring accessors
HPC_API SemiringID semiring_plus_times_int32() {
    return SemiringManager::plus_times_int32();
}

HPC_API SemiringID semiring_plus_times_int64() {
    return SemiringManager::plus_times_int64();
}

HPC_API SemiringID semiring_plus_times_float() {
    return SemiringManager::plus_times_float();
}

HPC_API SemiringID semiring_plus_times_double() {
    return SemiringManager::plus_times_double();
}

HPC_API SemiringID semiring_min_plus_int32() {
    return SemiringManager::min_plus_int32();
}

HPC_API SemiringID semiring_min_plus_int64() {
    return SemiringManager::min_plus_int64();
}

HPC_API SemiringID semiring_min_plus_float() {
    return SemiringManager::min_plus_float();
}

HPC_API SemiringID semiring_min_plus_double() {
    return SemiringManager::min_plus_double();
}

HPC_API SemiringID semiring_max_plus_int32() {
    return SemiringManager::max_plus_int32();
}

HPC_API SemiringID semiring_max_plus_int64() {
    return SemiringManager::max_plus_int64();
}

HPC_API SemiringID semiring_max_plus_float() {
    return SemiringManager::max_plus_float();
}

HPC_API SemiringID semiring_max_plus_double() {
    return SemiringManager::max_plus_double();
}

HPC_API SemiringID semiring_lor_land_bool() {
    return SemiringManager::lor_land_bool();
}

HPC_API SemiringID semiring_lor_land_int32() {
    return SemiringManager::lor_land_int32();
}

HPC_API SemiringID semiring_plus_first_int32() {
    return SemiringManager::plus_first_int32();
}

HPC_API SemiringID semiring_plus_second_int32() {
    return SemiringManager::plus_second_int32();
}

HPC_API SemiringID semiring_any_all_bool() {
    return SemiringManager::any_all_bool();
}

// Initialize all built-in semirings
struct SemiringInitializer {
    SemiringInitializer() {
        auto& manager = SemiringManager::instance();

        // Register built-in semirings
        manager.register_semiring<PlusTimesInt32>("plus_times_int32");
        manager.register_semiring<PlusTimesInt64>("plus_times_int64");
        manager.register_semiring<PlusTimesFloat>("plus_times_float");
        manager.register_semiring<PlusTimesDouble>("plus_times_double");

        manager.register_semiring<MinPlusInt32>("min_plus_int32");
        manager.register_semiring<MinPlusInt64>("min_plus_int64");
        manager.register_semiring<MinPlusFloat>("min_plus_float");
        manager.register_semiring<MinPlusDouble>("min_plus_double");

        manager.register_semiring<MaxPlusInt32>("max_plus_int32");
        manager.register_semiring<MaxPlusInt64>("max_plus_int64");
        manager.register_semiring<MaxPlusFloat>("max_plus_float");
        manager.register_semiring<MaxPlusDouble>("max_plus_double");

        manager.register_semiring<LorLandBool>("lor_land_bool");
        manager.register_semiring<LorLandInt32>("lor_land_int32");

        manager.register_semiring<PlusFirstInt32>("plus_first_int32");
        manager.register_semiring<PlusSecondInt32>("plus_second_int32");

        manager.register_semiring<AnyAllBool>("any_all_bool");
    }
};

// Static initializer
static SemiringInitializer g_semiring_initializer;

HPC_NAMESPACE_END