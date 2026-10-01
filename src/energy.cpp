#include "hpc/energy.hpp"
#include <atomic>
#include <chrono>
#include <thread>
#include <mutex>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <algorithm>

#if __has_include(<nvml.h>)
#define HPC_HAVE_NVML_HEADER 1
#else
#define HPC_HAVE_NVML_HEADER 0
#endif

#if HPC_HAVE_NVML_HEADER
#include <nvml.h>
#endif

HPC_NAMESPACE_BEGIN

namespace {

inline uint64_t now_ns() {
    using namespace std::chrono;
    return duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
}

inline uint64_t now_wall_ns() {
    using namespace std::chrono;
    return duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
}

} // namespace

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

double convert_energy(double value, EnergyUnit from, EnergyUnit to) {
    // Normalise to joules first.
    double joules = value;
    switch (from) {
        case EnergyUnit::JOULE:          joules = value; break;
        case EnergyUnit::KILOWATT_HOUR:  joules = value * 3.6e6; break;
        case EnergyUnit::MILLIJOULE:     joules = value * 1e-3; break;
        case EnergyUnit::MICROJOULE:     joules = value * 1e-6; break;
        case EnergyUnit::WATT:           joules = value; break; // assumes 1 second
    }
    switch (to) {
        case EnergyUnit::JOULE:          return joules;
        case EnergyUnit::KILOWATT_HOUR:  return joules / 3.6e6;
        case EnergyUnit::MILLIJOULE:     return joules * 1e3;
        case EnergyUnit::MICROJOULE:     return joules * 1e6;
        case EnergyUnit::WATT:           return joules;
    }
    return joules;
}

double convert_power(double value, EnergyUnit from, EnergyUnit to) {
    (void)to;
    switch (from) {
        case EnergyUnit::WATT:          return value;
        case EnergyUnit::KILOWATT_HOUR: return value / 3600.0; // W per kWh rate
        default:                        return value;
    }
}

std::string energy_unit_to_string(EnergyUnit unit) {
    switch (unit) {
        case EnergyUnit::JOULE:         return "J";
        case EnergyUnit::WATT:          return "W";
        case EnergyUnit::KILOWATT_HOUR: return "kWh";
        case EnergyUnit::MILLIJOULE:    return "mJ";
        case EnergyUnit::MICROJOULE:    return "uJ";
    }
    return "unknown";
}

std::string power_domain_to_string(PowerDomain domain) {
    switch (domain) {
        case PowerDomain::PACKAGE: return "package";
        case PowerDomain::CORE:    return "core";
        case PowerDomain::UNCORE:  return "uncore";
        case PowerDomain::DRAM:    return "dram";
        case PowerDomain::GPU:     return "gpu";
        case PowerDomain::GPU_MEMORY: return "gpu_memory";
        case PowerDomain::GPU_SM:  return "gpu_sm";
        case PowerDomain::PLATFORM:return "platform";
        case PowerDomain::CUSTOM:  return "custom";
    }
    return "unknown";
}

std::string gpu_power_state_to_string(GPUPowerState state) {
    switch (state) {
        case GPUPowerState::UNKNOWN:      return "unknown";
        case GPUPowerState::ACTIVE:       return "active";
        case GPUPowerState::IDLE:         return "idle";
        case GPUPowerState::THROTTLED:    return "throttled";
        case GPUPowerState::POWER_GATED:  return "power_gated";
    }
    return "unknown";
}

// ---------------------------------------------------------------------------
// NVMLMonitor
// ---------------------------------------------------------------------------

struct NVMLMonitor::Impl {
    EnergyConfig config;
    EnergyStats  stats;
    bool         initialized = false;
    bool         running = false;
#if HPC_HAVE_NVML_HEADER
    nvmlDevice_t device = nullptr;
#endif
    std::mutex   mutex;
    std::thread  sampler;
    std::atomic<bool> stop_flag{false};
    double       last_integral_j = 0.0;
    double       first_integral_j = 0.0;
    uint64_t     last_sample_ns = 0;
    uint64_t     first_sample_ns = 0;
    std::ofstream log_stream;
};

NVMLMonitor::NVMLMonitor() : impl_(std::make_unique<Impl>()) {}
NVMLMonitor::~NVMLMonitor() {
    stop();
#if HPC_HAVE_NVML_HEADER
    if (impl_->initialized) {
        nvmlDevice_t d = impl_->device;
        if (d) nvmlDeviceReleaseHandle(d);
        nvmlShutdown();
    }
#endif
}

int NVMLMonitor::initialize(const EnergyConfig& config) {
    impl_->config = config;

#if HPC_HAVE_NVML_HEADER
    nvmlReturn_t ret = nvmlInit();
    if (ret != NVML_SUCCESS) {
        fprintf(stderr, "[NVML] nvmlInit failed: %s\n", nvmlErrorString(ret));
        return static_cast<int>(ErrorCode::UNSUPPORTED_OPERATION);
    }

    ret = nvmlDeviceGetHandleByIndex(config.gpu_device_id, &impl_->device);
    if (ret != NVML_SUCCESS) {
        fprintf(stderr, "[NVML] device handle failed: %s\n", nvmlErrorString(ret));
        nvmlShutdown();
        return static_cast<int>(ErrorCode::UNSUPPORTED_OPERATION);
    }

    char name[NVML_DEVICE_NAME_V2_BUFFER_SIZE];
    nvmlDeviceGetName(impl_->device, name, NVML_DEVICE_NAME_V2_BUFFER_SIZE);

    if (config.log_to_file && config.log_csv) {
        impl_->log_stream.open(config.log_file_path);
        if (impl_->log_stream.is_open()) {
            impl_->log_stream << "timestamp_ns,energy_j,power_w,domain,device_id,"
                                 "gpu_state,sm_clock_mhz,mem_clock_mhz,temp_c,power_limit_w\n";
        }
    }

    impl_->initialized = true;
    fprintf(stderr, "[NVML] monitoring GPU %d (%s)\n", config.gpu_device_id, name);
    return static_cast<int>(ErrorCode::SUCCESS);
#else
    (void)config;
    fprintf(stderr, "[NVML] nvml.h not available at build time; "
                    "install the NVIDIA driver development headers to enable.\n");
    return static_cast<int>(ErrorCode::UNSUPPORTED_OPERATION);
#endif
}

int NVMLMonitor::start() {
    if (!impl_->initialized) return static_cast<int>(ErrorCode::NOT_INITIALIZED);
    if (impl_->running) return static_cast<int>(ErrorCode::SUCCESS);

    impl_->stop_flag = false;
    impl_->running = true;
    impl_->last_sample_ns = now_ns();
    impl_->last_integral_j = 0.0;

#if HPC_HAVE_NVML_HEADER
    if (impl_->config.continuous_sampling) {
        impl_->sampler = std::thread([this]() {
            while (!impl_->stop_flag.load()) {
                EnergySample s{};
                sample(&s);
                std::this_thread::sleep_for(std::chrono::microseconds(
                    impl_->config.sample_interval_ns / 1000));
            }
        });
    }
#endif
    return static_cast<int>(ErrorCode::SUCCESS);
}

int NVMLMonitor::stop() {
    if (!impl_->running) return static_cast<int>(ErrorCode::SUCCESS);
    impl_->stop_flag = true;
    if (impl_->sampler.joinable()) impl_->sampler.join();
    impl_->running = false;
    if (impl_->log_stream.is_open()) impl_->log_stream.close();
    return static_cast<int>(ErrorCode::SUCCESS);
}

int NVMLMonitor::sample(EnergySample* sample) {
    if (!sample) return static_cast<int>(ErrorCode::INVALID_ARGUMENT);

#if HPC_HAVE_NVML_HEADER
    if (!impl_->initialized) return static_cast<int>(ErrorCode::NOT_INITIALIZED);

    std::lock_guard<std::mutex> lock(impl_->mutex);

    unsigned int pwr_mw = 0, pwr_limit_mw = 0, temp = 0, sm_clk = 0, mem_clk = 0;

    nvmlDevice_t d = impl_->device;
    nvmlDeviceGetPowerUsage(d, &pwr_mw);
    nvmlDeviceGetEnforcedPowerLimit(d, &pwr_limit_mw);
    nvmlDeviceGetTemperature(d, NVML_TEMPERATURE_GPU, &temp);
    nvmlDeviceGetClockInfo(d, NVML_CLOCK_SM, &sm_clk);
    nvmlDeviceGetClockInfo(d, NVML_CLOCK_MEM, &mem_clk);

    nvmlPstate_t pstate;
    GPUPowerState state = GPUPowerState::UNKNOWN;
    if (nvmlDeviceGetPState(d, &pstate) == NVML_SUCCESS) {
        if (pstate == NVML_PSTATE_0) state = GPUPowerState::ACTIVE;
        else if (pstate >= NVML_PSTATE_8) state = GPUPowerState::POWER_GATED;
        else state = GPUPowerState::IDLE;
    }
    nvmlDeviceGetClockThrottleReasons(d, nullptr);

    const uint64_t t = now_ns();
    const double power_w = pwr_mw / 1000.0;
    const double dt = (t - impl_->last_sample_ns) / 1e9;
    const double energy_j = impl_->last_integral_j + power_w * dt;
    if (impl_->first_sample_ns == 0) {
        impl_->first_sample_ns = impl_->last_sample_ns;
        impl_->first_integral_j = impl_->last_integral_j;
    }
    impl_->last_integral_j = energy_j;
    impl_->last_sample_ns = t;

    sample->timestamp_ns = now_wall_ns();
    sample->energy_joules = energy_j;
    sample->power_watts = power_w;
    sample->domain = PowerDomain::GPU;
    sample->device_id = impl_->config.gpu_device_id;
    sample->gpu_state = state;
    sample->gpu_sm_clock_mhz = sm_clk;
    sample->gpu_mem_clock_mhz = mem_clk;
    sample->gpu_temperature_c = temp;
    sample->gpu_power_limit_w = pwr_limit_mw / 1000;
    sample->gpu_current_power_w = pwr_mw / 1000;

    // Accumulate stats
    auto& st = impl_->stats;
    st.total_energy_joules = energy_j;
    st.peak_power_watts = std::max(st.peak_power_watts, power_w);
    if (st.min_power_watts == 0.0 || power_w < st.min_power_watts) st.min_power_watts = power_w;
    if (impl_->first_sample_ns == 0) impl_->first_sample_ns = t;
    st.duration_ns = t - impl_->first_sample_ns;
    st.avg_power_watts = st.duration_ns
        ? (energy_j - impl_->first_integral_j) / (st.duration_ns / 1e9)
        : power_w;
    ++st.num_samples;
    if (st.samples.size() < impl_->config.max_samples) st.samples.push_back(*sample);

    // Alert hook
    if (impl_->config.enable_power_alerts && impl_->config.alert_callback) {
        if (impl_->config.power_threshold_w > 0.0 && power_w > impl_->config.power_threshold_w) {
            impl_->config.alert_callback(*sample);
        }
    }

    if (impl_->log_stream.is_open()) {
        impl_->log_stream << sample->timestamp_ns << ',' << sample->energy_joules << ','
                          << sample->power_watts << ",gpu," << sample->device_id << ','
                          << gpu_power_state_to_string(state) << ',' << sm_clk << ','
                          << mem_clk << ',' << temp << ',' << pwr_limit_mw << '\n';
        impl_->log_stream.flush();
    }
    return static_cast<int>(ErrorCode::SUCCESS);
#else
    (void)impl_;
    return static_cast<int>(ErrorCode::UNSUPPORTED_OPERATION);
#endif
}

const EnergyStats& NVMLMonitor::stats() const { return impl_->stats; }
EnergyStats& NVMLMonitor::stats() { return impl_->stats; }
void NVMLMonitor::reset_stats() {
    impl_->stats.reset();
    impl_->last_integral_j = 0.0;
    impl_->last_sample_ns = now_ns();
}

int NVMLMonitor::set_power_limit(uint32_t watts) {
#if HPC_HAVE_NVML_HEADER
    if (!impl_->initialized) return static_cast<int>(ErrorCode::NOT_INITIALIZED);
    nvmlReturn_t r = nvmlDeviceSetPowerManagementLimit(impl_->device, watts);
    return r == NVML_SUCCESS ? static_cast<int>(ErrorCode::SUCCESS)
                             : static_cast<int>(ErrorCode::UNSUPPORTED_OPERATION);
#else
    (void)watts;
    return static_cast<int>(ErrorCode::UNSUPPORTED_OPERATION);
#endif
}

int NVMLMonitor::set_sm_clock(uint32_t mhz) {
#if HPC_HAVE_NVML_HEADER
    if (!impl_->initialized) return static_cast<int>(ErrorCode::NOT_INITIALIZED);
    nvmlReturn_t r = nvmlDeviceSetGpuClockInfo(impl_->device, mhz);
    return r == NVML_SUCCESS ? static_cast<int>(ErrorCode::SUCCESS)
                             : static_cast<int>(ErrorCode::UNSUPPORTED_OPERATION);
#else
    (void)mhz;
    return static_cast<int>(ErrorCode::UNSUPPORTED_OPERATION);
#endif
}

int NVMLMonitor::set_mem_clock(uint32_t mhz) {
#if HPC_HAVE_NVML_HEADER
    if (!impl_->initialized) return static_cast<int>(ErrorCode::NOT_INITIALIZED);
    nvmlReturn_t r = nvmlDeviceSetMemoryClockInfo(impl_->device, mhz);
    return r == NVML_SUCCESS ? static_cast<int>(ErrorCode::SUCCESS)
                             : static_cast<int>(ErrorCode::UNSUPPORTED_OPERATION);
#else
    (void)mhz;
    return static_cast<int>(ErrorCode::UNSUPPORTED_OPERATION);
#endif
}

int NVMLMonitor::apply_dvfs_policy(const DVFSConfig& config) {
    if (!impl_->initialized) return static_cast<int>(ErrorCode::NOT_INITIALIZED);

    // Query supported clocks and pick the one matching the policy.
    uint32_t min_clk = 0, max_clk = 0;
#if HPC_HAVE_NVML_HEADER
    nvmlDeviceGetMinMaxClockInfo(impl_->device, NVML_CLOCK_SM, &min_clk, &max_clk);
#endif

    const uint32_t lo = config.min_frequency_mhz ? config.min_frequency_mhz : min_clk;
    const uint32_t hi = config.max_frequency_mhz ? config.max_frequency_mhz : max_clk;

    double w_perf = config.performance_weight;
    double w_en = config.energy_weight;
    if (w_perf + w_en <= 0.0) { w_perf = 0.5; w_en = 0.5; }

    uint32_t target = hi;
    switch (config.policy) {
        case DVFSPolicy::PERFORMANCE:     target = hi; break;
        case DVFSPolicy::BALANCED:
            target = static_cast<uint32_t>(lo + (hi - lo) * (w_en / (w_perf + w_en)));
            break;
        case DVFSPolicy::ENERGY_EFFICIENT: target = lo; break;
        case DVFSPolicy::CUSTOM:
            target = config.target_power_w ? hi : lo;
            break;
    }

    // Align to the configured step.
    if (config.step_frequency_mhz > 0) {
        target = ((target / config.step_frequency_mhz) + 1) * config.step_frequency_mhz;
        target = std::min(target, hi);
        target = std::max(target, lo);
    }

    int rc = set_sm_clock(target);
    if (config.target_power_w) set_power_limit(config.target_power_w);
    return rc;
}

int NVMLMonitor::get_gpu_metrics(GPUMetrics* metrics) {
    if (!metrics) return static_cast<int>(ErrorCode::INVALID_ARGUMENT);

#if HPC_HAVE_NVML_HEADER
    if (!impl_->initialized) return static_cast<int>(ErrorCode::NOT_INITIALIZED);

    nvmlDevice_t d = impl_->device;
    nvmlDeviceGetName(d, metrics->name, sizeof(metrics->name) - 1);
    nvmlDeviceGetClockInfo(d, NVML_CLOCK_SM, &metrics->sm_clock_mhz);
    nvmlDeviceGetClockInfo(d, NVML_CLOCK_MEM, &metrics->mem_clock_mhz);
    nvmlDeviceGetTemperature(d, NVML_TEMPERATURE_GPU, &metrics->temperature_c);
    nvmlDeviceGetFanSpeed(d, &metrics->fan_speed_percent);

    unsigned int pwr = 0, limit = 0;
    nvmlDeviceGetPowerUsage(d, &pwr);
    nvmlDeviceGetEnforcedPowerLimit(d, &limit);
    metrics->power_w = pwr / 1000;
    metrics->power_limit_w = limit / 1000;

    nvmlMemory_t mem;
    if (nvmlDeviceGetMemoryInfo(d, &mem) == NVML_SUCCESS) {
        metrics->memory_used_bytes = mem.used;
        metrics->memory_total_bytes = mem.total;
    }

    nvmlUtilization_t util;
    if (nvmlDeviceGetUtilizationRates(d, &util) == NVML_SUCCESS) {
        metrics->sm_utilization = util.gpu / 100.0;
        metrics->mem_utilization = util.memory / 100.0;
    }
    if (nvmlDeviceGetEncoderUtilization(d, &util) == NVML_SUCCESS) {
        metrics->encoder_utilization = util.encoder / 100.0;
    }
    if (nvmlDeviceGetDecoderUtilization(d, &util) == NVML_SUCCESS) {
        metrics->decoder_utilization = util.decoder / 100.0;
    }

    nvmlPstate_t pstate;
    if (nvmlDeviceGetPState(d, &pstate) == NVML_SUCCESS) {
        metrics->power_state = (pstate == NVML_PSTATE_0) ? GPUPowerState::ACTIVE : GPUPowerState::IDLE;
    }

    metrics->device_id = impl_->config.gpu_device_id;
    return static_cast<int>(ErrorCode::SUCCESS);
#else
    return static_cast<int>(ErrorCode::UNSUPPORTED_OPERATION);
#endif
}

bool NVMLMonitor::supports_domain(PowerDomain domain) const {
    return domain == PowerDomain::GPU || domain == PowerDomain::GPU_MEMORY ||
           domain == PowerDomain::GPU_SM;
}
bool NVMLMonitor::supports_dvfs() const { return HPC_HAVE_NVML_HEADER; }
std::vector<PowerDomain> NVMLMonitor::available_domains() const {
    return {PowerDomain::GPU, PowerDomain::GPU_MEMORY, PowerDomain::GPU_SM};
}

// ---------------------------------------------------------------------------
// ROCmMonitor (AMD) - uses rocm_smi when available
// ---------------------------------------------------------------------------

struct ROCmMonitor::Impl {
    EnergyConfig config;
    EnergyStats  stats;
    bool         initialized = false;
    double       last_integral_j = 0.0;
    uint64_t     last_sample_ns = 0;
};

ROCmMonitor::ROCmMonitor() : impl_(std::make_unique<Impl>()) {}
ROCmMonitor::~ROCmMonitor() = default;

int ROCmMonitor::initialize(const EnergyConfig& config) {
    impl_->config = config;
    // rocm_smi integration is guarded behind HPC_HAVE_ROCM_SMI
#if defined(HPC_HAVE_ROCM_SMI) && HPC_HAVE_ROCM_SMI
    impl_->initialized = true;
    return static_cast<int>(ErrorCode::SUCCESS);
#else
    fprintf(stderr, "[ROCm-SMI] not available at build time.\n");
    return static_cast<int>(ErrorCode::UNSUPPORTED_OPERATION);
#endif
}
int ROCmMonitor::start() { return static_cast<int>(ErrorCode::SUCCESS); }
int ROCmMonitor::stop()  { return static_cast<int>(ErrorCode::SUCCESS); }

int ROCmMonitor::sample(EnergySample* sample) {
    if (!sample) return static_cast<int>(ErrorCode::INVALID_ARGUMENT);
#if defined(HPC_HAVE_ROCM_SMI) && HPC_HAVE_ROCM_SMI
    // Fill from rocm_smi_enl_* API
    return static_cast<int>(ErrorCode::SUCCESS);
#else
    return static_cast<int>(ErrorCode::UNSUPPORTED_OPERATION);
#endif
}

const EnergyStats& ROCmMonitor::stats() const { return impl_->stats; }
EnergyStats& ROCmMonitor::stats() { return impl_->stats; }
void ROCmMonitor::reset_stats() { impl_->stats.reset(); }
int ROCmMonitor::set_power_limit(uint32_t) { return static_cast<int>(ErrorCode::UNSUPPORTED_OPERATION); }
int ROCmMonitor::set_sm_clock(uint32_t) { return static_cast<int>(ErrorCode::UNSUPPORTED_OPERATION); }
int ROCmMonitor::set_mem_clock(uint32_t) { return static_cast<int>(ErrorCode::UNSUPPORTED_OPERATION); }
int ROCmMonitor::apply_dvfs_policy(const DVFSConfig&) { return static_cast<int>(ErrorCode::UNSUPPORTED_OPERATION); }
int ROCmMonitor::get_gpu_metrics(GPUMetrics*) { return static_cast<int>(ErrorCode::UNSUPPORTED_OPERATION); }
bool ROCmMonitor::supports_domain(PowerDomain d) const {
    return d == PowerDomain::GPU || d == PowerDomain::GPU_MEMORY;
}
bool ROCmMonitor::supports_dvfs() const { return false; }
std::vector<PowerDomain> ROCmMonitor::available_domains() const {
    return {PowerDomain::GPU, PowerDomain::GPU_MEMORY};
}

// ---------------------------------------------------------------------------
// RAPLMonitor (Intel CPUs) - reads energy registers via MSR / perf events
// ---------------------------------------------------------------------------

struct RAPLMonitor::Impl {
    EnergyConfig config;
    EnergyStats  stats;
    bool         initialized = false;
    bool         running = false;
    double       last_package_j = 0.0;
    double       last_integral_j = 0.0;
    uint64_t     last_sample_ns = 0;
    uint64_t     first_sample_ns = 0;
    std::mutex   mutex;
};

RAPLMonitor::RAPLMonitor() : impl_(std::make_unique<Impl>()) {}
RAPLMonitor::~RAPLMonitor() = default;

int RAPLMonitor::initialize(const EnergyConfig& config) {
    impl_->config = config;
#if HPC_PLATFORM_LINUX && HPC_ARCH_X86_64
    // Probe for RAPL MSR availability. Requires root or the msr kernel module.
    if (access("/sys/class/powercap/intel-rapl/energy_uj", R_OK) == 0) {
        impl_->initialized = true;
        return static_cast<int>(ErrorCode::SUCCESS);
    }
    if (access("/dev/cpu/0/msr", R_OK) == 0) {
        impl_->initialized = true;
        return static_cast<int>(ErrorCode::SUCCESS);
    }
    fprintf(stderr, "[RAPL] energy counters not accessible "
                    "(need root or the 'msr' kernel module).\n");
#else
    fprintf(stderr, "[RAPL] only supported on Linux x86-64.\n");
#endif
    return static_cast<int>(ErrorCode::UNSUPPORTED_OPERATION);
}

int RAPLMonitor::start() {
    if (!impl_->initialized) return static_cast<int>(ErrorCode::NOT_INITIALIZED);
    impl_->running = true;
    impl_->last_sample_ns = now_ns();
    impl_->first_sample_ns = impl_->last_sample_ns;
    return static_cast<int>(ErrorCode::SUCCESS);
}

int RAPLMonitor::stop() {
    impl_->running = false;
    return static_cast<int>(ErrorCode::SUCCESS);
}

int RAPLMonitor::sample(EnergySample* sample) {
    if (!sample) return static_cast<int>(ErrorCode::INVALID_ARGUMENT);
    if (!impl_->initialized) return static_cast<int>(ErrorCode::UNSUPPORTED_OPERATION);

    std::lock_guard<std::mutex> lock(impl_->mutex);

    // Read the RAPL package energy counter (in microjoules since boot).
    double package_j = impl_->last_package_j;
    FILE* f = fopen("/sys/class/powercap/intel-rapl/energy_uj", "r");
    if (f) {
        long long uj = 0;
        if (fscanf(f, "%lld", &uj) == 1) {
            package_j = static_cast<double>(uj) * 1e-6;
        }
        fclose(f);
    }
    impl_->last_package_j = package_j;

    const uint64_t t = now_ns();
    const double dt = (t - impl_->last_sample_ns) / 1e9;
    const double power_w = dt > 0 ? (package_j - impl_->last_integral_j) / dt : 0.0;
    impl_->last_integral_j = package_j;
    impl_->last_sample_ns = t;

    sample->timestamp_ns = now_wall_ns();
    sample->energy_joules = package_j;
    sample->power_watts = std::max(power_w, 0.0);
    sample->domain = PowerDomain::PACKAGE;
    sample->device_id = -1;

    auto& st = impl_->stats;
    st.total_energy_joules = package_j;
    st.duration_ns = t - impl_->first_sample_ns;
    st.peak_power_watts = std::max(st.peak_power_watts, sample->power_watts);
    if (st.min_power_watts == 0.0) st.min_power_watts = sample->power_watts;
    else st.min_power_watts = std::min(st.min_power_watts, sample->power_watts);
    st.avg_power_watts = st.duration_ns ? st.total_energy_joules / (st.duration_ns / 1e9) : 0.0;
    ++st.num_samples;
    if (st.samples.size() < impl_->config.max_samples) st.samples.push_back(*sample);

    return static_cast<int>(ErrorCode::SUCCESS);
}

const EnergyStats& RAPLMonitor::stats() const { return impl_->stats; }
EnergyStats& RAPLMonitor::stats() { return impl_->stats; }
void RAPLMonitor::reset_stats() {
    impl_->stats.reset();
    impl_->last_integral_j = impl_->last_package_j;
    impl_->first_sample_ns = now_ns();
    impl_->last_sample_ns = impl_->first_sample_ns;
}
int RAPLMonitor::set_power_limit(uint32_t) { return static_cast<int>(ErrorCode::UNSUPPORTED_OPERATION); }
int RAPLMonitor::apply_dvfs_policy(const DVFSConfig&) { return static_cast<int>(ErrorCode::UNSUPPORTED_OPERATION); }
bool RAPLMonitor::supports_domain(PowerDomain d) const {
    return d == PowerDomain::PACKAGE || d == PowerDomain::CORE || d == PowerDomain::UNCORE;
}
bool RAPLMonitor::supports_dvfs() const { return false; }
std::vector<PowerDomain> RAPLMonitor::available_domains() const {
    return {PowerDomain::PACKAGE, PowerDomain::CORE, PowerDomain::UNCORE, PowerDomain::DRAM};
}

// ---------------------------------------------------------------------------
// CompositeMonitor
// ---------------------------------------------------------------------------

struct CompositeMonitor::Impl {
    std::vector<std::unique_ptr<EnergyMonitor>> monitors;
};

CompositeMonitor::CompositeMonitor() : impl_(std::make_unique<Impl>()) {}
CompositeMonitor::~CompositeMonitor() = default;

void CompositeMonitor::add_monitor(std::unique_ptr<EnergyMonitor> monitor) {
    impl_->monitors.push_back(std::move(monitor));
}

int CompositeMonitor::initialize(const EnergyConfig& config) {
    int rc = static_cast<int>(ErrorCode::SUCCESS);
    for (auto& m : impl_->monitors) {
        int r = m->initialize(config);
        if (r != static_cast<int>(ErrorCode::SUCCESS)) rc = r;
    }
    return rc;
}
int CompositeMonitor::start() {
    for (auto& m : impl_->monitors) m->start();
    return static_cast<int>(ErrorCode::SUCCESS);
}
int CompositeMonitor::stop() {
    for (auto& m : impl_->monitors) m->stop();
    return static_cast<int>(ErrorCode::SUCCESS);
}
int CompositeMonitor::sample(EnergySample* sample) {
    // Return the highest-power domain sample (GPU if present).
    int rc = static_cast<int>(ErrorCode::UNSUPPORTED_OPERATION);
    double best = -1.0;
    for (auto& m : impl_->monitors) {
        EnergySample s{};
        if (m->sample(&s) == static_cast<int>(ErrorCode::SUCCESS)) {
            if (s.power_watts > best) { best = s.power_watts; *sample = s; rc = 0; }
        }
    }
    return rc;
}
const EnergyStats& CompositeMonitor::stats() const {
    static EnergyStats s;
    return s;
}
EnergyStats& CompositeMonitor::stats() { static EnergyStats s; return s; }
void CompositeMonitor::reset_stats() { for (auto& m : impl_->monitors) m->reset_stats(); }
int CompositeMonitor::set_power_limit(uint32_t w) {
    for (auto& m : impl_->monitors) m->set_power_limit(w);
    return static_cast<int>(ErrorCode::SUCCESS);
}
int CompositeMonitor::set_sm_clock(uint32_t mhz) {
    for (auto& m : impl_->monitors) m->set_sm_clock(mhz);
    return static_cast<int>(ErrorCode::SUCCESS);
}
int CompositeMonitor::set_mem_clock(uint32_t mhz) {
    for (auto& m : impl_->monitors) m->set_mem_clock(mhz);
    return static_cast<int>(ErrorCode::SUCCESS);
}
int CompositeMonitor::apply_dvfs_policy(const DVFSConfig& c) {
    for (auto& m : impl_->monitors) m->apply_dvfs_policy(c);
    return static_cast<int>(ErrorCode::SUCCESS);
}
int CompositeMonitor::get_gpu_metrics(GPUMetrics* metrics) {
    for (auto& m : impl_->monitors) {
        if (m->get_gpu_metrics(metrics) == static_cast<int>(ErrorCode::SUCCESS)) {
            return static_cast<int>(ErrorCode::SUCCESS);
        }
    }
    return static_cast<int>(ErrorCode::UNSUPPORTED_OPERATION);
}
bool CompositeMonitor::supports_domain(PowerDomain d) const {
    for (const auto& m : impl_->monitors) if (m->supports_domain(d)) return true;
    return false;
}
bool CompositeMonitor::supports_dvfs() const {
    for (const auto& m : impl_->monitors) if (m->supports_dvfs()) return true;
    return false;
}
std::vector<PowerDomain> CompositeMonitor::available_domains() const {
    std::vector<PowerDomain> all;
    for (const auto& m : impl_->monitors) {
        auto d = m->available_domains();
        all.insert(all.end(), d.begin(), d.end());
    }
    std::sort(all.begin(), all.end());
    all.erase(std::unique(all.begin(), all.end()), all.end());
    return all;
}

// ---------------------------------------------------------------------------
// EnergyManager
// ---------------------------------------------------------------------------

EnergyManager::EnergyManager() = default;
EnergyManager::~EnergyManager() = default;

std::unique_ptr<EnergyMonitor> EnergyManager::create_monitor() {
#if HPC_HAVE_NVML_HEADER
    // Probe whether NVML actually initialises on this machine.
    EnergyConfig probe;
    auto nvml = std::make_unique<NVMLMonitor>();
    if (nvml->initialize(probe) == static_cast<int>(ErrorCode::SUCCESS)) {
        auto rapl = std::make_unique<RAPLMonitor>();
        EnergyConfig cfg;
        if (rapl->initialize(cfg) == static_cast<int>(ErrorCode::SUCCESS)) {
            auto composite = std::make_unique<CompositeMonitor>();
            composite->add_monitor(std::move(nvml));
            composite->add_monitor(std::move(rapl));
            return composite;
        }
        return nvml;
    }
#endif
    auto rapl = std::make_unique<RAPLMonitor>();
    EnergyConfig cfg;
    if (rapl->initialize(cfg) == static_cast<int>(ErrorCode::SUCCESS)) return rapl;

    // Fall back to a composite that simply reports nothing but stays usable.
    return std::make_unique<CompositeMonitor>();
}

int EnergyManager::initialize(const EnergyConfig& config) {
    config_ = config;
    monitor_ = create_monitor();
    cumulative_stats_.reset();

    EnergyConfig mcfg = config;
    mcfg.log_to_file = false;  // manager handles logging
    return monitor_->initialize(mcfg);
}

EnergyManager::ScopedMeasurement::ScopedMeasurement(EnergyManager& manager, const char* label)
    : manager_(manager), label_(label), start_time_ns_(now_ns()) {
    manager_.start_measurement(label);
}

EnergyManager::ScopedMeasurement::~ScopedMeasurement() {
    manager_.stop_measurement();
}

EnergyStats EnergyManager::ScopedMeasurement::get_stats() const {
    return manager_.current_stats();
}

int EnergyManager::start_measurement(const char* label) {
    if (!monitor_) return static_cast<int>(ErrorCode::NOT_INITIALIZED);
    monitor_->reset_stats();
    monitor_->start();
    measuring_ = true;
    current_label_ = label ? label : "";
    return static_cast<int>(ErrorCode::SUCCESS);
}

EnergyStats EnergyManager::stop_measurement() {
    if (!monitor_) return EnergyStats{};
    monitor_->stop();
    measuring_ = false;

    const EnergyStats& s = monitor_->stats();
    cumulative_stats_.total_energy_joules += s.total_energy_joules;
    cumulative_stats_.duration_ns += s.duration_ns;
    cumulative_stats_.num_samples += s.num_samples;
    cumulative_stats_.peak_power_watts =
        std::max(cumulative_stats_.peak_power_watts, s.peak_power_watts);

    EnergyStats result = s;
    result.duration_ns = s.duration_ns;
    return result;
}

const EnergyStats& EnergyManager::current_stats() const {
    return monitor_ ? monitor_->stats() : cumulative_stats_;
}

EnergyStats& EnergyManager::current_stats() {
    return monitor_ ? monitor_->stats() : cumulative_stats_;
}

int EnergyManager::set_power_limit(uint32_t watts) {
    return monitor_ ? monitor_->set_power_limit(watts) : static_cast<int>(ErrorCode::NOT_INITIALIZED);
}

int EnergyManager::apply_dvfs_policy(const DVFSConfig& config) {
    return monitor_ ? monitor_->apply_dvfs_policy(config) : static_cast<int>(ErrorCode::NOT_INITIALIZED);
}

int EnergyManager::get_gpu_metrics(GPUMetrics* metrics, int device_id) {
    (void)device_id;
    return monitor_ ? monitor_->get_gpu_metrics(metrics) : static_cast<int>(ErrorCode::NOT_INITIALIZED);
}

void EnergyManager::log_stats(const char* label) const {
    const EnergyStats& s = current_stats();
    fprintf(stderr,
            "[energy] %-24s energy=%.3f J  avg=%.2f W  peak=%.2f W  "
            "duration=%.3f s  EDP=%.3f\n",
            label ? label : "", s.total_energy_joules, s.avg_power_watts,
            s.peak_power_watts, s.duration_ns / 1e9, s.energy_delay_product());
}

void EnergyManager::write_csv(const char* filename) const {
    std::ofstream f(filename);
    if (!f.is_open()) return;
    f << "timestamp_ns,energy_j,power_w,domain,device_id,gpu_state,"
         "sm_clock_mhz,mem_clock_mhz,temp_c,power_limit_w\n";
    for (const auto& s : current_stats().samples) {
        f << s.timestamp_ns << ',' << s.energy_joules << ',' << s.power_watts << ','
          << power_domain_to_string(s.domain) << ',' << s.device_id << ','
          << gpu_power_state_to_string(s.gpu_state) << ',' << s.gpu_sm_clock_mhz << ','
          << s.gpu_mem_clock_mhz << ',' << s.gpu_temperature_c << ',' << s.gpu_power_limit_w << '\n';
    }
}

void EnergyManager::write_json(const char* filename) const {
    std::ofstream f(filename);
    if (!f.is_open()) return;
    const EnergyStats& s = current_stats();
    f << "{\n"
      << "  \"label\": \"" << current_label_ << "\",\n"
      << "  \"total_energy_joules\": " << s.total_energy_joules << ",\n"
      << "  \"avg_power_watts\": " << s.avg_power_watts << ",\n"
      << "  \"peak_power_watts\": " << s.peak_power_watts << ",\n"
      << "  \"duration_seconds\": " << s.duration_ns / 1e9 << ",\n"
      << "  \"energy_delay_product\": " << s.energy_delay_product() << ",\n"
      << "  \"num_samples\": " << s.num_samples << "\n"
      << "}\n";
}

double EnergyManager::estimate_carbon_grams(double carbon_intensity_g_per_kwh) const {
    const EnergyStats& s = current_stats();
    const double kwh = s.total_energy_joules / 3.6e6;
    return kwh * carbon_intensity_g_per_kwh;
}

// ---------------------------------------------------------------------------
// EnergyAwareScheduler
// ---------------------------------------------------------------------------

struct EnergyAwareScheduler::Impl {
    EnergyManager& energy;
    SchedulingStats stats;
};

EnergyAwareScheduler::EnergyAwareScheduler(EnergyManager& energy_manager)
    : impl_(std::make_unique<Impl>()) { impl_->energy = energy_manager; }

EnergyAwareScheduler::~EnergyAwareScheduler() = default;

int EnergyAwareScheduler::submit_task(const Task& task) {
    // Tasks are executed immediately in this reference implementation;
    // a production version would queue them for DVFS-aware dispatch.
    if (!task.work) return static_cast<int>(ErrorCode::INVALID_ARGUMENT);
    return static_cast<int>(ErrorCode::SUCCESS);
}

int EnergyAwareScheduler::execute_tasks(std::vector<Task>& tasks, DVFSPolicy global_policy) {
    // Group tasks by preferred policy so that DVFS state changes are amortised.
    std::stable_sort(tasks.begin(), tasks.end(),
                     [](const Task& a, const Task& b) {
                         return static_cast<int>(a.preferred_policy) <
                                static_cast<int>(b.preferred_policy);
                     });

    DVFSConfig dvfs;
    dvfs.policy = global_policy;

    EnergyStats total_energy;
    double total_time = 0.0;
    uint64_t completed = 0;

    for (const auto& t : tasks) {
        if (t.preferred_policy != global_policy) {
            dvfs.policy = t.preferred_policy;
            impl_->energy.apply_dvfs_policy(dvfs);
        }

        const uint64_t t0 = now_ns();
        EnergyManager::ScopedMeasurement scope(impl_->energy, "task");
        t.work();
        const uint64_t t1 = now_ns();

        EnergyStats ts = scope.get_stats();
        total_energy.total_energy_joules += ts.total_energy_joules;
        total_time += (t1 - t0) / 1e9;
        ++completed;
    }

    impl_->stats.tasks_completed = completed;
    impl_->stats.total_energy_joules = total_energy.total_energy_joules;
    impl_->stats.total_time_s = total_time;
    impl_->stats.avg_energy_per_task = completed ? impl_->stats.total_energy_joules / completed : 0.0;

    return static_cast<int>(ErrorCode::SUCCESS);
}

const EnergyAwareScheduler::SchedulingStats& EnergyAwareScheduler::scheduling_stats() const {
    return impl_->stats;
}

HPC_NAMESPACE_END