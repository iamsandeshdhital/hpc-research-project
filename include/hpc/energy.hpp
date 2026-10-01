#pragma once

#include "config.hpp"
#include <cstdint>
#include <vector>
#include <string>
#include <chrono>
#include <functional>
#include <memory>

HPC_NAMESPACE_BEGIN

// Energy unit types
enum class EnergyUnit : uint8_t {
    JOULE = 0,
    WATT = 1,
    KILOWATT_HOUR = 2,
    MILLIJOULE = 3,
    MICROJOULE = 4
};

// Power domain types
enum class PowerDomain : uint8_t {
    PACKAGE = 0,        // CPU package
    CORE = 1,           // CPU cores
    UNCORE = 2,         // CPU uncore (cache, memory controller)
    DRAM = 3,           // DRAM
    GPU = 4,            // GPU
    GPU_MEMORY = 5,     // GPU memory
    GPU_SM = 6,         // GPU streaming multiprocessors
    PLATFORM = 7,       // Entire platform
    CUSTOM = 8
};

// GPU power state
enum class GPUPowerState : uint8_t {
    UNKNOWN = 0,
    ACTIVE = 1,
    IDLE = 2,
    THROTTLED = 3,
    POWER_GATED = 4
};

// Energy measurement sample
struct HPC_API EnergySample {
    uint64_t timestamp_ns = 0;
    double energy_joules = 0.0;
    double power_watts = 0.0;
    PowerDomain domain = PowerDomain::PACKAGE;
    int device_id = -1;
    GPUPowerState gpu_state = GPUPowerState::UNKNOWN;
    uint32_t gpu_sm_clock_mhz = 0;
    uint32_t gpu_mem_clock_mhz = 0;
    uint32_t gpu_temperature_c = 0;
    uint32_t gpu_power_limit_w = 0;
    uint32_t gpu_current_power_w = 0;
};

// Energy measurement configuration
struct HPC_API EnergyConfig {
    // Sampling
    uint64_t sample_interval_ns = 1'000'000;  // 1 ms default
    bool continuous_sampling = true;
    size_t max_samples = 100000;

    // Domains to monitor
    bool monitor_cpu_package = true;
    bool monitor_cpu_cores = false;
    bool monitor_cpu_uncore = false;
    bool monitor_dram = false;
    bool monitor_gpu = true;
    bool monitor_gpu_memory = false;
    bool monitor_gpu_sm = false;

    // GPU-specific
    int gpu_device_id = 0;
    bool enable_dvfs_control = false;
    uint32_t target_power_limit_w = 0;  // 0 = unlimited
    uint32_t min_sm_clock_mhz = 0;
    uint32_t max_sm_clock_mhz = 0;
    uint32_t min_mem_clock_mhz = 0;
    uint32_t max_mem_clock_mhz = 0;

    // Output
    bool log_to_file = false;
    std::string log_file_path = "energy_log.csv";
    bool log_csv = true;
    bool log_json = false;

    // Alerts
    bool enable_power_alerts = false;
    double power_threshold_w = 0.0;
    double energy_threshold_j = 0.0;
    std::function<void(const EnergySample&)> alert_callback = nullptr;
};

// Energy statistics
struct HPC_API EnergyStats {
    double total_energy_joules = 0.0;
    double avg_power_watts = 0.0;
    double peak_power_watts = 0.0;
    double min_power_watts = 0.0;
    double energy_per_flop = 0.0;
    double energy_per_byte = 0.0;
    uint64_t num_samples = 0;
    uint64_t duration_ns = 0;
    std::vector<EnergySample> samples;

    void reset() {
        *this = EnergyStats{};
    }

    double energy_delay_product() const {
        return total_energy_joules * (duration_ns / 1e9);
    }

    double energy_delay_squared_product() const {
        double duration_s = duration_ns / 1e9;
        return total_energy_joules * duration_s * duration_s;
    }
};

// DVFS policy types
enum class DVFSPolicy : uint8_t {
    PERFORMANCE = 0,      // Maximum performance
    BALANCED = 1,         // Balance performance and energy
    ENERGY_EFFICIENT = 2, // Minimum energy
    CUSTOM = 3            // User-defined policy
};

// DVFS configuration
struct HPC_API DVFSConfig {
    DVFSPolicy policy = DVFSPolicy::BALANCED;
    uint32_t target_power_w = 0;
    double performance_weight = 0.5;
    double energy_weight = 0.5;
    uint32_t min_frequency_mhz = 0;
    uint32_t max_frequency_mhz = 0;
    uint32_t step_frequency_mhz = 50;
    uint64_t evaluation_interval_ns = 10'000'000;  // 10 ms
    bool enable_predictive = false;
};

// GPU metrics
struct HPC_API GPUMetrics {
    uint32_t device_id = 0;
    std::string name;
    uint32_t sm_count = 0;
    uint32_t sm_clock_mhz = 0;
    uint32_t mem_clock_mhz = 0;
    uint32_t temperature_c = 0;
    uint32_t power_w = 0;
    uint32_t power_limit_w = 0;
    uint32_t fan_speed_percent = 0;
    uint64_t memory_used_bytes = 0;
    uint64_t memory_total_bytes = 0;
    double sm_utilization = 0.0;
    double mem_utilization = 0.0;
    double encoder_utilization = 0.0;
    double decoder_utilization = 0.0;
    GPUPowerState power_state = GPUPowerState::UNKNOWN;
    uint32_t pcie_throughput_mbps = 0;
    uint32_t nvlink_throughput_mbps = 0;
};

// Energy monitor interface
class HPC_API EnergyMonitor {
public:
    virtual ~EnergyMonitor() = default;

    // Initialize monitoring
    virtual int initialize(const EnergyConfig& config) = 0;

    // Start/stop monitoring
    virtual int start() = 0;
    virtual int stop() = 0;

    // Get current sample
    virtual int sample(EnergySample* sample) = 0;

    // Get statistics
    virtual const EnergyStats& stats() const = 0;
    virtual EnergyStats& stats() = 0;

    // Reset statistics
    virtual void reset_stats() = 0;

    // DVFS control
    virtual int set_power_limit(uint32_t watts) = 0;
    virtual int set_sm_clock(uint32_t mhz) = 0;
    virtual int set_mem_clock(uint32_t mhz) = 0;
    virtual int apply_dvfs_policy(const DVFSConfig& config) = 0;

    // Get GPU metrics
    virtual int get_gpu_metrics(GPUMetrics* metrics) = 0;

    // Check capabilities
    virtual bool supports_domain(PowerDomain domain) const = 0;
    virtual bool supports_dvfs() const = 0;
    virtual std::vector<PowerDomain> available_domains() const = 0;

    // Get monitor name
    virtual const char* name() const = 0;
};

// NVML-based energy monitor (NVIDIA GPUs)
class HPC_API NVMLMonitor : public EnergyMonitor {
public:
    NVMLMonitor();
    ~NVMLMonitor() override;

    int initialize(const EnergyConfig& config) override;
    int start() override;
    int stop() override;
    int sample(EnergySample* sample) override;
    const EnergyStats& stats() const override;
    EnergyStats& stats() override;
    void reset_stats() override;
    int set_power_limit(uint32_t watts) override;
    int set_sm_clock(uint32_t mhz) override;
    int set_mem_clock(uint32_t mhz) override;
    int apply_dvfs_policy(const DVFSConfig& config) override;
    int get_gpu_metrics(GPUMetrics* metrics) override;
    bool supports_domain(PowerDomain domain) const override;
    bool supports_dvfs() const override;
    std::vector<PowerDomain> available_domains() const override;
    const char* name() const override { return "NVML"; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ROCm-SMI based energy monitor (AMD GPUs)
class HPC_API ROCmMonitor : public EnergyMonitor {
public:
    ROCmMonitor();
    ~ROCmMonitor() override;

    int initialize(const EnergyConfig& config) override;
    int start() override;
    int stop() override;
    int sample(EnergySample* sample) override;
    const EnergyStats& stats() const override;
    EnergyStats& stats() override;
    void reset_stats() override;
    int set_power_limit(uint32_t watts) override;
    int set_sm_clock(uint32_t mhz) override;
    int set_mem_clock(uint32_t mhz) override;
    int apply_dvfs_policy(const DVFSConfig& config) override;
    int get_gpu_metrics(GPUMetrics* metrics) override;
    bool supports_domain(PowerDomain domain) const override;
    bool supports_dvfs() const override;
    std::vector<PowerDomain> available_domains() const override;
    const char* name() const override { return "ROCm-SMI"; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// RAPL-based energy monitor (Intel CPUs)
class HPC_API RAPLMonitor : public EnergyMonitor {
public:
    RAPLMonitor();
    ~RAPLMonitor() override;

    int initialize(const EnergyConfig& config) override;
    int start() override;
    int stop() override;
    int sample(EnergySample* sample) override;
    const EnergyStats& stats() const override;
    EnergyStats& stats() override;
    void reset_stats() override;
    int set_power_limit(uint32_t watts) override;
    int set_sm_clock(uint32_t mhz) override { return -1; }
    int set_mem_clock(uint32_t mhz) override { return -1; }
    int apply_dvfs_policy(const DVFSConfig& config) override;
    int get_gpu_metrics(GPUMetrics* metrics) override { return -1; }
    bool supports_domain(PowerDomain domain) const override;
    bool supports_dvfs() const override;
    std::vector<PowerDomain> available_domains() const override;
    const char* name() const override { return "RAPL"; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Composite energy monitor (combines multiple monitors)
class HPC_API CompositeMonitor : public EnergyMonitor {
public:
    CompositeMonitor();
    ~CompositeMonitor() override;

    int initialize(const EnergyConfig& config) override;
    int start() override;
    int stop() override;
    int sample(EnergySample* sample) override;
    const EnergyStats& stats() const override;
    EnergyStats& stats() override;
    void reset_stats() override;
    int set_power_limit(uint32_t watts) override;
    int set_sm_clock(uint32_t mhz) override;
    int set_mem_clock(uint32_t mhz) override;
    int apply_dvfs_policy(const DVFSConfig& config) override;
    int get_gpu_metrics(GPUMetrics* metrics) override;
    bool supports_domain(PowerDomain domain) const override;
    bool supports_dvfs() const override;
    std::vector<PowerDomain> available_domains() const override;
    const char* name() const override { return "Composite"; }

    // Add sub-monitor
    void add_monitor(std::unique_ptr<EnergyMonitor> monitor);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Energy manager - high-level interface
class HPC_API EnergyManager {
public:
    EnergyManager();
    ~EnergyManager();

    // Initialize with configuration
    int initialize(const EnergyConfig& config = EnergyConfig{});

    // Create appropriate monitor based on hardware
    static std::unique_ptr<EnergyMonitor> create_monitor();

    // Scoped energy measurement
    class ScopedMeasurement {
    public:
        ScopedMeasurement(EnergyManager& manager, const char* label);
        ~ScopedMeasurement();

        EnergyStats get_stats() const;

    private:
        EnergyManager& manager_;
        const char* label_;
        EnergyStats start_stats_;
        uint64_t start_time_ns_;
    };

    // Start/stop measurement
    int start_measurement(const char* label = "");
    EnergyStats stop_measurement();

    // Get current stats
    const EnergyStats& current_stats() const;
    EnergyStats& current_stats();

    // DVFS control
    int set_power_limit(uint32_t watts);
    int apply_dvfs_policy(const DVFSConfig& config);

    // GPU metrics
    int get_gpu_metrics(GPUMetrics* metrics, int device_id = 0);

    // Get monitor
    EnergyMonitor* monitor() { return monitor_.get(); }
    const EnergyMonitor* monitor() const { return monitor_.get(); }

    // Logging
    void log_stats(const char* label = "") const;
    void write_csv(const char* filename) const;
    void write_json(const char* filename) const;

    // Carbon tracking (if available)
    double estimate_carbon_grams(double carbon_intensity_g_per_kwh = 475.0) const;

private:
    std::unique_ptr<EnergyMonitor> monitor_;
    EnergyConfig config_;
    EnergyStats cumulative_stats_;
    bool measuring_ = false;
    std::string current_label_;
};

// Energy-aware scheduler
class HPC_API EnergyAwareScheduler {
public:
    struct Task {
        std::function<void()> work;
        double estimated_flops = 0;
        double estimated_bytes = 0;
        double priority = 1.0;
        DVFSPolicy preferred_policy = DVFSPolicy::BALANCED;
        int preferred_gpu = -1;
    };

    EnergyAwareScheduler(EnergyManager& energy_manager);
    ~EnergyAwareScheduler();

    // Submit task
    int submit_task(const Task& task);

    // Execute tasks with energy awareness
    int execute_tasks(std::vector<Task>& tasks, DVFSPolicy global_policy = DVFSPolicy::BALANCED);

    // Get scheduling stats
    struct SchedulingStats {
        uint64_t tasks_completed = 0;
        double total_energy_joules = 0.0;
        double total_time_s = 0.0;
        double avg_energy_per_task = 0.0;
    };
    const SchedulingStats& scheduling_stats() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Utility functions
HPC_API double convert_energy(double value, EnergyUnit from, EnergyUnit to);
HPC_API double convert_power(double value, EnergyUnit from, EnergyUnit to);
HPC_API std::string energy_unit_to_string(EnergyUnit unit);
HPC_API std::string power_domain_to_string(PowerDomain domain);
HPC_API std::string gpu_power_state_to_string(GPUPowerState state);

// Energy profiling macros
#define HPC_ENERGY_SCOPE(manager, label) hpc::EnergyManager::ScopedMeasurement _energy_scope(manager, label)

HPC_NAMESPACE_END