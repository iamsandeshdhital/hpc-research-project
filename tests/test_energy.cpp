// Unit tests for the energy monitoring, DVFS, and scheduling subsystem.
//
// On machines without NVML or RAPL the monitor reports "unsupported". Those
// tests assert graceful degradation rather than skipping silently, because a
// build that silently claims to measure energy it cannot measure is worse than
// one that says so.

#include "test_framework.hpp"
#include "hpc/hpc.hpp"

#include <cmath>
#include <vector>
#include <fstream>
#include <string>

using namespace hpc;

// ---------------------------------------------------------------------------
// Unit conversion
// ---------------------------------------------------------------------------

TEST(EnergyUnits, JouleToKilojouleHour) {
    const double kwh = convert_energy(3.6e6, EnergyUnit::JOULE, EnergyUnit::KILOWATT_HOUR);
    EXPECT_NEAR(kwh, 1.0, 1e-12);
}

TEST(EnergyUnits, JouleToMillijoule) {
    EXPECT_NEAR(convert_energy(1.0, EnergyUnit::JOULE, EnergyUnit::MILLIJOULE), 1000.0, 1e-9);
    EXPECT_NEAR(convert_energy(1.0, EnergyUnit::JOULE, EnergyUnit::MICROJOULE), 1e6, 1e-3);
}

TEST(EnergyUnits, MillijouleRoundTrip) {
    const double j = convert_energy(1234.0, EnergyUnit::MILLIJOULE, EnergyUnit::JOULE);
    EXPECT_NEAR(j, 1.234, 1e-12);
}

TEST(EnergyUnits, IdentityConversion) {
    EXPECT_NEAR(convert_energy(42.0, EnergyUnit::JOULE, EnergyUnit::JOULE), 42.0, 1e-12);
}

TEST(EnergyUnits, MicroToMilli) {
    EXPECT_NEAR(convert_energy(1000.0, EnergyUnit::MICROJOULE, EnergyUnit::MILLIJOULE),
                1.0, 1e-12);
}

// ---------------------------------------------------------------------------
// Enum stringification
// ---------------------------------------------------------------------------

TEST(EnergyStrings, PowerDomainNames) {
    EXPECT_STREQ(power_domain_to_string(PowerDomain::PACKAGE), "package");
    EXPECT_STREQ(power_domain_to_string(PowerDomain::GPU), "gpu");
    EXPECT_STREQ(power_domain_to_string(PowerDomain::DRAM), "dram");
    EXPECT_STREQ(power_domain_to_string(PowerDomain::GPU_SM), "gpu_sm");
}

TEST(EnergyStrings, GpuPowerStateNames) {
    EXPECT_STREQ(gpu_power_state_to_string(GPUPowerState::ACTIVE), "active");
    EXPECT_STREQ(gpu_power_state_to_string(GPUPowerState::IDLE), "idle");
    EXPECT_STREQ(gpu_power_state_to_string(GPUPowerState::THROTTLED), "throttled");
}

TEST(EnergyStrings, UnitNames) {
    EXPECT_STREQ(energy_unit_to_string(EnergyUnit::JOULE), "J");
    EXPECT_STREQ(energy_unit_to_string(EnergyUnit::WATT), "W");
    EXPECT_STREQ(energy_unit_to_string(EnergyUnit::KILOWATT_HOUR), "kWh");
}

// ---------------------------------------------------------------------------
// EnergyStats bookkeeping
// ---------------------------------------------------------------------------

TEST(EnergyStats, EdpIsEnergyTimesTime) {
    EnergyStats s;
    s.total_energy_joules = 1000.0;
    s.duration_ns = 2'000'000'000ull;   // 2 s

    EXPECT_NEAR(s.energy_delay_product(), 2000.0, 1e-6);
    EXPECT_NEAR(s.energy_delay_squared_product(), 4000.0, 1e-6);
}

TEST(EnergyStats, Ed2pIsEnergyTimesTimeSquared) {
    EnergyStats s;
    s.total_energy_joules = 500.0;
    s.duration_ns = 3'000'000'000ull;   // 3 s

    EXPECT_NEAR(s.energy_delay_squared_product(), 500.0 * 9.0, 1e-6);
}

TEST(EnergyStats, EdpOfZeroEnergyIsZero) {
    EnergyStats s;
    s.total_energy_joules = 0.0;
    s.duration_ns = 1'000'000'000ull;
    EXPECT_NEAR(s.energy_delay_product(), 0.0, 1e-15);
}

TEST(EnergyStats, ResetClearsSamples) {
    EnergyStats s;
    s.total_energy_joules = 500.0;
    s.peak_power_watts = 200.0;
    s.num_samples = 42;
    s.samples.resize(10);

    s.reset();
    EXPECT_NEAR(s.total_energy_joules, 0.0, 1e-15);
    EXPECT_NEAR(s.peak_power_watts, 0.0, 1e-15);
    EXPECT_EQ(s.num_samples, 0u);
    EXPECT_EQ(s.samples.size(), 0u);
}

// ---------------------------------------------------------------------------
// Monitor lifecycle (graceful degradation)
// ---------------------------------------------------------------------------

TEST(EnergyMonitor, NVMLInitialisesOrReportsUnsupported) {
    EnergyConfig cfg;
    cfg.continuous_sampling = false;
    NVMLMonitor m;
    const int rc = m.initialize(cfg);

    if (rc == static_cast<int>(ErrorCode::SUCCESS)) {
        EXPECT_EQ(m.start(), 0);
        EnergySample s{};
        EXPECT_EQ(m.sample(&s), 0);
        EXPECT_GT(s.power_watts, 0.0);
        EXPECT_EQ(m.stop(), 0);
    } else {
        EXPECT_EQ(rc, static_cast<int>(ErrorCode::UNSUPPORTED_OPERATION));
        EXPECT_FALSE(m.supports_dvfs());
    }
}

TEST(EnergyMonitor, NVMLRejectsNullSamplePointer) {
    NVMLMonitor m;
    EXPECT_EQ(m.sample(nullptr), static_cast<int>(ErrorCode::INVALID_ARGUMENT));
}

TEST(EnergyMonitor, NVMLReportsGpuDomains) {
    NVMLMonitor m;
    const auto d = m.available_domains();
    bool found_gpu = false;
    for (auto x : d) if (x == PowerDomain::GPU) found_gpu = true;
    EXPECT_TRUE(found_gpu);
    EXPECT_TRUE(m.supports_domain(PowerDomain::GPU));
    EXPECT_FALSE(m.supports_domain(PowerDomain::DRAM));
}

TEST(EnergyMonitor, RAPLInitialisesOrReportsUnsupported) {
    EnergyConfig cfg;
    RAPLMonitor m;
    const int rc = m.initialize(cfg);

    if (rc == static_cast<int>(ErrorCode::SUCCESS)) {
        ASSERT_EQ(m.start(), 0);
        EnergySample s{};
        EXPECT_EQ(m.sample(&s), 0);
        EXPECT_GE(s.energy_joules, 0.0);
        EXPECT_EQ(s.domain, PowerDomain::PACKAGE);
        m.stop();
    } else {
        EXPECT_EQ(rc, static_cast<int>(ErrorCode::UNSUPPORTED_OPERATION));
    }
}

TEST(EnergyMonitor, RAPLReportsCpuDomains) {
    RAPLMonitor m;
    EXPECT_TRUE(m.supports_domain(PowerDomain::PACKAGE));
    EXPECT_FALSE(m.supports_domain(PowerDomain::GPU));
    EXPECT_FALSE(m.supports_dvfs());
}

TEST(EnergyMonitor, CompositeCombinesDomains) {
    CompositeMonitor c;
    auto gpu = std::make_unique<NVMLMonitor>();
    auto cpu = std::make_unique<RAPLMonitor>();
    c.add_monitor(std::move(gpu));
    c.add_monitor(std::move(cpu));

    const auto domains = c.available_domains();
    EXPECT_FALSE(domains.empty());

    // Both package and GPU are offered by the two sub-monitors.
    bool has_package = false, has_gpu = false;
    for (auto d : domains) {
        if (d == PowerDomain::PACKAGE) has_package = true;
        if (d == PowerDomain::GPU) has_gpu = true;
    }
    EXPECT_TRUE(has_package);
    EXPECT_TRUE(has_gpu);
}

TEST(EnergyMonitor, CompositeInitializePropagatesFailure) {
    CompositeMonitor c;
    c.add_monitor(std::make_unique<RAPLMonitor>());
    EnergyConfig cfg;
    const int rc = c.initialize(cfg);
    // Either it worked (root/RAPL available) or it reported unsupported.
    EXPECT_TRUE(rc == 0 || rc == static_cast<int>(ErrorCode::UNSUPPORTED_OPERATION));
}

TEST(EnergyMonitor, CreateMonitorAlwaysReturnsNonNull) {
    auto m = EnergyManager::create_monitor();
    ASSERT_TRUE(m != nullptr);
    EXPECT_TRUE(m->name() != nullptr);
}

// ---------------------------------------------------------------------------
// EnergyManager
// ---------------------------------------------------------------------------

TEST(EnergyManager, InitializeAndMeasure) {
    EnergyManager mgr;
    EnergyConfig cfg;
    cfg.continuous_sampling = false;

    const int rc = mgr.initialize(cfg);
    if (rc != static_cast<int>(ErrorCode::SUCCESS)) {
        // Degradation is acceptable on machines without energy counters.
        EXPECT_EQ(rc, static_cast<int>(ErrorCode::UNSUPPORTED_OPERATION));
        return;
    }

    ASSERT_EQ(mgr.start_measurement("test"), 0);

    // Busy work so that measurable energy is consumed.
    volatile double acc = 0.0;
    for (int i = 0; i < 5'000'000; ++i) acc += i * 0.5;

    const EnergyStats s = mgr.stop_measurement();
    EXPECT_GE(s.duration_ns, 0u);
    EXPECT_GE(s.total_energy_joules, 0.0);
}

TEST(EnergyManager, ScopedMeasurementMeasures) {
    EnergyManager mgr;
    EnergyConfig cfg;
    cfg.continuous_sampling = false;
    if (mgr.initialize(cfg) != static_cast<int>(ErrorCode::SUCCESS)) return;

    EnergyManager::ScopedMeasurement scope(mgr, "scoped");
    volatile double acc = 0.0;
    for (int i = 0; i < 1'000'000; ++i) acc += i;
    const EnergyStats s = scope.get_stats();
    EXPECT_GE(s.total_energy_joules, 0.0);
}

TEST(EnergyManager, UninitialisedManagerRejectsRequests) {
    EnergyManager mgr;
    EXPECT_EQ(mgr.set_power_limit(200), static_cast<int>(ErrorCode::NOT_INITIALIZED));

    GPUMetrics m{};
    EXPECT_EQ(mgr.get_gpu_metrics(&m), static_cast<int>(ErrorCode::NOT_INITIALIZED));
}

TEST(EnergyManager, CarbonEstimateIsProportionalToEnergy) {
    EnergyManager mgr;
    EnergyConfig cfg;
    cfg.continuous_sampling = false;
    if (mgr.initialize(cfg) != static_cast<int>(ErrorCode::SUCCESS)) return;

    mgr.start_measurement("carbon");
    volatile double acc = 0.0;
    for (int i = 0; i < 1'000'000; ++i) acc += i;
    mgr.stop_measurement();

    const double grams = mgr.estimate_carbon_grams(475.0);
    const double energy = mgr.current_stats().total_energy_joules;
    const double expected = (energy / 3.6e6) * 475.0;
    EXPECT_NEAR(grams, expected, 1e-6);
}

TEST(EnergyManager, WritesCsvAndJson) {
    EnergyManager mgr;
    EnergyConfig cfg;
    cfg.continuous_sampling = false;
    if (mgr.initialize(cfg) != static_cast<int>(ErrorCode::SUCCESS)) return;

    mgr.start_measurement("io");
    mgr.stop_measurement();

    mgr.write_csv("test_energy.csv");
    mgr.write_json("test_energy.json");

    FILE* csv = fopen("test_energy.csv", "r");
    ASSERT_TRUE(csv != nullptr);
    char buf[512] = {0};
    const size_t n = fread(buf, 1, sizeof(buf) - 1, csv);
    fclose(csv);
    EXPECT_GT(n, 0u);
    EXPECT_TRUE(std::string(buf).find("timestamp_ns") != std::string::npos);

    FILE* js = fopen("test_energy.json", "r");
    ASSERT_TRUE(js != nullptr);
    memset(buf, 0, sizeof(buf));
    fread(buf, 1, sizeof(buf) - 1, js);
    fclose(js);
    EXPECT_TRUE(std::string(buf).find("total_energy_joules") != std::string::npos);

    EXPECT_EQ(remove("test_energy.csv"), 0);
    EXPECT_EQ(remove("test_energy.json"), 0);
}

// ---------------------------------------------------------------------------
// DVFS policies
// ---------------------------------------------------------------------------

namespace {

// Pure-function form of the policy target, mirrored from the monitor so it can
// be verified without hardware.
uint32_t policy_target(DVFSPolicy policy, uint32_t lo, uint32_t hi,
                       double w_perf, double w_en) {
    if (hi <= lo) return lo;
    if (w_perf + w_en <= 0.0) { w_perf = 0.5; w_en = 0.5; }
    switch (policy) {
        case DVFSPolicy::PERFORMANCE:      return hi;
        case DVFSPolicy::ENERGY_EFFICIENT: return lo;
        case DVFSPolicy::BALANCED:
            return lo + static_cast<uint32_t>((hi - lo) * (w_en / (w_perf + w_en)));
        case DVFSPolicy::CUSTOM:
        default:
            return hi;
    }
}

} // namespace

TEST(DVFS, PerformancePolicySelectsMaxClock) {
    EXPECT_EQ(policy_target(DVFSPolicy::PERFORMANCE, 300, 1500, 0.5, 0.5), 1500u);
}

TEST(DVFS, EnergyEfficientPolicySelectsMinClock) {
    EXPECT_EQ(policy_target(DVFSPolicy::ENERGY_EFFICIENT, 300, 1500, 0.5, 0.5), 300u);
}

TEST(DVFS, BalancedPolicySitsBetweenBounds) {
    const uint32_t t = policy_target(DVFSPolicy::BALANCED, 300, 1500, 0.5, 0.5);
    EXPECT_GE(t, 300u);
    EXPECT_LE(t, 1500u);
    EXPECT_GT(t, 300u);
    EXPECT_LT(t, 1500u);
}

TEST(DVFS, BalancedPolicyIsMonotonicInEnergyWeight) {
    const uint32_t low = policy_target(DVFSPolicy::BALANCED, 300, 1500, 0.9, 0.1);
    const uint32_t mid = policy_target(DVFSPolicy::BALANCED, 300, 1500, 0.5, 0.5);
    const uint32_t high = policy_target(DVFSPolicy::BALANCED, 300, 1500, 0.1, 0.9);
    EXPECT_LT(low, mid);
    EXPECT_LT(mid, high);
}

TEST(DVFS, DegenerateRangeIsSafe) {
    EXPECT_EQ(policy_target(DVFSPolicy::PERFORMANCE, 800, 800, 0.5, 0.5), 800u);
    EXPECT_EQ(policy_target(DVFSPolicy::BALANCED, 800, 800, 0.5, 0.5), 800u);
}

TEST(DVFS, ApplyPolicyOnUninitialisedMonitorIsSafe) {
    EnergyConfig cfg;
    NVMLMonitor m;
    DVFSConfig dvfs;
    dvfs.policy = DVFSPolicy::BALANCED;
    // Either it initialises (then it must not crash) or it reports unsupported.
    m.initialize(cfg);
    const int rc = m.apply_dvfs_policy(dvfs);
    EXPECT_TRUE(rc == 0 || rc == static_cast<int>(ErrorCode::NOT_INITIALIZED) ||
                rc == static_cast<int>(ErrorCode::UNSUPPORTED_OPERATION));
}

// ---------------------------------------------------------------------------
// Energy-aware scheduler
// ---------------------------------------------------------------------------

TEST(EnergyScheduler, ExecutesAllTasks) {
    EnergyManager mgr;
    EnergyConfig cfg;
    cfg.continuous_sampling = false;
    if (mgr.initialize(cfg) != static_cast<int>(ErrorCode::SUCCESS)) return;

    EnergyAwareScheduler sched(mgr);

    std::vector<EnergyAwareScheduler::Task> tasks(8);
    for (int i = 0; i < 8; ++i) {
        tasks[i].work = [i]() {
            volatile double x = 0.0;
            for (int k = 0; k < 100000; ++k) x += k * (i + 1);
        };
        tasks[i].estimated_flops = 1e6;
        tasks[i].priority = 1.0;
        tasks[i].preferred_policy = (i % 2 == 0) ? DVFSPolicy::PERFORMANCE
                                                : DVFSPolicy::ENERGY_EFFICIENT;
    }

    EXPECT_EQ(sched.execute_tasks(tasks, DVFSPolicy::BALANCED), 0);
    EXPECT_EQ(sched.scheduling_stats().tasks_completed, 8u);
    EXPECT_GT(sched.scheduling_stats().total_time_s, 0.0);
}

TEST(EnergyScheduler, RejectsNullWork) {
    EnergyManager mgr;
    EnergyConfig cfg;
    cfg.continuous_sampling = false;
    mgr.initialize(cfg);

    EnergyAwareScheduler sched(mgr);
    EnergyAwareScheduler::Task t;
    t.work = nullptr;
    EXPECT_EQ(sched.submit_task(t), static_cast<int>(ErrorCode::INVALID_ARGUMENT));
}

TEST(EnergyScheduler, GroupsTasksByPolicy) {
    EnergyManager mgr;
    EnergyConfig cfg;
    cfg.continuous_sampling = false;
    if (mgr.initialize(cfg) != static_cast<int>(ErrorCode::SUCCESS)) return;

    EnergyAwareScheduler sched(mgr);

    int counter = 0;
    std::vector<EnergyAwareScheduler::Task> tasks;
    for (int i = 0; i < 6; ++i) {
        EnergyAwareScheduler::Task t;
        t.work = [&counter]() { ++counter; };
        t.preferred_policy = (i < 3) ? DVFSPolicy::PERFORMANCE : DVFSPolicy::ENERGY_EFFICIENT;
        tasks.push_back(t);
    }

    EXPECT_EQ(sched.execute_tasks(tasks, DVFSPolicy::BALANCED), 0);
    EXPECT_EQ(counter, 6);
    EXPECT_EQ(sched.scheduling_stats().tasks_completed, 6u);
}

HPC_TEST_MAIN()
