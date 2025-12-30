//! Multi-Tenancy Benchmarks
//!
//! Benchmarks for tenant management and data isolation operations.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use pandrs::dataframe::DataFrame;
use pandrs::multitenancy::{Permission, ResourceQuota, TenantConfig, TenantManager};
use pandrs::series::Series;

/// Create a test DataFrame with specified size
fn create_test_df(n_rows: usize, n_cols: usize) -> DataFrame {
    let mut df = DataFrame::new();

    let mut rng_state: u64 = 42;
    let rand_f64 = |state: &mut u64| -> f64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (*state >> 33) as f64 / (u32::MAX as f64)
    };

    for c in 0..n_cols {
        let values: Vec<f64> = (0..n_rows).map(|_| rand_f64(&mut rng_state)).collect();
        let series = Series::new(values, Some(format!("col_{}", c))).unwrap();
        df.add_column(format!("col_{}", c), series).unwrap();
    }

    df
}

/// Create a tenant manager with multiple tenants
fn create_manager_with_tenants(n_tenants: usize) -> TenantManager {
    let mut manager = TenantManager::new();

    for i in 0..n_tenants {
        let config = TenantConfig::new(format!("tenant_{}", i))
            .with_permission(Permission::Read)
            .with_permission(Permission::Write)
            .with_permission(Permission::Create)
            .with_permission(Permission::Delete)
            .with_quota(ResourceQuota::unlimited());
        manager.register_tenant(config).unwrap();
    }

    manager
}

fn bench_tenant_registration(c: &mut Criterion) {
    let mut group = c.benchmark_group("Tenant Registration");

    for n_tenants in [10, 50, 100, 500].iter() {
        group.bench_with_input(
            BenchmarkId::new("register", n_tenants),
            n_tenants,
            |b, &n| {
                b.iter(|| {
                    let mut manager = TenantManager::new();
                    for i in 0..n {
                        let config = TenantConfig::default_rw(format!("tenant_{}", i));
                        manager
                            .register_tenant(std::hint::black_box(config))
                            .unwrap();
                    }
                });
            },
        );
    }

    group.finish();
}

fn bench_data_storage(c: &mut Criterion) {
    let mut group = c.benchmark_group("Data Storage");

    for n_rows in [100, 1000, 10000].iter() {
        let df = create_test_df(*n_rows, 10);

        group.bench_with_input(BenchmarkId::new("store", n_rows), &df, |b, df| {
            b.iter(|| {
                let mut manager = TenantManager::new();
                let config =
                    TenantConfig::default_rw("test_tenant").with_quota(ResourceQuota::unlimited());
                manager.register_tenant(config).unwrap();
                manager
                    .store_dataframe("test_tenant", "data", std::hint::black_box(df.clone()))
                    .unwrap();
            });
        });
    }

    group.finish();
}

fn bench_data_retrieval(c: &mut Criterion) {
    let mut group = c.benchmark_group("Data Retrieval");

    for n_rows in [100, 1000, 10000].iter() {
        let df = create_test_df(*n_rows, 10);
        let mut manager = TenantManager::new();
        let config = TenantConfig::default_rw("test_tenant").with_quota(ResourceQuota::unlimited());
        manager.register_tenant(config).unwrap();
        manager.store_dataframe("test_tenant", "data", df).unwrap();

        group.bench_with_input(BenchmarkId::new("get", n_rows), &manager, |b, manager| {
            // Need mutable reference for get_dataframe
            let mut manager_clone = TenantManager::new();
            let config =
                TenantConfig::default_rw("test_tenant").with_quota(ResourceQuota::unlimited());
            manager_clone.register_tenant(config).unwrap();
            let df = create_test_df(*n_rows, 10);
            manager_clone
                .store_dataframe("test_tenant", "data", df)
                .unwrap();

            b.iter(|| {
                manager_clone
                    .get_dataframe(
                        std::hint::black_box("test_tenant"),
                        std::hint::black_box("data"),
                    )
                    .unwrap();
            });
        });
    }

    group.finish();
}

fn bench_permission_check(c: &mut Criterion) {
    let mut group = c.benchmark_group("Permission Check");

    let manager = create_manager_with_tenants(100);

    group.bench_function("has_permission", |b| {
        b.iter(|| {
            manager.has_permission(
                std::hint::black_box("tenant_50"),
                std::hint::black_box(Permission::Read),
            )
        });
    });

    group.bench_function("get_tenant", |b| {
        b.iter(|| manager.get_tenant(std::hint::black_box("tenant_50")));
    });

    group.finish();
}

fn bench_multi_tenant_isolation(c: &mut Criterion) {
    let mut group = c.benchmark_group("Multi-Tenant Isolation");
    group.sample_size(10);

    for n_tenants in [10, 50, 100].iter() {
        group.bench_with_input(
            BenchmarkId::new("store_per_tenant", n_tenants),
            n_tenants,
            |b, &n| {
                b.iter(|| {
                    let mut manager = create_manager_with_tenants(n);
                    let df = create_test_df(100, 5);

                    for i in 0..n {
                        manager
                            .store_dataframe(&format!("tenant_{}", i), "data", df.clone())
                            .unwrap();
                    }
                });
            },
        );
    }

    group.finish();
}

fn bench_quota_checking(c: &mut Criterion) {
    let mut group = c.benchmark_group("Quota Checking");

    // Manager with quotas enabled
    let mut manager = TenantManager::new().with_quota_enforcement(true);
    let config = TenantConfig::default_rw("limited_tenant")
        .with_max_rows(1_000_000)
        .with_max_datasets(1000);
    manager.register_tenant(config).unwrap();

    let df = create_test_df(100, 10);

    group.bench_function("with_quota_check", |b| {
        let mut m = TenantManager::new().with_quota_enforcement(true);
        let config = TenantConfig::default_rw("t")
            .with_max_rows(1_000_000)
            .with_permission(Permission::Delete);
        m.register_tenant(config).unwrap();

        b.iter(|| {
            m.store_dataframe("t", "data", std::hint::black_box(df.clone()))
                .unwrap();
            m.delete_dataframe("t", "data").unwrap();
        });
    });

    group.bench_function("without_quota_check", |b| {
        let mut m = TenantManager::new().with_quota_enforcement(false);
        let config = TenantConfig::default_rw("t").with_permission(Permission::Delete);
        m.register_tenant(config).unwrap();

        b.iter(|| {
            m.store_dataframe("t", "data", std::hint::black_box(df.clone()))
                .unwrap();
            m.delete_dataframe("t", "data").unwrap();
        });
    });

    group.finish();
}

fn bench_audit_logging(c: &mut Criterion) {
    let mut group = c.benchmark_group("Audit Logging");

    group.bench_function("with_audit", |b| {
        let mut manager = TenantManager::new().with_max_audit_entries(10000);
        let config = TenantConfig::default_rw("t")
            .with_permission(Permission::Delete)
            .with_quota(ResourceQuota::unlimited());
        manager.register_tenant(config).unwrap();
        let df = create_test_df(100, 5);

        b.iter(|| {
            manager
                .store_dataframe("t", "data", std::hint::black_box(df.clone()))
                .unwrap();
            let _ = manager.get_dataframe("t", "data");
            manager.delete_dataframe("t", "data").unwrap();
        });
    });

    group.bench_function("audit_retrieval", |b| {
        let mut manager = TenantManager::new();
        let config = TenantConfig::default_rw("t")
            .with_permission(Permission::Delete)
            .with_quota(ResourceQuota::unlimited());
        manager.register_tenant(config).unwrap();
        let df = create_test_df(100, 5);

        // Generate some audit entries
        for i in 0..100 {
            manager
                .store_dataframe("t", &format!("data_{}", i), df.clone())
                .unwrap();
        }

        b.iter(|| {
            let audit = manager.get_audit_log(std::hint::black_box(Some("t")));
            std::hint::black_box(audit.len())
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_tenant_registration,
    bench_data_storage,
    bench_data_retrieval,
    bench_permission_check,
    bench_multi_tenant_isolation,
    bench_quota_checking,
    bench_audit_logging,
);

criterion_main!(benches);
