use std::path::Path;

use criterion::{black_box, criterion_group, criterion_main, Criterion, SamplingMode};

use Pulsecure::adapters::tfhe::TfheAdapter;
use Pulsecure::domain::{PatientData, PatientFeatures};
use Pulsecure::ports::FheEngine;

fn adapter() -> TfheAdapter {
    std::env::set_var("PULSECURE_ALLOW_UNSIGNED_MODELS", "true");
    let mut adapter = TfheAdapter::new();
    adapter.load_model(Path::new("models")).expect("load model");
    adapter
}

fn sample_patient() -> PatientData {
    PatientData::new(PatientFeatures {
        age: 55.0,
        hypertension: 1.0,
        sys_bp: 142.0,
        smoking: 1.0,
        hdl_chol: 45.0,
        creatinine: 1.1,
        waist_circ: 102.0,
        diabetes: 0.0,
        hba1c: 5.9,
        ..Default::default()
    })
}

fn fhe_pipeline_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("fhe_pipeline");
    group.sample_size(10);
    group.sampling_mode(SamplingMode::Flat);

    group.bench_function("keygen", |b| {
        b.iter(|| {
            let adapter = adapter();
            black_box(adapter.generate_keys().expect("keygen"));
        });
    });

    let adapter = adapter();
    let keys = adapter.generate_keys().expect("keygen");
    let patient = sample_patient();

    group.bench_function("encrypt_9_features", |b| {
        b.iter(|| {
            black_box(
                adapter
                    .encrypt(black_box(&patient), black_box(&keys.client))
                    .expect("encrypt"),
            );
        });
    });

    let encrypted = adapter.encrypt(&patient, &keys.client).expect("encrypt");

    group.bench_function("ciphertext_size_bytes", |b| {
        b.iter(|| black_box(encrypted.size_bytes()));
    });

    group.bench_function("compute_linear_fhe", |b| {
        b.iter(|| {
            black_box(
                adapter
                    .compute(black_box(&encrypted), black_box(&keys.server))
                    .expect("compute"),
            );
        });
    });

    let encrypted_result = adapter.compute(&encrypted, &keys.server).expect("compute");

    group.bench_function("decrypt_calibrate_threshold", |b| {
        b.iter(|| {
            black_box(
                adapter
                    .decrypt(black_box(&encrypted_result), black_box(&keys.client))
                    .expect("decrypt"),
            );
        });
    });

    group.bench_function("total_inference_without_keygen", |b| {
        b.iter(|| {
            let encrypted = adapter
                .encrypt(black_box(&patient), black_box(&keys.client))
                .expect("encrypt");
            let encrypted_result = adapter
                .compute(black_box(&encrypted), black_box(&keys.server))
                .expect("compute");
            black_box(
                adapter
                    .decrypt(black_box(&encrypted_result), black_box(&keys.client))
                    .expect("decrypt"),
            );
        });
    });

    group.finish();
}

criterion_group!(benches, fhe_pipeline_benchmarks);
criterion_main!(benches);
