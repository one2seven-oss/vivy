use rand::Rng;
use std::collections::HashMap;
use std::hint::black_box;
use std::time::{Duration, Instant};
use tempfile::tempdir;
use vivy_core::concurrent::VivyIndex;
use vivy_core::distance::Metric;
use vivy_core::flat::FlatIndex;
use vivy_memory::*;

fn main() {
    println!("==================================================");
    println!("      VIVY ARCHITECTURE BENCHMARK SUITE          ");
    println!("==================================================");

    println!("\n--- Part 1: Core Vector Engine Benchmarks ---");
    println!("\n[64 Dimensions]");
    run_vector_engine_benchmark(64, 2000, 50, 10);

    println!("\n[512 Dimensions]");
    run_vector_engine_benchmark(512, 1000, 50, 10);

    println!("\n--- Part 2: Long-Term Memory (LTM) Store Benchmarks ---");
    run_memory_store_benchmark(128, 500, 50);

    println!("\n==================================================");
    println!(" Benchmark suite completed successfully.");
    println!("==================================================");
}

fn run_vector_engine_benchmark(dims: usize, record_count: usize, query_count: usize, top_k: usize) {
    let (dataset, queries) = generate_vector_dataset(dims, record_count, query_count);

    let flat_start = Instant::now();
    let mut flat = FlatIndex::new(Metric::L2);
    for (idx, vec) in dataset.iter().enumerate() {
        flat.insert((idx + 1) as u64, vec.clone());
    }
    let flat_build_duration = flat_start.elapsed();
    println!("  Flat index build ({record_count} items): {:?}", flat_build_duration);

    let ground_truth: Vec<Vec<u64>> = queries
        .iter()
        .map(|query_vec| flat.search(query_vec, top_k).into_iter().map(|(id, _)| id).collect())
        .collect();
    drop(flat);

    let hnsw_build_start = Instant::now();
    let index = VivyIndex::new(dims, Metric::L2, None::<&str>, None::<&str>).unwrap();
    for vec in &dataset {
        black_box(index.insert(black_box(vec.clone()))).unwrap();
    }
    let hnsw_build_duration = hnsw_build_start.elapsed();
    let build_qps = record_count as f64 / hnsw_build_duration.as_secs_f64();
    println!(
        "  HNSW index build ({record_count} items): {:?} ({:.0} insertions/sec)",
        hnsw_build_duration, build_qps
    );

    // Warm-up query loop
    for query_vec in queries.iter().take(5) {
        black_box(index.search(query_vec, top_k)).unwrap();
    }

    let search_start = Instant::now();
    let mut total_hits = 0usize;
    let mut total_query_duration = Duration::ZERO;
    for (query_idx, query_vec) in queries.iter().enumerate() {
        let single_query_start = Instant::now();
        let search_results = index.search(query_vec, top_k).unwrap();
        total_query_duration += single_query_start.elapsed();

        for ground_truth_id in &ground_truth[query_idx] {
            if search_results.iter().any(|(id, _)| id == ground_truth_id) {
                total_hits += 1;
            }
        }
    }
    let total_search_duration = search_start.elapsed();
    let total_expected_hits = query_count * top_k;
    let recall_score = total_hits as f64 / total_expected_hits as f64;
    let search_qps = query_count as f64 / total_search_duration.as_secs_f64();
    let mean_latency_ms = (total_query_duration.as_secs_f64() / query_count as f64) * 1000.0;

    println!("  Recall@{top_k}: {:.4}", recall_score);
    println!("  Search throughput: {:.1} QPS", search_qps);
    println!("  Mean query latency: {:.3} ms", mean_latency_ms);
}

fn run_memory_store_benchmark(dims: usize, initial_records: usize, test_queries: usize) {
    let temp_directory = tempdir().unwrap();
    let config = MemoryConfig::builder(temp_directory.path())
        .dimensions(dims)
        .embedding_model("benchmark-model")
        .build()
        .unwrap();

    // 1. Cold Start Latency (Fresh DB creation)
    let cold_start_timer = Instant::now();
    let store = MemoryStore::open(config.clone()).unwrap();
    let cold_start_latency = cold_start_timer.elapsed();
    println!("  Cold start latency (fresh database): {:?}", cold_start_latency);

    let scope = MemoryScope::new("bench-tenant", "bench-namespace").unwrap();
    let mut random_generator = rand::rng();

    // 2. Write Throughput Benchmark
    let write_timer = Instant::now();
    let mut inserted_ids = Vec::with_capacity(initial_records);
    for idx in 0..initial_records {
        let embedding: Vec<f32> = (0..dims).map(|_| random_generator.random()).collect();
        let content = format!("Benchmark record #{} with domain search text item", idx);

        let mem_id = store
            .remember(RememberRequest {
                operation_id: None,
                scope: scope.clone(),
                content,
                embedding,
                kind: MemoryKind::Fact,
                importance: 0.8,
                expires_at_ms: None,
                metadata: HashMap::new(),
                source: HashMap::new(),
            })
            .unwrap();
        inserted_ids.push(mem_id);
    }
    let total_write_duration = write_timer.elapsed();
    let write_qps = initial_records as f64 / total_write_duration.as_secs_f64();
    println!(
        "  Write throughput ({initial_records} memories): {:?} ({:.1} writes/sec)",
        total_write_duration, write_qps
    );

    // 3. Warm Start Latency (Reopen with journal replay & index rebuild)
    drop(store);
    let warm_start_timer = Instant::now();
    let reopened_store = MemoryStore::open(config).unwrap();
    let warm_start_latency = warm_start_timer.elapsed();
    println!(
        "  Warm start recovery latency ({initial_records} memories): {:?}",
        warm_start_latency
    );

    // 4. Hybrid Recall Throughput (Vector KNN + FTS5 + RRF Fusion)
    let recall_timer = Instant::now();
    let mut accum_query_time = Duration::ZERO;
    for _ in 0..test_queries {
        let query_vec: Vec<f32> = (0..dims).map(|_| random_generator.random()).collect();
        let single_recall_start = Instant::now();

        let recall_response = reopened_store
            .recall(RecallRequest {
                scope: scope.clone(),
                query_embedding: query_vec,
                query_text: Some("domain search text".to_string()),
                limit: 10,
                filters: MemoryFilter::default(),
                include_explanations: true,
                mmr_lambda: Some(0.7),
            })
            .unwrap();

        accum_query_time += single_recall_start.elapsed();
        black_box(recall_response);
    }
    let total_recall_duration = recall_timer.elapsed();
    let recall_qps = test_queries as f64 / total_recall_duration.as_secs_f64();
    let mean_recall_latency_ms = (accum_query_time.as_secs_f64() / test_queries as f64) * 1000.0;

    println!(
        "  Hybrid Recall throughput (Dense + FTS + RRF + MMR): {:.1} QPS",
        recall_qps
    );
    println!("  Mean Hybrid Recall latency: {:.3} ms", mean_recall_latency_ms);

    // 5. Tombstone Vacuum Throughput Benchmark
    // Delete 20% of records
    let delete_count = initial_records / 5;
    for id in inserted_ids.iter().take(delete_count) {
        reopened_store.forget(ForgetRequest {
            operation_id: None,
            scope: scope.clone(),
            id: id.clone(),
        }).unwrap();
    }

    let vacuum_start = Instant::now();
    let purged_count = reopened_store.vacuum_tombstones(100).unwrap();
    let vacuum_duration = vacuum_start.elapsed();
    println!(
        "  Resumable Vacuum throughput ({purged_count} tombstones scrubbed): {:?}",
        vacuum_duration
    );
}

fn generate_vector_dataset(
    dims: usize,
    record_count: usize,
    query_count: usize,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut random_generator = rand::rng();
    let base_vectors: Vec<Vec<f32>> = (0..record_count)
        .map(|_| (0..dims).map(|_| random_generator.random()).collect())
        .collect();
    let query_vectors: Vec<Vec<f32>> = (0..query_count)
        .map(|_| (0..dims).map(|_| random_generator.random()).collect())
        .collect();
    (base_vectors, query_vectors)
}
