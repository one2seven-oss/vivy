use rand::Rng;
use std::hint::black_box;
use std::time::{Duration, Instant};
use vivy_core::concurrent::VivyIndex;
use vivy_core::distance::Metric;
use vivy_core::flat::FlatIndex;

fn main() {
    println!("Vivy benchmark");
    println!("=== 64 Dimensions ===");
    run_benchmark(64);
    println!("\n=== 512 Dimensions ===");
    run_benchmark(512);
}

fn run_benchmark(dims: usize) {
    let n = 5000;
    let n_queries = 100;
    let k = 10;

    let (base_vectors, query_vectors) = generate_data(dims, n, n_queries);

    println!("\nBuilding brute-force index ({n} vectors)...");
    let start_time = Instant::now();
    let mut flat = FlatIndex::new(Metric::L2);
    for (i, v) in base_vectors.iter().enumerate() {
        flat.insert((i + 1) as u64, v.clone());
    }
    println!("  flat build: {:?}", start_time.elapsed());

    println!("Computing ground truth...");
    let ground_truth: Vec<Vec<u64>> = query_vectors
        .iter()
        .map(|q| flat.search(q, k).into_iter().map(|(id, _)| id).collect())
        .collect();
    drop(flat);

    println!("\nBuilding HNSW index...");
    let build_timer = Instant::now();
    let idx = VivyIndex::new(dims, Metric::L2, None::<&str>, None::<&str>).unwrap();
    for v in &base_vectors {
        black_box(idx.insert(black_box(v.clone()))).unwrap();
    }
    let index_time = build_timer.elapsed();
    println!(
        "  hnsw build: {index_time:?} ({:.0} vec/s)",
        n as f64 / index_time.as_secs_f64(),
    );

    for q in query_vectors.iter().take(10) {
        black_box(idx.search(q, k)).unwrap();
    }

    println!("\nRecall@{k} benchmark ({n_queries} queries)...");
    let bench_timer = Instant::now();
    let mut hits = 0usize;
    let mut search_latency = Duration::ZERO;
    for (i, q) in query_vectors.iter().enumerate() {
        let query_timer = Instant::now();
        let results = idx.search(q, k).unwrap();
        search_latency += query_timer.elapsed();

        for gt_id in &ground_truth[i] {
            if results.iter().any(|(id, _)| id == gt_id) {
                hits += 1;
            }
        }
    }
    let elapsed = bench_timer.elapsed();
    let total_possible = n_queries * k;
    let recall = hits as f64 / total_possible as f64;
    let qps = n_queries as f64 / elapsed.as_secs_f64();

    println!("  recall@{k}: {recall:.4}");
    println!("  total time: {elapsed:?}");
    println!("  QPS: {qps:.1}");
    println!(
        "  p50 latency: {:.2}ms",
        search_latency.as_secs_f64() / n_queries as f64 * 1000.0,
    );

    let data_mb = (n * dims * 4) as f64 / 1_048_576.0;
    println!("\n  data: {n} x {dims}-dim f32 = {data_mb:.2} MB");
}

fn generate_data(dims: usize, n: usize, nq: usize) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut rng = rand::rng();
    let base_vectors: Vec<Vec<f32>> = (0..n)
        .map(|_| (0..dims).map(|_| rng.random()).collect())
        .collect();
    let query_vectors: Vec<Vec<f32>> = (0..nq)
        .map(|_| (0..dims).map(|_| rng.random()).collect())
        .collect();
    (base_vectors, query_vectors)
}
