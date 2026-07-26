use rand::Rng;
use std::hint::black_box;
use std::time::{Duration, Instant};
use vivy_core::concurrent::VivyIndex;
use vivy_core::distance::Metric;
use vivy_core::flat::FlatIndex;

fn main() {
    println!("Vivy benchmark");

    let dims = 64;
    let n = 5000;
    let n_queries = 100;
    let k = 10;

    let (data, queries) = generate_data(dims, n, n_queries);

    // brute-force index
    println!("\nBuilding brute-force index ({n} vectors)...");
    let t = Instant::now();
    let mut flat = FlatIndex::new(Metric::L2);
    for (i, v) in data.iter().enumerate() {
        flat.insert(i as u64, v.clone());
    }
    let flat_time = t.elapsed();
    println!("  flat build: {flat_time:?}");

    // exact k-NN via full scan
    println!("Computing ground truth...");
    let gt: Vec<Vec<u64>> = queries
        .iter()
        .map(|q| flat.search(q, k).into_iter().map(|(id, _)| id).collect())
        .collect();
    drop(flat);

    // HNSW index
    println!("\nBuilding HNSW index...");
    let t = Instant::now();
    let idx = VivyIndex::new(Metric::L2, None::<&str>, None::<&str>).unwrap();
    for v in &data {
        black_box(idx.insert(black_box(v.clone()))).unwrap();
    }
    let index_time = t.elapsed();
    println!(
        "  hnsw build: {index_time:?} ({:.0} vec/s)",
        n as f64 / index_time.as_secs_f64(),
    );

    // 10 queries
    for q in queries.iter().take(10) {
        black_box(idx.search(q, k));
    }

    // Recall & latency
    println!("\nRecall@{k} benchmark ({n_queries} queries)...");
    let t = Instant::now();
    let mut hits = 0usize;
    let mut search_latency = Duration::ZERO;
    for (i, q) in queries.iter().enumerate() {
        let tq = Instant::now();
        let results = idx.search(q, k);
        search_latency += tq.elapsed();

        for gt_id in &gt[i] {
            if results.iter().any(|(id, _)| id == gt_id) {
                hits += 1;
            }
        }
    }
    let elapsed = t.elapsed();
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
    let data: Vec<Vec<f32>> = (0..n)
        .map(|_| (0..dims).map(|_| rng.random()).collect())
        .collect();
    let queries: Vec<Vec<f32>> = (0..nq)
        .map(|_| (0..dims).map(|_| rng.random()).collect())
        .collect();
    (data, queries)
}
