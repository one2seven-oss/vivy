use rand::Rng;
use std::time::Instant;
use tempfile::tempdir;
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
    println!("\nBuilding brute-force index ({} vectors)...", n);
    let start = Instant::now();
    let mut flat = FlatIndex::new(Metric::L2);
    for (i, v) in data.iter().enumerate() {
        flat.insert(i as u64, v.clone());
    }
    let flat_time = start.elapsed();
    println!("  flat build: {:?}", flat_time);

    // Compute ground truth
    println!("Computing ground truth...");
    let gt: Vec<Vec<u64>> = queries
        .iter()
        .map(|q| flat.search(q, k).into_iter().map(|(id, _)| id).collect())
        .collect();
    drop(flat);

    // Build HNSW index
    println!("\nBuilding HNSW index...");
    let _dir = tempdir().unwrap();
    let start = Instant::now();
    let idx = VivyIndex::new(Metric::L2, Option::<&str>::None, Option::<&str>::None).unwrap();
    for v in &data {
        idx.insert(v.clone()).unwrap();
    }
    let index_time = start.elapsed();
    println!(
        "  hnsw build: {:?} ({} vec/s)",
        index_time,
        n as f64 / index_time.as_secs_f64()
    );

    // Warmup
    for q in &queries[..10] {
        let _ = idx.search(q, k);
    }

    // Recall benchmark
    println!("\nRecall@{} benchmark ({} queries)...", k, n_queries);
    let start = Instant::now();
    let mut hits = 0usize;
    for (i, q) in queries.iter().enumerate() {
        let results = idx.search(q, k);
        let result_ids: Vec<u64> = results.into_iter().map(|(id, _)| id).collect();
        for gt_id in &gt[i] {
            if result_ids.contains(gt_id) {
                hits += 1;
            }
        }
    }
    let elapsed = start.elapsed();
    let total_possible = n_queries * k;
    let recall = hits as f64 / total_possible as f64;
    let qps = n_queries as f64 / elapsed.as_secs_f64();

    println!("  recall@{}: {:.4}", k, recall);
    println!("  total time: {:?}", elapsed);
    println!("  QPS: {:.1}", qps);
    println!(
        "  p50 latency: {:.2}ms",
        (elapsed.as_secs_f64() / n_queries as f64) * 1000.0
    );
}

fn generate_data(dims: usize, n: usize, nq: usize) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut rng = rand::rng();
    let data: Vec<Vec<f32>> = (0..n)
        .map(|_| (0..dims).map(|_| rng.random::<f32>()).collect())
        .collect();
    let queries: Vec<Vec<f32>> = (0..nq)
        .map(|_| (0..dims).map(|_| rng.random::<f32>()).collect())
        .collect();
    (data, queries)
}
