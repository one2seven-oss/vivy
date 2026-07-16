//! Python bindings (PyO3) for vivy.
//!
//! Key decisions:
//! - `allow_threads()` releases the GIL on every insert/search so Python
//!   threads can do data loading while Rust handles vector search.
//! - Currently accepts `Vec<f32>` from Python. Large arrays should use
//!   the buffer protocol (NumPy `__array_interface__`) — next optimisation.
//! - Metadata dicts → `Vec<(String, String)>`. Simplistic: all values are
//!   strings, filters are strict equality. A richer type system needs a
//!   more expressive Rust-side filter DSL first.
//! - Filter dicts are always AND-combined. The flat dict form is ergonomic
//!   for the common case (`color=red AND year=2024`).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use vivy_core::concurrent::VivyIndex;
use vivy_core::distance::Metric;
use vivy_core::filter::FilterExpr;

// Python-facing handle wrapping VivyIndex. Single opaque object with insert/search.
#[pyclass]
struct Index {
    inner: VivyIndex,
}

#[pymethods]
impl Index {
    // metric: "l2", "cosine", "dot" (case-insensitive, fixed for lifetime).
    #[new]
    fn new(_dims: usize, metric: &str) -> PyResult<Self> {
        let m = match metric {
            "l2" | "L2" => Metric::L2,
            "cosine" | "Cosine" => Metric::Cosine,
            "dot" | "Dot" => Metric::Dot,
            other => return Err(PyValueError::new_err(format!("unknown metric: {other}"))),
        };
        let inner = VivyIndex::new(m, Option::<&str>::None, Option::<&str>::None)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    // vector: list[f32], metadata: optional dict[str, str]. Returns auto-assigned u64 ID.
    #[pyo3(signature = (vector, metadata=None))]
    fn insert(&self, py: Python<'_>, vector: Vec<f32>, metadata: Option<&Bound<'_, PyDict>>) -> PyResult<u64> {
        let meta = parse_metadata(metadata)?;
        py.allow_threads(move || {
            self.inner.insert_with_metadata(vector, meta)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })
    }

    // query: list[f32], k: int, filter: optional dict[str, str].
    // Returns list of (id, distance) sorted by increasing distance.
    #[pyo3(signature = (query, k, filter=None))]
    fn search(
        &self,
        py: Python<'_>,
        query: Vec<f32>,
        k: usize,
        filter: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Vec<(u64, f32)>> {
        let expr = parse_filter(filter)?;
        let results = py.allow_threads(move || {
            self.inner.search_filtered(&query, k, expr.as_ref())
        });
        Ok(results)
    }

    // Delta segment count (excludes sealed — approximate "recent inserts").
    fn __len__(&self) -> usize {
        self.inner.delta_len()
    }

    fn metrics(&self, py: Python<'_>) -> PyResult<PyObject> {
        let snap = self.inner.metrics.snapshot();
        let d = PyDict::new(py);
        d.set_item("inserts_total", snap.inserts_total)?;
        d.set_item("searches_total", snap.searches_total)?;
        d.set_item("compactions_total", snap.compactions_total)?;
        d.set_item("avg_insert_time_ns", snap.avg_insert_time_ns)?;
        d.set_item("avg_search_time_ns", snap.avg_search_time_ns)?;
        d.set_item("delta_size", snap.delta_size)?;
        d.set_item("num_sealed", snap.num_sealed)?;
        Ok(d.into())
    }
}

// Python dict → Vec<(String, String)>. All values coerced to string. None/empty = no-op.
fn parse_metadata(dict: Option<&Bound<'_, PyDict>>) -> PyResult<Vec<(String, String)>> {
    let Some(d) = dict else { return Ok(Vec::new()) };
    let mut meta = Vec::with_capacity(d.len());
    for (key, val) in d.iter() {
        let k: String = key.extract()?;
        let v: String = val.extract()?;
        meta.push((k, v));
    }
    Ok(meta)
}

// Python dict → Option<FilterExpr>. AND-combined. Single pair = no And wrapper. None/empty = None.
fn parse_filter(dict: Option<&Bound<'_, PyDict>>) -> PyResult<Option<FilterExpr>> {
    let Some(d) = dict else { return Ok(None) };
    let mut exprs = Vec::new();
    for (key, val) in d.iter() {
        let field: String = key.extract()?;
        let value: String = val.extract()?;
        exprs.push(FilterExpr::Equals { field, value });
    }
    if exprs.is_empty() {
        return Ok(None);
    }
    match exprs.len() {
        1 => Ok(Some(exprs.into_iter().next().unwrap())),
        _ => Ok(Some(FilterExpr::And(exprs))),
    }
}

// Registers the Index class under the "vivy" namespace.
#[pymodule]
fn vivy(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Index>()?;
    Ok(())
}
