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
    #[new]
    fn new(dims: usize, metric: &str) -> PyResult<Self> {
        let m = match metric {
            "l2" | "L2" => Metric::L2,
            "cosine" | "Cosine" => Metric::Cosine,
            "dot" | "Dot" => Metric::Dot,
            other => return Err(PyValueError::new_err(format!("unknown metric: {other}"))),
        };
        let inner = VivyIndex::new(dims, m, Option::<&str>::None, Option::<&str>::None)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (vector, metadata=None))]
    fn insert(&self, py: Python<'_>, vector: Vec<f32>, metadata: Option<&Bound<'_, PyDict>>) -> PyResult<u64> {
        let meta = parse_metadata(metadata)?;
        py.allow_threads(move || {
            self.inner.insert_with_metadata(vector, meta)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })
    }

    #[pyo3(signature = (vectors, metadata=None))]
    fn insert_batch(
        &self,
        py: Python<'_>,
        vectors: Vec<Vec<f32>>,
        metadata: Option<Vec<Option<Bound<'_, PyDict>>>>,
    ) -> PyResult<Vec<u64>> {
        let metas = match metadata {
            Some(list) => {
                let mut parsed = Vec::with_capacity(list.len());
                for item in list {
                    let m = match item {
                        Some(ref d) => parse_metadata(Some(d))?,
                        None => Vec::new(),
                    };
                    parsed.push(m);
                }
                Some(parsed)
            }
            None => None,
        };

        py.allow_threads(move || {
            self.inner.insert_batch_with_metadata(vectors, metas)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })
    }

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
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })?;
        Ok(results)
    }

    // Delta segment count (excludes sealed — approximate "recent inserts").
    fn __len__(&self) -> usize {
        self.inner.delta_len()
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
        if let Ok(value) = val.extract::<String>() {
            exprs.push(FilterExpr::Equals { field, value });
        } else if let Ok(values) = val.extract::<Vec<String>>() {
            exprs.push(FilterExpr::In { field, values });
        } else {
            return Err(PyValueError::new_err("filter values must be a string or a list of strings"));
        }
    }
    if exprs.is_empty() {
        return Ok(None);
    }
    match exprs.len() {
        1 => Ok(Some(exprs.into_iter().next().unwrap())),
        _ => Ok(Some(FilterExpr::And(exprs))),
    }
}

// Python facade for vivy_memory::MemoryStore
#[pyclass(name = "MemoryStore")]
struct PyMemoryStore {
    inner: std::sync::Arc<vivy_memory::MemoryStore>,
}

#[pymethods]
impl PyMemoryStore {
    #[staticmethod]
    #[pyo3(signature = (path, dimensions, embedding_model, max_recall_limit=100))]
    fn open(
        path: &str,
        dimensions: usize,
        embedding_model: &str,
        max_recall_limit: usize,
    ) -> PyResult<Self> {
        let config = vivy_memory::MemoryConfig::builder(path)
            .dimensions(dimensions)
            .embedding_model(embedding_model)
            .max_recall_limit(max_recall_limit)
            .build()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let store = vivy_memory::MemoryStore::open(config)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(Self {
            inner: std::sync::Arc::new(store),
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (tenant_id, namespace, content, embedding, kind="fact", importance=0.5, agent_id=None, user_id=None, operation_id=None, expires_at_ms=None))]
    fn remember(
        &self,
        py: Python<'_>,
        tenant_id: &str,
        namespace: &str,
        content: &str,
        embedding: Vec<f32>,
        kind: &str,
        importance: f32,
        agent_id: Option<&str>,
        user_id: Option<&str>,
        operation_id: Option<String>,
        expires_at_ms: Option<i64>,
    ) -> PyResult<String> {
        let mut scope = vivy_memory::MemoryScope::new(tenant_id, namespace)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        if let Some(agent) = agent_id {
            scope = scope.with_agent(agent).map_err(|e| PyValueError::new_err(e.to_string()))?;
        }
        if let Some(user) = user_id {
            scope = scope.with_user(user).map_err(|e| PyValueError::new_err(e.to_string()))?;
        }

        let m_kind = match kind.to_lowercase().as_str() {
            "preference" => vivy_memory::MemoryKind::Preference,
            "fact" => vivy_memory::MemoryKind::Fact,
            "instruction" => vivy_memory::MemoryKind::Instruction,
            "context" => vivy_memory::MemoryKind::Context,
            _ => vivy_memory::MemoryKind::Episodic,
        };

        let req = vivy_memory::RememberRequest {
            operation_id,
            scope,
            content: content.to_string(),
            embedding,
            kind: m_kind,
            importance,
            expires_at_ms,
            metadata: std::collections::HashMap::new(),
            source: std::collections::HashMap::new(),
        };

        let store = self.inner.clone();
        py.allow_threads(move || {
            store
                .remember(req)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (tenant_id, namespace, query_embedding, query_text=None, limit=5, agent_id=None, user_id=None, include_explanations=true, mmr_lambda=None))]
    fn recall(
        &self,
        py: Python<'_>,
        tenant_id: &str,
        namespace: &str,
        query_embedding: Vec<f32>,
        query_text: Option<String>,
        limit: usize,
        agent_id: Option<&str>,
        user_id: Option<&str>,
        include_explanations: bool,
        mmr_lambda: Option<f32>,
    ) -> PyResult<Vec<(String, String, f32)>> {
        let mut scope = vivy_memory::MemoryScope::new(tenant_id, namespace)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        if let Some(agent) = agent_id {
            scope = scope.with_agent(agent).map_err(|e| PyValueError::new_err(e.to_string()))?;
        }
        if let Some(user) = user_id {
            scope = scope.with_user(user).map_err(|e| PyValueError::new_err(e.to_string()))?;
        }

        let req = vivy_memory::RecallRequest {
            scope,
            query_embedding,
            query_text,
            limit,
            filters: vivy_memory::MemoryFilter::default(),
            include_explanations,
            mmr_lambda,
        };

        let store = self.inner.clone();
        let resp = py.allow_threads(move || {
            store
                .recall(req)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })?;

        let results = resp
            .items
            .into_iter()
            .map(|item| (item.memory.id, item.memory.content, item.score))
            .collect();

        Ok(results)
    }

    #[pyo3(signature = (tenant_id, namespace, id))]
    fn forget(
        &self,
        py: Python<'_>,
        tenant_id: &str,
        namespace: &str,
        id: &str,
    ) -> PyResult<()> {
        let scope = vivy_memory::MemoryScope::new(tenant_id, namespace)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let req = vivy_memory::ForgetRequest::new(scope, id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let store = self.inner.clone();
        py.allow_threads(move || {
            store
                .forget(req)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })
    }
}

#[pymodule]
fn vivy(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Index>()?;
    m.add_class::<PyMemoryStore>()?;
    Ok(())
}
