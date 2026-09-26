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

use numpy::PyReadonlyArray1;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use vivy_core::concurrent::VivyIndex;
use vivy_core::distance::Metric;
use vivy_core::filter::FilterExpr;

enum PyVectorInput<'py> {
    List(Vec<f32>),
    NumPy(PyReadonlyArray1<'py, f32>),
}

impl<'py> PyVectorInput<'py> {
    fn extract(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        if obj.is_instance_of::<pyo3::types::PyList>() {
            if let Ok(list) = obj.extract::<Vec<f32>>() {
                return Ok(PyVectorInput::List(list));
            }
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArray1<'py, f32>>() {
            if arr.as_slice().is_ok() {
                Ok(PyVectorInput::NumPy(arr))
            } else {
                Err(PyValueError::new_err(
                    "numpy array must be a contiguous 1D array with dtype=float32",
                ))
            }
        } else if obj.hasattr("__array_interface__")? {
            Err(PyTypeError::new_err(
                "vector must be a list of floats or a contiguous 1D numpy array with dtype=float32",
            ))
        } else if let Ok(list) = obj.extract::<Vec<f32>>() {
            Ok(PyVectorInput::List(list))
        } else {
            Err(PyTypeError::new_err(
                "vector must be a list of floats or a contiguous 1D numpy array with dtype=float32",
            ))
        }
    }

    fn as_slice(&self) -> PyResult<&[f32]> {
        match self {
            PyVectorInput::List(v) => Ok(v.as_slice()),
            PyVectorInput::NumPy(arr) => arr.as_slice().map_err(|e| {
                PyValueError::new_err(format!("failed to access contiguous numpy slice: {e}"))
            }),
        }
    }

    fn into_vec(self) -> PyResult<Vec<f32>> {
        match self {
            PyVectorInput::List(v) => Ok(v),
            PyVectorInput::NumPy(arr) => {
                let slice = arr.as_slice().map_err(|e| {
                    PyValueError::new_err(format!("failed to access contiguous numpy slice: {e}"))
                })?;
                Ok(slice.to_vec())
            }
        }
    }
}

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
    fn insert(
        &self,
        py: Python<'_>,
        vector: Bound<'_, PyAny>,
        metadata: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<u64> {
        let meta = parse_metadata(metadata)?;
        let vec_input = PyVectorInput::extract(&vector)?;
        let vec_data = vec_input.into_vec()?;

        py.allow_threads(move || {
            self.inner
                .insert_with_metadata(vec_data, meta)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })
    }

    #[pyo3(signature = (vectors, metadata=None))]
    fn insert_batch(
        &self,
        py: Python<'_>,
        vectors: Bound<'_, PyAny>,
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

        let parsed_vectors: Vec<Vec<f32>> = if let Ok(arr2) = vectors.extract::<numpy::PyReadonlyArray2<'_, f32>>() {
            let slice2 = arr2.as_array();
            let mut vecs = Vec::with_capacity(slice2.shape()[0]);
            for row in slice2.outer_iter() {
                if let Some(s) = row.as_slice() {
                    vecs.push(s.to_vec());
                } else {
                    return Err(PyValueError::new_err("2D numpy array must be C-contiguous float32"));
                }
            }
            vecs
        } else if let Ok(seq) = vectors.extract::<Vec<Bound<'_, PyAny>>>() {
            let mut vecs = Vec::with_capacity(seq.len());
            for item in seq {
                let vec_input = PyVectorInput::extract(&item)?;
                vecs.push(vec_input.into_vec()?);
            }
            vecs
        } else {
            return Err(PyTypeError::new_err(
                "vectors must be a list of vectors or a 2D float32 numpy array",
            ));
        };

        py.allow_threads(move || {
            self.inner
                .insert_batch_with_metadata(parsed_vectors, metas)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })
    }

    #[pyo3(signature = (query, k, filter=None))]
    fn search(
        &self,
        py: Python<'_>,
        query: Bound<'_, PyAny>,
        k: usize,
        filter: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Vec<(u64, f32)>> {
        let expr = parse_filter(filter)?;
        let vec_input = PyVectorInput::extract(&query)?;
        let query_slice = vec_input.as_slice()?;

        let results = py.allow_threads(move || {
            self.inner
                .search_filtered(query_slice, k, expr.as_ref())
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
        embedding: Bound<'_, PyAny>,
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

        let vec_input = PyVectorInput::extract(&embedding)?;
        let vec_data = vec_input.into_vec()?;

        let req = vivy_memory::RememberRequest {
            operation_id,
            scope,
            content: content.to_string(),
            embedding: vec_data,
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

    #[pyo3(signature = (records))]
    fn remember_batch(
        &self,
        py: Python<'_>,
        records: Vec<Bound<'_, PyDict>>,
    ) -> PyResult<Vec<String>> {
        let mut rust_reqs = Vec::with_capacity(records.len());

        for dict in records {
            let tenant_id: String = dict
                .get_item("tenant_id")?
                .ok_or_else(|| PyValueError::new_err("missing tenant_id"))?
                .extract()?;
            let namespace: String = dict
                .get_item("namespace")?
                .ok_or_else(|| PyValueError::new_err("missing namespace"))?
                .extract()?;
            let content: String = dict
                .get_item("content")?
                .ok_or_else(|| PyValueError::new_err("missing content"))?
                .extract()?;
            let embedding_obj = dict
                .get_item("embedding")?
                .ok_or_else(|| PyValueError::new_err("missing embedding"))?;
            let vec_input = PyVectorInput::extract(&embedding_obj)?;
            let embedding = vec_input.into_vec()?;

            let importance: f32 = match dict.get_item("importance")? {
                Some(val) => val.extract().unwrap_or(0.5),
                None => 0.5,
            };

            let agent_id: Option<String> = match dict.get_item("agent_id")? {
                Some(val) => val.extract().ok(),
                None => None,
            };

            let user_id: Option<String> = match dict.get_item("user_id")? {
                Some(val) => val.extract().ok(),
                None => None,
            };

            let operation_id: Option<String> = match dict.get_item("operation_id")? {
                Some(val) => val.extract().ok(),
                None => None,
            };

            let expires_at_ms: Option<i64> = match dict.get_item("expires_at_ms")? {
                Some(val) => val.extract().ok(),
                None => None,
            };

            let kind_str: Option<String> = match dict.get_item("kind")? {
                Some(val) => val.extract().ok(),
                None => None,
            };

            let mut scope = vivy_memory::MemoryScope::new(&tenant_id, &namespace)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            if let Some(ref agent) = agent_id {
                scope = scope.with_agent(agent).map_err(|e| PyValueError::new_err(e.to_string()))?;
            }
            if let Some(ref user) = user_id {
                scope = scope.with_user(user).map_err(|e| PyValueError::new_err(e.to_string()))?;
            }

            let m_kind = match kind_str.as_deref().unwrap_or("fact").to_lowercase().as_str() {
                "preference" => vivy_memory::MemoryKind::Preference,
                "instruction" => vivy_memory::MemoryKind::Instruction,
                "context" => vivy_memory::MemoryKind::Context,
                "episodic" => vivy_memory::MemoryKind::Episodic,
                _ => vivy_memory::MemoryKind::Fact,
            };

            rust_reqs.push(vivy_memory::RememberRequest {
                operation_id,
                scope,
                content,
                embedding,
                kind: m_kind,
                importance,
                expires_at_ms,
                metadata: std::collections::HashMap::new(),
                source: std::collections::HashMap::new(),
            });
        }

        let store = self.inner.clone();
        py.allow_threads(move || {
            store
                .remember_batch(rust_reqs)
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
        query_embedding: Bound<'_, PyAny>,
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

        let vec_input = PyVectorInput::extract(&query_embedding)?;
        let query_vec = vec_input.into_vec()?;

        let req = vivy_memory::RecallRequest {
            scope,
            query_embedding: query_vec,
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

    #[pyo3(signature = (tenant_id, namespace, id, agent_id=None, user_id=None))]
    fn get(
        &self,
        py: Python<'_>,
        tenant_id: &str,
        namespace: &str,
        id: &str,
        agent_id: Option<&str>,
        user_id: Option<&str>,
    ) -> PyResult<Option<PyObject>> {
        let mut scope = vivy_memory::MemoryScope::new(tenant_id, namespace)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        if let Some(agent) = agent_id {
            scope = scope.with_agent(agent).map_err(|e| PyValueError::new_err(e.to_string()))?;
        }
        if let Some(user) = user_id {
            scope = scope.with_user(user).map_err(|e| PyValueError::new_err(e.to_string()))?;
        }

        let store = self.inner.clone();
        let record = py.allow_threads(move || {
            store
                .get(&scope, id)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })?;

        match record {
            Some(rec) if rec.status == vivy_memory::MemoryStatus::Active => {
                let dict = PyDict::new(py);
                dict.set_item("id", rec.id)?;
                dict.set_item("tenant_id", rec.scope.tenant_id())?;
                dict.set_item("namespace", rec.scope.namespace())?;
                dict.set_item("content", rec.content)?;
                dict.set_item("importance", rec.importance)?;
                dict.set_item("revision", rec.revision)?;
                dict.set_item("created_at_ms", rec.created_at_ms)?;
                dict.set_item("updated_at_ms", rec.updated_at_ms)?;
                dict.set_item("expires_at_ms", rec.expires_at_ms)?;
                Ok(Some(dict.into()))
            }
            _ => Ok(None),
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (tenant_id, namespace, id, expected_revision, content=None, embedding=None, kind=None, importance=None, agent_id=None, user_id=None, operation_id=None, expires_at_ms=None))]
    fn update(
        &self,
        py: Python<'_>,
        tenant_id: &str,
        namespace: &str,
        id: &str,
        expected_revision: u64,
        content: Option<String>,
        embedding: Option<Bound<'_, PyAny>>,
        kind: Option<&str>,
        importance: Option<f32>,
        agent_id: Option<&str>,
        user_id: Option<&str>,
        operation_id: Option<String>,
        expires_at_ms: Option<i64>,
    ) -> PyResult<()> {
        let mut scope = vivy_memory::MemoryScope::new(tenant_id, namespace)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        if let Some(agent) = agent_id {
            scope = scope.with_agent(agent).map_err(|e| PyValueError::new_err(e.to_string()))?;
        }
        if let Some(user) = user_id {
            scope = scope.with_user(user).map_err(|e| PyValueError::new_err(e.to_string()))?;
        }

        let m_kind = match kind {
            Some(k) => match k.to_lowercase().as_str() {
                "preference" => Some(vivy_memory::MemoryKind::Preference),
                "fact" => Some(vivy_memory::MemoryKind::Fact),
                "instruction" => Some(vivy_memory::MemoryKind::Instruction),
                "context" => Some(vivy_memory::MemoryKind::Context),
                _ => Some(vivy_memory::MemoryKind::Episodic),
            },
            None => None,
        };

        let embedding_vec = match embedding {
            Some(obj) => Some(PyVectorInput::extract(&obj)?.into_vec()?),
            None => None,
        };

        let req = vivy_memory::UpdateRequest {
            operation_id,
            scope,
            id: id.to_string(),
            expected_revision,
            content,
            embedding: embedding_vec,
            kind: m_kind,
            importance,
            expires_at_ms: expires_at_ms.map(Some),
            metadata_patch: None,
        };

        let store = self.inner.clone();
        py.allow_threads(move || {
            store
                .update(req)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })
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

    #[pyo3(signature = (tenant_id, namespace, ids))]
    fn forget_batch(
        &self,
        py: Python<'_>,
        tenant_id: &str,
        namespace: &str,
        ids: Vec<String>,
    ) -> PyResult<()> {
        let scope = vivy_memory::MemoryScope::new(tenant_id, namespace)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let id_strs: Vec<&str> = ids.iter().map(|s| s.as_str()).collect();
        let store = self.inner.clone();
        py.allow_threads(move || {
            store
                .forget_batch(&scope, &id_strs)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })
    }

    fn health(&self, py: Python<'_>) -> PyResult<PyObject> {
        let store = self.inner.clone();
        let health = py.allow_threads(move || {
            store.health().map_err(|e| PyValueError::new_err(e.to_string()))
        })?;

        let dict = PyDict::new(py);
        dict.set_item("is_healthy", health.is_healthy)?;
        dict.set_item("total_active_records", health.total_active_records)?;
        dict.set_item("total_tombstoned_records", health.total_tombstoned_records)?;
        dict.set_item("pending_operations_count", health.pending_operations_count)?;
        dict.set_item("index_rebuild_required", health.index_rebuild_required)?;
        dict.set_item("db_size_bytes", health.db_size_bytes)?;
        dict.set_item("wal_size_bytes", health.wal_size_bytes)?;
        Ok(dict.into())
    }

    #[pyo3(signature = (batch_size=100))]
    fn vacuum_tombstones(&self, py: Python<'_>, batch_size: usize) -> PyResult<usize> {
        let store = self.inner.clone();
        py.allow_threads(move || {
            store
                .vacuum_tombstones(batch_size)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })
    }

    fn rebuild_index(&self, py: Python<'_>) -> PyResult<()> {
        let store = self.inner.clone();
        py.allow_threads(move || {
            store
                .rebuild_index()
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
