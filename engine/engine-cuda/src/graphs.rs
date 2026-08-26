//! RAII wrappers for CUDA Graphs (`CudaGraph` and `CudaGraphExec`).
//!
//! CUDA Graphs allow capturing an arbitrary sequence of stream operations
//! (kernel launches, device-to-device transfers, event waits) into a directed
//! acyclic graph (DAG) and instantiating it once, enabling the entire graph
//! to be launched in a single driver call with near-zero host CPU overhead.

use crate::error::CudaError;
use crate::streams::CudaStream;
use cudarc::driver::CudaDevice;
use cudarc::driver::sys::{self, CUgraph, CUgraphExec, CUresult};
use std::sync::Arc;

/// RAII wrapper around a captured CUDA graph topology (`CUgraph`).
#[derive(Debug)]
pub struct CudaGraph {
    graph: CUgraph,
    device: Arc<CudaDevice>,
}

// SAFETY: `CUgraph` is an opaque CUDA driver handle tied to `Arc<CudaDevice>`.
unsafe impl Send for CudaGraph {}
unsafe impl Sync for CudaGraph {}

impl CudaGraph {
    /// Creates a `CudaGraph` from a raw `CUgraph` handle and owning device.
    ///
    /// # Safety
    /// `graph` must be a valid, un-destroyed `CUgraph` handle created in `device`'s context.
    pub unsafe fn from_raw(graph: CUgraph, device: Arc<CudaDevice>) -> Self {
        Self { graph, device }
    }

    /// Returns the raw `CUgraph` handle.
    pub fn raw(&self) -> CUgraph {
        self.graph
    }

    /// Instantiates the captured graph into an executable graph (`CudaGraphExec`).
    pub fn instantiate(&self) -> Result<CudaGraphExec, CudaError> {
        self.device.bind_to_thread()?;

        let mut exec: CUgraphExec = std::ptr::null_mut();

        // SAFETY:
        // `self.device.bind_to_thread()` ensured valid CUDA context.
        // `self.graph` is a valid `CUgraph` handle.
        // `&mut exec` is a valid pointer on the stack.
        let res = unsafe {
            let lib = sys::lib();
            lib.cuGraphInstantiateWithFlags(&mut exec, self.graph, 0)
        };

        if res != CUresult::CUDA_SUCCESS || exec.is_null() {
            return Err(CudaError::GraphFailed("cuGraphInstantiateWithFlags", res));
        }

        Ok(CudaGraphExec {
            exec,
            device: Arc::clone(&self.device),
        })
    }
}

impl Drop for CudaGraph {
    fn drop(&mut self) {
        if !self.graph.is_null() {
            let _ = self.device.bind_to_thread();
            // SAFETY: `self.graph` is non-null and owned by this instance.
            unsafe {
                let lib = sys::lib();
                let _ = lib.cuGraphDestroy(self.graph);
            }
            self.graph = std::ptr::null_mut();
        }
    }
}

/// RAII wrapper around an instantiated, executable CUDA graph (`CUgraphExec`).
#[derive(Debug)]
pub struct CudaGraphExec {
    exec: CUgraphExec,
    device: Arc<CudaDevice>,
}

// SAFETY: `CUgraphExec` is an opaque CUDA driver handle tied to `Arc<CudaDevice>`.
unsafe impl Send for CudaGraphExec {}
unsafe impl Sync for CudaGraphExec {}

impl CudaGraphExec {
    /// Returns the raw `CUgraphExec` handle.
    pub fn raw(&self) -> CUgraphExec {
        self.exec
    }

    /// Launches the executable graph asynchronously on `stream`.
    pub fn launch(&self, stream: &CudaStream) -> Result<(), CudaError> {
        self.device.bind_to_thread()?;

        // SAFETY:
        // `self.device.bind_to_thread()` ensured valid CUDA context.
        // `self.exec` is a valid executable graph handle.
        // `stream.raw()` is a valid CUDA stream handle.
        let res = unsafe {
            let lib = sys::lib();
            lib.cuGraphLaunch(self.exec, stream.raw())
        };

        if res != CUresult::CUDA_SUCCESS {
            return Err(CudaError::GraphFailed("cuGraphLaunch", res));
        }

        Ok(())
    }
}

impl Drop for CudaGraphExec {
    fn drop(&mut self) {
        if !self.exec.is_null() {
            let _ = self.device.bind_to_thread();
            // SAFETY: `self.exec` is non-null and owned by this instance.
            unsafe {
                let lib = sys::lib();
                let _ = lib.cuGraphExecDestroy(self.exec);
            }
            self.exec = std::ptr::null_mut();
        }
    }
}
