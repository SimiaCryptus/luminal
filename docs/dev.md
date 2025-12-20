Here is the comprehensive **Luminal Developer Guide**. It combines the core architecture, neural network primitives, training mechanisms, and backend implementation details into a single, cohesive document.

---

# Luminal Framework Developer Guide

Luminal is a deep learning compiler framework designed to optimize computational graphs via equality saturation (using `egglog`) and execute them efficiently on hardware (Apple Silicon via Metal, NVIDIA GPUs via CUDA, and CPU).

Unlike eager-execution frameworks (like PyTorch), Luminal is **lazy**: operations build a graph, which is then compiled, optimized, and executed.

---

## Table of Contents

1.  [**Core Architecture**](#1-core-architecture)
    *   The Graph Engine
    *   Tensors & Data
    *   Symbolic Shapes
2.  [**Neural Network Primitives (`luminal_nn`)**](#2-neural-network-primitives-luminal_nn)
    *   The Module System
    *   Layers & Activations
3.  [**Training & Autograd (`luminal_training`)**](#3-training--autograd-luminal_training)
    *   Automatic Differentiation
    *   Optimizers & State
4.  [**The Compiler Pipeline**](#4-the-compiler-pipeline)
    *   Compiler Traits & Passes
    *   Egglog Optimization
5.  [**Backends & Execution**](#5-backends--execution)
    *   Metal (Apple Silicon)
    *   CUDA (NVIDIA) & The Megakernel

---

## 1. Core Architecture

The core of Luminal (`luminal/src`) handles graph construction, shape tracking, and the interface between high-level operations and low-level execution.

### The Graph Engine (`graph.rs`)
The `Graph` struct is the central hub. It acts as both the definition of the neural network and the runtime state.

*   **Topology:** Stores nodes (`Operator`s) and edges (`Dependency`s).
*   **Data:** Manages tensor data (`Vec<f32>` or GPU buffers) indexed by `(NodeIndex, output_index)`.
*   **Lifecycle:**
    1.  **Construction:** Users add nodes via high-level ops.
    2.  **Compilation:** `Compiler` passes transform the graph structure.
    3.  **Execution:** The graph is topologically sorted and executed (or JIT compiled).

### Tensors (`graph_tensor.rs`)
The `GraphTensor` is a lightweight handle. It **does not** hold data directly.
*   **Structure:** Contains a `NodeIndex`, a pointer to the `Graph`, and a `ShapeTracker`.
*   **Role:** It provides the fluent API (e.g., `x.matmul(y).relu()`).
*   **Lifecycle Methods:**
    *   `keep()`: Prevents the compiler from optimizing the node away.
    *   `retrieve()`: Marks data to be fetched from the GPU after execution.
    *   `set(data)`: Injects data into the graph.

### Symbolic Shapes (`shape/`)
Luminal uses symbolic algebra for shapes, allowing for zero-copy views and dynamic resolution.

*   **`ShapeTracker` (`tracker.rs`):** Tracks dimensions and strides. It allows operations like `permute`, `slice`, and `broadcast` to be performed purely by modifying metadata, without moving memory.
*   **`Expression` (`symbolic.rs`):** Represents dimensions as algebraic formulas (e.g., `(BatchSize * 2) + 1`).
    *   Uses **Reverse Polish Notation (RPN)** internally.
    *   Can be compiled to C-style strings for GPU kernels (e.g., `(idx / 512) % 10`).
    *   Uses `egg` to simplify expressions (e.g., resolving `x * 1` to `x`).

---

## 2. Neural Network Primitives (`luminal_nn`)

The `luminal_nn` crate provides high-level abstractions for building deep learning models.

### The Module System
All layers implement the `Module` trait. State (weights/biases) is managed via `GraphTensor`s stored within the structs.

```rust
pub trait Module<I> {
    type Output;
    fn forward(&self, input: I) -> Self::Output;
}
```

*   **Stateful Modules:** Layers like `Linear`, `Conv2D`, and `LayerNorm` hold `GraphTensor` fields for weights.
*   **Stateless Modules:** Activations like `ReLU` or `Tanh` are unit structs that simply call the underlying graph operation.
*   **Serialization:** The `SerializeModule` trait allows traversing the module tree to save/load weights.

### Key Layers
*   **Linear:** Implements dense layers. Supports permuted weights for optimization.
*   **Convolution:** `Conv1D`, `Conv2D`, `Conv3D`. Implemented via `im2col` logic (lowered to `pool_last_dim` -> `reshape` -> `matmul`).
*   **Normalization:** `LayerNorm` (Standardization + Affine transform).
*   **Embedding:** Lookup tables. Supports permuted storage for memory access optimization.

---

## 3. Training & Autograd (`luminal_training`)

Training is handled by extending the graph with backward-pass nodes and update rules.

### Automatic Differentiation (`autograd.rs`)
The `Autograd` struct is a **Compiler Pass**. It transforms a forward-only graph into a training graph.

1.  **Pruning:** Identifies the "Valid Set" of nodes—those that are descendants of parameters and ancestors of the loss.
2.  **Reverse Traversal:** Walks the graph from Loss to Inputs.
3.  **Gradient Injection:** For every operation (e.g., `MatMul`), it injects the corresponding gradient math nodes (e.g., `Transpose` + `MatMul`) into the graph.
4.  **Accumulation:** Gradients from multiple consumers are summed (Multivariate Chain Rule).

### Optimizers (`optimizer.rs`)
Optimizers like `SGD` and `Adam` are not external loops; they are graph constructors.

*   **Graph Extension:** Calling `Adam::new(...)` adds nodes to the graph that calculate $w_{t+1}$ based on $w_t$ and $\nabla w$.
*   **State Management:** Since graphs are DAGs (acyclic), state updates (Momentum, Variance) are handled by:
    1.  Creating input/output pairs for state tensors.
    2.  Using `step_after_execution` to manually copy the output buffer of step $T$ to the input buffer of step $T+1$.

---

## 4. The Compiler Pipeline

The compiler transforms the user's high-level graph into an optimized executable.

### Compiler Traits (`compiler_utils.rs`)
*   **`Compiler` Trait:** The interface for any pass (e.g., `CSE`, `Autograd`).
*   **`GenericCompiler`:** A standard suite of platform-agnostic passes:
    *   **CSE:** Common Subexpression Elimination.
    *   **DCE:** Dead Code Elimination (`RemoveUnusedNodes`).
    *   **ArithmeticElimination:** Simplifies math (e.g., `x + 0 -> x`, `log(exp(x)) -> x`).

### Egglog Optimization
Luminal integrates `egglog` for equality saturation.

1.  **Lowering (`op.rs`):** Operators implement `to_egglog` to convert themselves into S-expressions.
2.  **Saturation:** `egglog` applies rewrite rules (defined in `logical.rs` and `ops.rs`) to explore thousands of equivalent graph versions.
3.  **Extraction (`extract.rs`):** A cost-model search finds the most performant trajectory. It compiles candidate subgraphs to kernels, runs them on hardware to measure actual execution time, and selects the fastest version.

---

## 5. Backends & Execution

Once compiled, the graph is executed on a specific backend.

### Metal Backend (Apple Silicon)
Located in `luminal/src/run.rs`.

*   **JIT Compilation:** Converts `Kernel` nodes containing Metal Shading Language (MSL) code into `ComputePipelineState`s at runtime.
*   **Execution:**
    1.  Allocates `MTLBuffer`s.
    2.  Topologically sorts the graph.
    3.  Encodes commands into a `CommandBuffer`.
    4.  Dispatches thread groups based on the `ShapeTracker` dimensions.

### CUDA Backend (NVIDIA)
Located in `luminal_cuda`. This backend uses a hybrid execution model.

#### 1. The Megakernel (Block Interpreter)
To reduce CPU-GPU synchronization overhead, Luminal fuses complex subgraphs into a single "Megakernel".
*   **`interpreter.cu`:** A persistent C++ CUDA kernel that loops indefinitely, fetching `Task`s from a queue.
*   **`BlockOps`:** Fused operations (e.g., `RowSwishMul`, `TileMatmul`, `GQAAttention`) that run inside the interpreter.
*   **Synchronization:** Uses a custom semaphore system (`ready` array) in global memory to handle dependencies between tasks within the Megakernel.

#### 2. Standalone Kernels
Operations that cannot be fused (e.g., `Gather` or simple `Add` with complex broadcasting) are compiled as standalone CUDA kernels (`KernelOps`).

#### 3. Runtime (`runtime.rs`)
*   **Partitioning:** The runtime splits the graph into "convex" subgraphs suitable for the Megakernel and standalone nodes.
*   **NVRTC:** Uses NVIDIA Runtime Compilation to JIT compile the interpreter with the specific operations required by the graph.

### CPU Reference
*   **`op.rs`:** Every operator implements a `process` method. This is a slow, reference implementation used for debugging (`cx.execute_debug()`) or when no GPU is available. It uses `ShapeTracker` to handle views without copying data.

