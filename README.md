# BananaAI 🍌

A lightweight, high-performance **Tensor and Autograd Engine** built completely from scratch in pure C. Designed to explore the absolute low-level fundamentals of deep learning frameworks like PyTorch.

## 🚀 Key Features

* **Custom Storage Architecture:** Implemented manual reference counting via `Storage_Retain` and `Storage_Release` to manage dynamic memory layouts efficiently without leaks.
* **Strides & Offset Navigation:** Supports zero-copy tensor views and slicing by computing strides across contiguous 1D float arrays.
* **Computational Graph & Nodes:** Designed `B_Node` structures ready for topological sorting to enable full automatic differentiation (Autograd).

## 🛠️ Code Structure

* `tensor.h` / `tensor.c` - Core multi-dimensional array management.
* `banana_storage.h` / `banana_storage.c` - Reference-counted memory blocks.
* `node.h` / `node.c` - Computational graph configurations.
* `matrix.h` / `operations.h` - Optimized linear algebra and activation functions.
