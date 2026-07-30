#ifndef B_LINALG_H
#define B_LINALG_H

#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

Tensor* Tensor_MatMul(
    const Tensor* a,
    const Tensor* b);

#ifdef __cplusplus
}
#endif

#endif /* B_LINALG_H */