#ifndef B_REDUCTION_H
#define B_REDUCTION_H

#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * Reduction Axis
 *--------------------------------------------------------------------------*/

/* Reduce over all elements */
#define B_ALL_AXIS (-999)

/*--------------------------------------------------------------------------
 * Reduction Operations
 *--------------------------------------------------------------------------*/

/* Sum */
Tensor* Tensor_Sum(
    const Tensor* tensor,
    int axis);

/* Mean */
Tensor* Tensor_Mean(
    const Tensor* tensor,
    int axis);

/* Minimum */
Tensor* Tensor_Min(
    const Tensor* tensor,
    int axis);

/* Maximum */
Tensor* Tensor_Max(
    const Tensor* tensor,
    int axis);

#ifdef __cplusplus
}
#endif

#endif /* B_REDUCTION_H */