#pragma once

namespace cloudViewer {
namespace ml {
namespace contrib {

__global__ void three_nn_kernel(int b,
                                int n,
                                int m,
                                const float *__restrict__ unknown,
                                const float *__restrict__ known,
                                float *__restrict__ dist2,
                                int *__restrict__ idx);

__global__ void three_interpolate_kernel(int b,
                                         int c,
                                         int m,
                                         int n,
                                         const float *__restrict__ points,
                                         const int *__restrict__ idx,
                                         const float *__restrict__ weight,
                                         float *__restrict__ out);

__global__ void three_interpolate_grad_kernel(
        int b,
        int c,
        int n,
        int m,
        const float *__restrict__ grad_out,
        const int *__restrict__ idx,
        const float *__restrict__ weight,
        float *__restrict__ grad_points);

}  // namespace contrib
}  // namespace ml
}  // namespace cloudViewer
