#pragma once

namespace cloudViewer {
namespace ml {
namespace contrib {

__global__ void ball_query_kernel(int b,
                                  int n,
                                  int m,
                                  float radius,
                                  int nsample,
                                  const float *__restrict__ new_xyz,
                                  const float *__restrict__ xyz,
                                  int *__restrict__ idx);

}  // namespace contrib
}  // namespace ml
}  // namespace cloudViewer
