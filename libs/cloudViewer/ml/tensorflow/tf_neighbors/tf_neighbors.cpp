// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ml/contrib/neighbors.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;
using namespace cloudViewer::ml::contrib;

REGISTER_OP("CloudviewerOrderedNeighbors")
        .Input("queries: float")
        .Input("supports: float")
        .Input("radius: float")
        .Output("neighbors: int32")
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            ::tensorflow::shape_inference::ShapeHandle input;
            TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input));
            c->set_output(0, input);
            return Status::OK();
        });

class OrderedNeighborsOp : public OpKernel {
public:
    explicit OrderedNeighborsOp(OpKernelConstruction* context)
        : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        // Grab the input tensors
        const Tensor& queries_tensor = context->input(0);
        const Tensor& supports_tensor = context->input(1);
        const Tensor& radius_tensor = context->input(2);

        // check shapes of input and weights
        const TensorShape& queries_shape = queries_tensor.shape();
        const TensorShape& supports_shape = supports_tensor.shape();

        // check input are [N x 3] matrices
        DCHECK_EQ(queries_shape.dims(), 2);
        DCHECK_EQ(queries_shape.dim_size(1), 3);
        DCHECK_EQ(supports_shape.dims(), 2);
        DCHECK_EQ(supports_shape.dim_size(1), 3);

        // Dimensions
        int Nq = (int)queries_shape.dim_size(0);
        int Ns = (int)supports_shape.dim_size(0);

        // get the data as std vector of points
        float radius = radius_tensor.flat<float>().data()[0];
        std::vector<PointXYZ> queries = std::vector<PointXYZ>(
                (PointXYZ*)queries_tensor.flat<float>().data(),
                (PointXYZ*)queries_tensor.flat<float>().data() + Nq);

        std::vector<PointXYZ> supports = std::vector<PointXYZ>(
                (PointXYZ*)supports_tensor.flat<float>().data(),
                (PointXYZ*)supports_tensor.flat<float>().data() + Ns);

        // Create result containers
        std::vector<int> neighbors_indices;

        // Compute results
        ordered_neighbors(queries, supports, neighbors_indices, radius);

        // Maximal number of neighbors
        int max_neighbors = neighbors_indices.size() / Nq;

        // create output shape
        TensorShape output_shape;
        output_shape.AddDim(Nq);
        output_shape.AddDim(max_neighbors);

        // create output tensor
        Tensor* output = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, output_shape, &output));
        auto output_tensor = output->matrix<int>();

        // Fill output tensor
        for (int i = 0; i < output->shape().dim_size(0); i++) {
            for (int j = 0; j < output->shape().dim_size(1); j++) {
                output_tensor(i, j) = neighbors_indices[max_neighbors * i + j];
            }
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("CloudviewerOrderedNeighbors").Device(DEVICE_CPU),
                        OrderedNeighborsOp);
