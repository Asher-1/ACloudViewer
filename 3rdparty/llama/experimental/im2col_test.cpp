// Exact gather oracle for the GPU image-stem optimizations and their fallbacks.
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <vector>

struct Shape {
    int width, height, channels, kw, kh, sx, sy, px, py, dx, dy;
    bool is_2d = true;
};

static bool check(ggml_backend_t backend, ggml_type type, Shape s,
                  int batches, bool padded_storage) {
    const int ow = (s.width + 2*s.px - s.dx*(s.kw-1) - 1)/s.sx + 1;
    const int oh = (s.height + 2*s.py - s.dy*(s.kh-1) - 1)/s.sy + 1;
    const size_t cs = size_t(s.width)*(s.height + (padded_storage ? 1 : 0));
    const size_t bs = cs*(s.channels + (padded_storage ? 1 : 0));
    ggml_init_params init{1024*1024, nullptr, true};
    ggml_context* ctx = ggml_init(init);
    if (!ctx) return false;
    ggml_tensor* storage = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, bs*batches);
    ggml_tensor* input = s.is_2d
        ? ggml_view_4d(ctx, storage, s.width, s.height, s.channels, batches,
                       s.width*sizeof(float), cs*sizeof(float), bs*sizeof(float), 0)
        : ggml_view_3d(ctx, storage, s.width, s.channels, batches,
                       cs*sizeof(float), bs*sizeof(float), 0);
    ggml_tensor* kernel = s.is_2d
        ? ggml_new_tensor_4d(ctx, GGML_TYPE_F32, s.kw, s.kh, s.channels, 1)
        : ggml_new_tensor_3d(ctx, GGML_TYPE_F32, s.kw, s.channels, 1);
    ggml_tensor* output = ggml_im2col(ctx, kernel, input, s.sx, s.sy,
                                     s.px, s.py, s.dx, s.dy, s.is_2d, type);
    ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    const bool supported = ggml_backend_supports_op(backend, output);
    // Both GPU contracts reject non-contiguous inputs. Verify that the new
    // kernels do not broaden dispatch to an unsupported layout.
    if (supported != !padded_storage) {
        std::fprintf(stderr, "unexpected GPU im2col layout support\n");
        ggml_free(ctx);
        return false;
    }
    if (!supported) {
        std::printf("{\"type\":\"%s\",\"width\":%d,\"height\":%d,\"channels\":%d,"
                    "\"batches\":%d,\"padded_storage\":true,\"supported\":false,\"pass\":true}\n",
                    ggml_type_name(type), s.width, s.height, s.channels, batches);
        ggml_free(ctx);
        return true;
    }
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        ggml_free(ctx);
        return false;
    }
    std::vector<float> values(bs*batches);
    for (size_t i = 0; i < values.size(); ++i)
        values[i] = (int((i*7 + i/17)%3001)-1500)/1001.0f;
    ggml_backend_tensor_set(storage, values.data(), 0, values.size()*sizeof(float));
    bool valid = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    const size_t size = size_t(ow)*oh*s.channels*s.kw*s.kh*batches;
    std::vector<float> actual(size);
    if (type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> half(size);
        ggml_backend_tensor_get(output, half.data(), 0, half.size()*sizeof(half[0]));
        for (size_t i = 0; i < size; ++i) actual[i] = ggml_fp16_to_fp32(half[i]);
    } else {
        ggml_backend_tensor_get(output, actual.data(), 0, actual.size()*sizeof(float));
    }
    // Walk window coordinates rather than decode the kernel's flattened index.
    size_t mismatches = 0;
    for (int batch = 0; batch < batches; ++batch)
        for (int channel = 0; channel < s.channels; ++channel)
            for (int ky = 0; ky < s.kh; ++ky)
                for (int kx = 0; kx < s.kw; ++kx)
                    for (int y = 0; y < oh; ++y)
                        for (int x = 0; x < ow; ++x) {
                            const int iy = y*s.sy + ky*s.dy - s.py;
                            const int ix = x*s.sx + kx*s.dx - s.px;
                            float expected = 0;
                            if (iy >= 0 && iy < s.height && ix >= 0 && ix < s.width)
                                expected = values[batch*bs + channel*cs + iy*s.width + ix];
                            if (type == GGML_TYPE_F16)
                                expected = ggml_fp16_to_fp32(ggml_fp32_to_fp16(expected));
                            const size_t index = (((size_t(batch)*oh + y)*ow + x)*
                                s.channels + channel)*s.kh*s.kw + ky*s.kw + kx;
                            mismatches += actual[index] != expected;
                        }
    valid &= mismatches == 0;
    std::printf("{\"type\":\"%s\",\"width\":%d,\"height\":%d,\"channels\":%d,"
                "\"batches\":%d,\"padded_storage\":%s,\"mismatches\":%zu,\"pass\":%s}\n",
                ggml_type_name(type), s.width, s.height, s.channels, batches,
                padded_storage ? "true" : "false", mismatches, valid ? "true" : "false");
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return valid;
}

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    ggml_backend_load_all_from_path(argv[1]);
    ggml_backend_t backend = ggml_backend_init_by_name(argv[2], nullptr);
    if (!backend) return 1;
    bool valid = true;
    for (ggml_type type : {GGML_TYPE_F32, GGML_TYPE_F16})
        for (int batches : {1, 3})
            for (bool padded : {false, true})
                for (Shape shape : {
                    Shape{512,256,3,3,3,2,2,1,1,1,1},
                    Shape{192,128,16,1,1,1,1,0,0,1,1},
                    Shape{192,128,3,3,2,1,1,2,1,2,2},
                    Shape{192,128,33,1,1,1,1,0,0,1,1},
                    Shape{19,11,3,3,2,2,1,1,0,1,2},
                    Shape{32768,1,3,3,1,1,1,2,0,2,1,false}})
                    valid = check(backend, type, shape, batches, padded) && valid;
    ggml_backend_free(backend);
    return valid ? 0 : 1;
}
