#include "tasks/loma/descriptor.hpp"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <ggml.h>
#include <gguf.h>

#include <cmath>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "tasks/lightglue/backend.hpp"

namespace aicore::loma {
namespace {
constexpr size_t kMaxGraphNodes = 8192;
constexpr int32_t kImageSize = 784;
constexpr int32_t kDescriptorDimB = 128;
constexpr int32_t kDescriptorDimG = 256;

class File {
public:
    ~File() { Close(); }
    bool Open(const std::string& path) {
        Close(); error.clear();
        gguf_init_params p{/*no_alloc=*/false, &ctx}; gguf = gguf_init_from_file(path.c_str(), p);
        if (!gguf || !ctx) { error = "failed to read DeDoDe GGUF: " + path; Close(); return false; }
        if (Str("general.architecture") != "loma") {
            if (error.empty()) error = "GGUF is not a LoMa descriptor";
            Close(); return false;
        }
        variant = Str("loma.variant");
        if (variant != "descriptor_dedode_b" && variant != "descriptor_dedode_g") {
            if (error.empty()) error = "GGUF is not a pinned LoMa DeDoDe descriptor";
            Close(); return false;
        }
        return error.empty();
    }
    void Close() { device_tensors.clear(); if (ctx) { ggml_free(ctx); ctx=nullptr; } if (gguf) { gguf_free(gguf); gguf=nullptr; } }
    ggml_tensor* Req(const std::string& n) { auto device=device_tensors.find(n);if(device!=device_tensors.end())return device->second;auto* t=ctx?ggml_get_tensor(ctx,n.c_str()):nullptr; if (!t&&error.empty()) error="DeDoDe GGUF missing tensor: "+n; return t; }
    int64_t N() const { return gguf_get_n_tensors(gguf); }
    const char* Name(int64_t i) const { return gguf_get_tensor_name(gguf,i); }
    ggml_tensor* Tensor(int64_t i) const { return ctx?ggml_get_tensor(ctx,Name(i)):nullptr; }
    void SetDeviceTensors(std::unordered_map<std::string,ggml_tensor*> tensors) { device_tensors=std::move(tensors); }
    std::string Str(const char* key) { auto i=gguf_find_key(gguf,key); if(i<0||gguf_get_kv_type(gguf,i)!=GGUF_TYPE_STRING){if(error.empty())error=std::string("DeDoDe GGUF missing key: ")+key;return{};} return gguf_get_val_str(gguf,i); }
    int32_t U32(const char* key) { auto i=gguf_find_key(gguf,key); if(i<0||gguf_get_kv_type(gguf,i)!=GGUF_TYPE_UINT32){if(error.empty())error=std::string("DeDoDe GGUF missing uint32 key: ")+key;return 0;} return static_cast<int32_t>(gguf_get_val_u32(gguf,i)); }
    float F32(const char* key) { auto i=gguf_find_key(gguf,key); if(i<0||gguf_get_kv_type(gguf,i)!=GGUF_TYPE_FLOAT32){if(error.empty())error=std::string("DeDoDe GGUF missing float key: ")+key;return 0.f;} return gguf_get_val_f32(gguf,i); }
    std::vector<std::string> Strings(const char* key) { auto i=gguf_find_key(gguf,key); if(i<0||gguf_get_kv_type(gguf,i)!=GGUF_TYPE_ARRAY||gguf_get_arr_type(gguf,i)!=GGUF_TYPE_STRING){if(error.empty())error=std::string("DeDoDe GGUF missing string-array key: ")+key;return{};} std::vector<std::string> v;const auto n=gguf_get_arr_n(gguf,i);v.reserve(n);for(size_t j=0;j<n;++j)v.emplace_back(gguf_get_arr_str(gguf,i,j));return v; }
    gguf_context* gguf=nullptr; ggml_context* ctx=nullptr; std::unordered_map<std::string,ggml_tensor*> device_tensors; std::string error; std::string variant;
};

ggml_tensor* Bias(ggml_context* c, ggml_tensor* x, ggml_tensor* b) {
    return ggml_add(c,x,ggml_reshape_4d(c,b,1,1,b->ne[0],1));
}
ggml_tensor* Slice(ggml_context* c, ggml_tensor* x, int64_t off, int64_t n) {
    return ggml_cont(c,ggml_view_4d(c,x,x->ne[0],x->ne[1],n,x->ne[3],x->nb[1],x->nb[2],x->nb[3],static_cast<size_t>(off)*x->nb[2]));
}
struct Build {
    ggml_context* c; File* f; bool use_accelerator_convolution;
    ggml_tensor* Conv(ggml_tensor* x,const std::string& p,int pad,bool dw=false) {
        auto*w=f->Req("loma."+p+".weight");auto*b=f->Req("loma."+p+".bias");if(!w||!b)return nullptr;
        return Bias(c,dw?ggml_conv_2d_dw_direct(c,w,x,1,1,pad,pad,1,1):use_accelerator_convolution?ggml_conv_2d(c,w,x,1,1,pad,pad,1,1):ggml_conv_2d_direct(c,w,x,1,1,pad,pad,1,1),b);
    }
    ggml_tensor* Ref(ggml_tensor* feat,ggml_tensor* ctx,const char* scale) {
        auto*x=ctx?ggml_concat(c,feat,ctx,2):feat;std::string r=std::string("desc.decoder.layers.")+scale;
        x=Conv(x,r+".block1.0",0);if(!x)return nullptr;x=ggml_relu(c,x);x=Conv(x,r+".block1.3",0);if(!x)return nullptr;auto*res=x;
        for(int i=0;i<5;++i){auto p=r+".hidden_blocks."+std::to_string(i);x=Conv(x,p+".0",2,true);if(!x)return nullptr;x=ggml_relu(c,x);x=Conv(x,p+".3",0);if(!x)return nullptr;}
        return Conv(ggml_scale(c,ggml_add(c,x,res),1.0f/1.4f),r+".out_conv",0);
    }
};

ggml_tensor* Linear(ggml_context* c, ggml_tensor* x, ggml_tensor* w,
                    ggml_tensor* bias) {
    auto* y=ggml_mul_mat(c,w,x);ggml_mul_mat_set_prec(y,GGML_PREC_F32);
    return bias?ggml_add(c,y,bias):y;
}
ggml_tensor* Norm(ggml_context* c, ggml_tensor* x, ggml_tensor* w,
                  ggml_tensor* bias, float epsilon) {
    return ggml_add(c,ggml_mul(c,ggml_norm(c,x,epsilon),w),bias);
}
ggml_tensor* DinoAttention(ggml_context* c, ggml_tensor* x, ggml_tensor* qkv,
                            ggml_tensor* qkv_bias, ggml_tensor* projection,
                            ggml_tensor* projection_bias, int tokens,
                            bool exact_f32_attention) {
    auto* packed=Linear(c,x,qkv,qkv_bias);
    // DINOv2's ONNX MatMul output is Q[1024] | K[1024] | V[1024] for each
    // token, unlike LightGlue's interleaved QKV packing.
    auto component=[&](int index){auto* part=ggml_cont(c,ggml_view_2d(c,packed,
            1024,tokens,packed->nb[1],static_cast<size_t>(index)*1024*packed->nb[0]));
        return ggml_cont(c,ggml_permute(c,ggml_reshape_3d(c,part,64,16,tokens),0,2,1,3));};
    auto* q=component(0);auto* k=component(1);auto* v=component(2);
    ggml_tensor* attention = nullptr;
    if (exact_f32_attention) {
        // CUDA's fused path converts F32 K/V to F16 despite the precision
        // request. Keep the DINOv2 ViT-L reduction in F32 until that kernel
        // exposes an equivalent F32 implementation.
        auto* scores = ggml_mul_mat(c, k, q);
        ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
        scores = ggml_soft_max_ext(c, scores, nullptr, 1.0f / 8.0f, 0.0f);
        auto* values = ggml_cont(c, ggml_permute(c, v, 1, 0, 2, 3));
        attention = ggml_mul_mat(c, values, scores);
        ggml_mul_mat_set_prec(attention, GGML_PREC_F32);
        attention = ggml_cont(c, ggml_permute(c, attention, 0, 2, 1, 3));
    } else {
        attention = ggml_flash_attn_ext(c, q, k, v, nullptr, 1.0f / 8.0f,
                                        0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(attention, GGML_PREC_F32);
    }
    return Linear(c,ggml_reshape_2d(c,attention,1024,tokens),projection,
                  projection_bias);
}

struct DinoContract {
    int32_t depth=0,heads=0,dimension=0,patch_size=0;float epsilon=0.f;
    std::vector<std::string> qkv,projection,fc1,fc2;
    bool Load(File* f) {
        dimension=f->U32("loma.descriptor_g.embedding_dimension");
        heads=f->U32("loma.descriptor_g.attention.head_count");
        depth=f->U32("loma.descriptor_g.block_count");
        patch_size=f->U32("loma.descriptor_g.patch_size");
        epsilon=f->F32("loma.descriptor_g.layer_norm_epsilon");
        qkv=f->Strings("loma.descriptor_g.qkv_weights");
        projection=f->Strings("loma.descriptor_g.attention_output_weights");
        fc1=f->Strings("loma.descriptor_g.mlp_fc1_weights");
        fc2=f->Strings("loma.descriptor_g.mlp_fc2_weights");
        return f->error.empty() && dimension==1024 && heads==16 && depth==24 &&
               patch_size==14 && epsilon==1e-6f && qkv.size()==24 &&
               projection.size()==24 && fc1.size()==24 && fc2.size()==24;
    }
};

std::vector<float> Whcn(const aicore_loma_rgb_image& im) {
    const size_t plane=static_cast<size_t>(im.width)*im.height;std::vector<float> x(plane*3);
    for(int y=0;y<im.height;++y){auto*r=im.rgb+static_cast<size_t>(y)*im.row_stride_bytes;for(int xx=0;xx<im.width;++xx)for(int ch=0;ch<3;++ch)x[xx+static_cast<size_t>(y)*im.width+static_cast<size_t>(ch)*plane]=r[3*xx+ch]/255.0f;} return x;
}
float Sample(const std::vector<float>& g,int w,int h,int c,float nx,float ny) {
    const float x=((nx+1)*w-1)*.5f,y=((ny+1)*h-1)*.5f;const int x0=static_cast<int>(std::floor(x)),y0=static_cast<int>(std::floor(y)),x1=x0+1,y1=y0+1;
    auto at=[&](int xx,int yy){return xx>=0&&xx<w&&yy>=0&&yy<h?g[static_cast<size_t>(xx)+static_cast<size_t>(yy)*w+static_cast<size_t>(c)*w*h]:0.f;};
    const float fx=x-x0,fy=y-y0;return (1-fy)*((1-fx)*at(x0,y0)+fx*at(x1,y0))+fy*((1-fx)*at(x0,y1)+fx*at(x1,y1));
}
}

class Descriptor::Impl {
public:
    ~Impl(){Release();}
    bool Load(const std::string& path,const DescriptorOptions&o){Release();opt=o;if(!file.Open(path)){err=file.error;return false;}is_g=file.variant=="descriptor_dedode_g";if(is_g&&!dino.Load(&file)){err=file.error.empty()?"invalid DeDoDe-G DINOv2 contract":file.error;return false;}if(!be.init(opt.device,opt.num_threads)){err=be.error;return false;}if(be.is_cpu()){weights=ggml_backend_cpu_buffer_from_ptr(ggml_get_mem_buffer(file.ctx),ggml_get_mem_size(file.ctx));if(!weights){err="failed to bind DeDoDe weights";return false;}for(int64_t i=0;i<file.N();++i)file.Tensor(i)->buffer=weights;return true;}return OffloadWeights();}
    bool OffloadWeights(){auto lock=be.lock();const int64_t count=file.N();ggml_init_params p{};p.mem_size=ggml_tensor_overhead()*static_cast<size_t>(count+8);p.no_alloc=true;device_context=ggml_init(p);if(!device_context){err="failed to create DeDoDe device-weight context";return false;}std::unordered_map<std::string,ggml_tensor*> tensors;tensors.reserve(static_cast<size_t>(count));std::vector<std::pair<ggml_tensor*,const void*>> uploads;uploads.reserve(static_cast<size_t>(count));std::vector<std::vector<float>> decoded_weights;decoded_weights.reserve(static_cast<size_t>(count));for(int64_t i=0;i<count;++i){auto*host=file.Tensor(i);if(!host){err="DeDoDe GGUF tensor table is inconsistent";return false;}const bool decode_to_f32=host->type==GGML_TYPE_F16||ggml_is_quantized(host->type);auto*device=ggml_new_tensor(device_context,decode_to_f32?GGML_TYPE_F32:host->type,GGML_MAX_DIMS,host->ne);if(!device){err=std::string("failed to create DeDoDe device tensor: ")+host->name;return false;}ggml_set_name(device,host->name);tensors.emplace(file.Name(i),device);if(decode_to_f32){const ggml_type_traits*traits=ggml_get_type_traits(host->type);const int64_t elements=ggml_nelements(host);if(!traits||!traits->to_float||elements<=0){err=std::string("cannot decode DeDoDe weight for GPU: ")+host->name;return false;}auto&decoded=decoded_weights.emplace_back();decoded.resize(static_cast<size_t>(elements));traits->to_float(host->data,decoded.data(),elements);uploads.emplace_back(device,decoded.data());}else{uploads.emplace_back(device,host->data);}}device_weights=ggml_backend_alloc_ctx_tensors(device_context,be.be);if(!device_weights){err="failed to allocate DeDoDe weights on requested ggml backend";return false;}for(const auto&upload:uploads)ggml_backend_tensor_set(upload.first,upload.second,0,ggml_nbytes(upload.first));ggml_backend_synchronize(be.be);file.SetDeviceTensors(std::move(tensors));return true;}
    bool RunB(const aicore_loma_rgb_image& im,const aicore_loma_keypoint* k,int n,int sw,int sh,std::vector<float>*out){
        err.clear();out->clear();if(!k||n<=0||sw<=0||sh<=0||im.rgb==nullptr||im.width!=kImageSize||im.height!=kImageSize||im.row_stride_bytes<im.width*3){err="DeDoDe-B expects a 784x784 RGB input and non-empty source-frame keypoints";return false;}
        auto lock=be.lock();ggml_init_params p{ggml_tensor_overhead()*kMaxGraphNodes+ggml_graph_overhead_custom(kMaxGraphNodes,false),nullptr,true};auto*c=ggml_init(p);if(!c){err="failed to create DeDoDe graph";return false;}auto cleanup=[&]{ggml_free(c);};
        auto*in=ggml_new_tensor_4d(c,GGML_TYPE_F32,im.width,im.height,3,1);Build b{c,&file,!be.is_cpu()};auto*x=b.Conv(in,"desc.encoder.layers.0",1);x=x?ggml_relu(c,x):nullptr;x=x?b.Conv(x,"desc.encoder.layers.3",1):nullptr;x=x?ggml_relu(c,x):nullptr;auto*f1=x;x=x?ggml_pool_2d(c,x,GGML_OP_POOL_MAX,2,2,2,2,0,0):nullptr;
        x=x?b.Conv(x,"desc.encoder.layers.7",1):nullptr;x=x?ggml_relu(c,x):nullptr;x=x?b.Conv(x,"desc.encoder.layers.10",1):nullptr;x=x?ggml_relu(c,x):nullptr;auto*f2=x;x=x?ggml_pool_2d(c,x,GGML_OP_POOL_MAX,2,2,2,2,0,0):nullptr;
        for(int layer: {14,17,20,23}){x=x?b.Conv(x,"desc.encoder.layers."+std::to_string(layer),1):nullptr;x=x?ggml_relu(c,x):nullptr;}auto*f4=x;x=x?ggml_pool_2d(c,x,GGML_OP_POOL_MAX,2,2,2,2,0,0):nullptr;
        for(int layer: {27,30,33,36}){x=x?b.Conv(x,"desc.encoder.layers."+std::to_string(layer),1):nullptr;x=x?ggml_relu(c,x):nullptr;}if(!x||!file.error.empty()){err=file.error.empty()?"failed to build DeDoDe encoder":file.error;cleanup();return false;}
        auto*d8=b.Ref(x,nullptr,"8");if(!d8){err=file.error;cleanup();return false;}auto*desc=Slice(c,d8,0,kDescriptorDimB);auto*ctx=Slice(c,d8,kDescriptorDimB,256);desc=ggml_interpolate(c,desc,f4->ne[0],f4->ne[1],kDescriptorDimB,1,GGML_SCALE_MODE_BILINEAR);ctx=ggml_interpolate(c,ctx,f4->ne[0],f4->ne[1],256,1,GGML_SCALE_MODE_BILINEAR);
        auto*d4=b.Ref(f4,ctx,"4");if(!d4){err=file.error;cleanup();return false;}desc=ggml_add(c,desc,Slice(c,d4,0,kDescriptorDimB));ctx=Slice(c,d4,kDescriptorDimB,128);desc=ggml_interpolate(c,desc,f2->ne[0],f2->ne[1],kDescriptorDimB,1,GGML_SCALE_MODE_BILINEAR);ctx=ggml_interpolate(c,ctx,f2->ne[0],f2->ne[1],128,1,GGML_SCALE_MODE_BILINEAR);
        auto*d2=b.Ref(f2,ctx,"2");if(!d2){err=file.error;cleanup();return false;}desc=ggml_add(c,desc,Slice(c,d2,0,kDescriptorDimB));ctx=Slice(c,d2,kDescriptorDimB,32);desc=ggml_interpolate(c,desc,f1->ne[0],f1->ne[1],kDescriptorDimB,1,GGML_SCALE_MODE_BILINEAR);ctx=ggml_interpolate(c,ctx,f1->ne[0],f1->ne[1],32,1,GGML_SCALE_MODE_BILINEAR);
        auto*d1=b.Ref(f1,ctx,"1");if(!d1){err=file.error;cleanup();return false;}desc=ggml_add(c,desc,Slice(c,d1,0,kDescriptorDimB));ggml_set_output(desc);auto*g=ggml_new_graph_custom(c,kMaxGraphNodes,false);ggml_build_forward_expand(g,desc);if(!ggml_gallocr_alloc_graph(be.galloc,g)){err="failed to allocate DeDoDe graph";cleanup();return false;}auto pixels=Whcn(im);ggml_backend_tensor_set(in,pixels.data(),0,pixels.size()*sizeof(float));if(ggml_backend_graph_compute(be.be,g)!=GGML_STATUS_SUCCESS){err="DeDoDe graph compute failed";cleanup();return false;}std::vector<float> grid(static_cast<size_t>(im.width)*im.height*kDescriptorDimB);ggml_backend_tensor_get(desc,grid.data(),0,grid.size()*sizeof(float));cleanup();out->resize(static_cast<size_t>(n)*kDescriptorDimB);for(int i=0;i<n;++i){float nx=2*k[i].x/sw-1,ny=2*k[i].y/sh-1;for(int d=0;d<kDescriptorDimB;++d)(*out)[static_cast<size_t>(i)*kDescriptorDimB+d]=Sample(grid,im.width,im.height,d,nx,ny);}return true;
    }
    bool RunG(const aicore_loma_rgb_image& im,const aicore_loma_keypoint* k,int n,int sw,int sh,std::vector<float>*out){
        err.clear();out->clear();if(!k||n<=0||sw<=0||sh<=0||im.rgb==nullptr||im.width!=kImageSize||im.height!=kImageSize||im.row_stride_bytes<im.width*3){err="DeDoDe-G expects a 784x784 RGB input and non-empty source-frame keypoints";return false;}
        auto lock=be.lock();ggml_init_params p{ggml_tensor_overhead()*kMaxGraphNodes+ggml_graph_overhead_custom(kMaxGraphNodes,false),nullptr,true};auto*c=ggml_init(p);if(!c){err="failed to create DeDoDe-G graph";return false;}auto cleanup=[&]{ggml_free(c);};
        auto*in=ggml_new_tensor_4d(c,GGML_TYPE_F32,im.width,im.height,3,1);Build b{c,&file,!be.is_cpu()};auto*x=b.Conv(in,"desc.encoder.vgg.layers.0",1);x=x?ggml_relu(c,x):nullptr;x=x?b.Conv(x,"desc.encoder.vgg.layers.3",1):nullptr;x=x?ggml_relu(c,x):nullptr;auto*f1=x;x=x?ggml_pool_2d(c,x,GGML_OP_POOL_MAX,2,2,2,2,0,0):nullptr;
        x=x?b.Conv(x,"desc.encoder.vgg.layers.7",1):nullptr;x=x?ggml_relu(c,x):nullptr;x=x?b.Conv(x,"desc.encoder.vgg.layers.10",1):nullptr;x=x?ggml_relu(c,x):nullptr;auto*f2=x;x=x?ggml_pool_2d(c,x,GGML_OP_POOL_MAX,2,2,2,2,0,0):nullptr;
        for(int layer: {14,17,20,23}){x=x?b.Conv(x,"desc.encoder.vgg.layers."+std::to_string(layer),1):nullptr;x=x?ggml_relu(c,x):nullptr;}auto*f4=x;x=x?ggml_pool_2d(c,x,GGML_OP_POOL_MAX,2,2,2,2,0,0):nullptr;
        for(int layer: {27,30,33,36}){x=x?b.Conv(x,"desc.encoder.vgg.layers."+std::to_string(layer),1):nullptr;x=x?ggml_relu(c,x):nullptr;}auto*f8=x;if(!x||!file.error.empty()){err=file.error.empty()?"failed to build DeDoDe-G VGG encoder":file.error;cleanup();return false;}
        auto*pew=file.Req("loma.desc.encoder.frozen_dinov2.dinov2_vitl14.patch_embed.proj.weight");auto*peb=file.Req("loma.desc.encoder.frozen_dinov2.dinov2_vitl14.patch_embed.proj.bias");auto*repeat=file.Req("loma.repeat");auto*pos=file.Req("loma.desc.encoder.frozen_dinov2.dinov2_vitl14.pos_embed");auto*pos56=file.Req("loma.descriptor_g.pos_embed_56x56");if(!pew||!peb||!repeat||!pos||!pos56){err=file.error;cleanup();return false;}if(pos56->ne[0]!=1024||pos56->ne[1]!=56||pos56->ne[2]!=56||pos56->ne[3]!=1){err="invalid DeDoDe-G 56x56 position-embedding tensor";cleanup();return false;}
        const int T=56*56+1;auto trace_tokens=[&](ggml_tensor* trace){ggml_set_output(trace);auto*g=ggml_new_graph_custom(c,kMaxGraphNodes,false);ggml_build_forward_expand(g,trace);if(!ggml_gallocr_alloc_graph(be.galloc,g)){err="failed to allocate DeDoDe-G DINO trace graph";return false;}auto pixels=Whcn(im);ggml_backend_tensor_set(in,pixels.data(),0,pixels.size()*sizeof(float));if(ggml_backend_graph_compute(be.be,g)!=GGML_STATUS_SUCCESS){err="DeDoDe-G DINO trace graph compute failed";return false;}out->resize(static_cast<size_t>(T)*dino.dimension);ggml_backend_tensor_get(trace,out->data(),0,out->size()*sizeof(float));return true;};
        // The generic im2col/GEMM lowering changes F32 reduction order before
        // the ViT. Use ggml's direct F32 convolution on every backend so the
        // token path starts from the same calibrated patch embedding.
        auto*patch=Bias(c,ggml_conv_2d_direct(c,pew,in,14,14,0,0,1,1),peb);auto*tokens=ggml_cont(c,ggml_permute(c,ggml_reshape_3d(c,patch,56*56,1024,1),1,0,2,3));auto*cls=ggml_reshape_2d(c,repeat,1024,1);tokens=ggml_concat(c,cls,tokens,1);if(opt.trace_g_tokens_for_validation&&opt.trace_g_block_for_validation==-3){const bool ok=trace_tokens(tokens);cleanup();return ok;}
        // DINOv2 uses scale_factor=56.1/37 for bicubic position encoding. The
        // exact fixed 56x56 result is generated with the GGUF so ggml does not
        // substitute its integral 56/37 scale at runtime.
        auto*pos_cls=ggml_cont(c,ggml_view_2d(c,pos,1024,1,pos->nb[1],0));tokens=ggml_add(c,tokens,ggml_concat(c,pos_cls,ggml_reshape_2d(c,pos56,1024,56*56),1));if(opt.trace_g_tokens_for_validation&&opt.trace_g_block_for_validation==-2){const bool ok=trace_tokens(tokens);cleanup();return ok;}
        // The F32 materialized graph is the canonical DINOv2 path on every
        // backend. It avoids comparing CUDA's exact reductions with the CPU
        // flash kernel's different online-softmax reduction order.
        const bool exact_f32_attention = true;
        for(int i=0;i<dino.depth;++i){const std::string pfx="desc.encoder.frozen_dinov2.dinov2_vitl14.blocks."+std::to_string(i)+".";auto*n1w=file.Req("loma."+pfx+"norm1.weight");auto*n1b=file.Req("loma."+pfx+"norm1.bias");auto*qb=file.Req("loma."+pfx+"attn.qkv.bias");auto*ob=file.Req("loma."+pfx+"attn.proj.bias");auto*ls1=file.Req("loma."+pfx+"ls1.gamma");auto*n2w=file.Req("loma."+pfx+"norm2.weight");auto*n2b=file.Req("loma."+pfx+"norm2.bias");auto*f1b=file.Req("loma."+pfx+"mlp.fc1.bias");auto*f2b=file.Req("loma."+pfx+"mlp.fc2.bias");auto*ls2=file.Req("loma."+pfx+"ls2.gamma");auto*qkv=file.Req(dino.qkv[i]);auto*proj=file.Req(dino.projection[i]);auto*fc1=file.Req(dino.fc1[i]);auto*fc2=file.Req(dino.fc2[i]);if(!n1w||!n1b||!qb||!ob||!ls1||!n2w||!n2b||!f1b||!f2b||!ls2||!qkv||!proj||!fc1||!fc2){err=file.error;cleanup();return false;}auto*attn=DinoAttention(c,Norm(c,tokens,n1w,n1b,dino.epsilon),qkv,qb,proj,ob,T,exact_f32_attention);tokens=ggml_add(c,tokens,ggml_mul(c,attn,ls1));auto*mlp=Linear(c,ggml_gelu_erf(c,Linear(c,Norm(c,tokens,n2w,n2b,dino.epsilon),fc1,f1b)),fc2,f2b);tokens=ggml_add(c,tokens,ggml_mul(c,mlp,ls2));if(opt.trace_g_tokens_for_validation&&opt.trace_g_block_for_validation==i){const bool ok=trace_tokens(tokens);cleanup();return ok;}}
        auto*fnw=file.Req("loma.desc.encoder.frozen_dinov2.dinov2_vitl14.norm.weight");auto*fnb=file.Req("loma.desc.encoder.frozen_dinov2.dinov2_vitl14.norm.bias");if(!fnw||!fnb){err=file.error;cleanup();return false;}tokens=Norm(c,tokens,fnw,fnb,dino.epsilon);if(opt.trace_g_tokens_for_validation){const bool ok=trace_tokens(tokens);cleanup();return ok;}auto*patches=ggml_cont(c,ggml_view_2d(c,tokens,1024,56*56,tokens->nb[1],tokens->nb[1]));auto*dino_map=ggml_cont(c,ggml_permute(c,ggml_reshape_3d(c,patches,1024,56,56),2,0,1,3));
        auto*d14=b.Ref(dino_map,nullptr,"14");if(!d14){err=file.error;cleanup();return false;}auto*desc=Slice(c,d14,0,kDescriptorDimG);auto*ctx=Slice(c,d14,kDescriptorDimG,512);desc=ggml_interpolate(c,desc,f8->ne[0],f8->ne[1],kDescriptorDimG,1,GGML_SCALE_MODE_BILINEAR);ctx=ggml_interpolate(c,ctx,f8->ne[0],f8->ne[1],512,1,GGML_SCALE_MODE_BILINEAR);
        auto*d8=b.Ref(f8,ctx,"8");if(!d8){err=file.error;cleanup();return false;}desc=ggml_add(c,desc,Slice(c,d8,0,kDescriptorDimG));ctx=Slice(c,d8,kDescriptorDimG,256);desc=ggml_interpolate(c,desc,f4->ne[0],f4->ne[1],kDescriptorDimG,1,GGML_SCALE_MODE_BILINEAR);ctx=ggml_interpolate(c,ctx,f4->ne[0],f4->ne[1],256,1,GGML_SCALE_MODE_BILINEAR);
        auto*d4=b.Ref(f4,ctx,"4");if(!d4){err=file.error;cleanup();return false;}desc=ggml_add(c,desc,Slice(c,d4,0,kDescriptorDimG));ctx=Slice(c,d4,kDescriptorDimG,128);desc=ggml_interpolate(c,desc,f2->ne[0],f2->ne[1],kDescriptorDimG,1,GGML_SCALE_MODE_BILINEAR);ctx=ggml_interpolate(c,ctx,f2->ne[0],f2->ne[1],128,1,GGML_SCALE_MODE_BILINEAR);
        auto*d2=b.Ref(f2,ctx,"2");if(!d2){err=file.error;cleanup();return false;}desc=ggml_add(c,desc,Slice(c,d2,0,kDescriptorDimG));ctx=Slice(c,d2,kDescriptorDimG,32);desc=ggml_interpolate(c,desc,f1->ne[0],f1->ne[1],kDescriptorDimG,1,GGML_SCALE_MODE_BILINEAR);ctx=ggml_interpolate(c,ctx,f1->ne[0],f1->ne[1],32,1,GGML_SCALE_MODE_BILINEAR);
        auto*d1=b.Ref(f1,ctx,"1");if(!d1){err=file.error;cleanup();return false;}desc=ggml_add(c,desc,Slice(c,d1,0,kDescriptorDimG));ggml_set_output(desc);auto*g=ggml_new_graph_custom(c,kMaxGraphNodes,false);ggml_build_forward_expand(g,desc);if(!ggml_gallocr_alloc_graph(be.galloc,g)){err="failed to allocate DeDoDe-G graph";cleanup();return false;}auto pixels=Whcn(im);ggml_backend_tensor_set(in,pixels.data(),0,pixels.size()*sizeof(float));if(ggml_backend_graph_compute(be.be,g)!=GGML_STATUS_SUCCESS){err="DeDoDe-G graph compute failed";cleanup();return false;}std::vector<float> grid(static_cast<size_t>(im.width)*im.height*kDescriptorDimG);ggml_backend_tensor_get(desc,grid.data(),0,grid.size()*sizeof(float));cleanup();out->resize(static_cast<size_t>(n)*kDescriptorDimG);for(int i=0;i<n;++i){float nx=2*k[i].x/sw-1,ny=2*k[i].y/sh-1;for(int d=0;d<kDescriptorDimG;++d)(*out)[static_cast<size_t>(i)*kDescriptorDimG+d]=Sample(grid,im.width,im.height,d,nx,ny);}return true;
    }
    bool Run(const aicore_loma_rgb_image& im,const aicore_loma_keypoint* k,int n,int sw,int sh,std::vector<float>*out){return is_g?RunG(im,k,n,sw,sh,out):RunB(im,k,n,sw,sh,out);}
    void Release(){if(device_weights){ggml_backend_buffer_free(device_weights);device_weights=nullptr;}if(device_context){ggml_free(device_context);device_context=nullptr;}if(weights){ggml_backend_buffer_free(weights);weights=nullptr;}be.release();file.Close();}
    int Dim()const{return is_g?kDescriptorDimG:kDescriptorDimB;}File file;DinoContract dino;bool is_g=false;lightglue::engine_backend be;DescriptorOptions opt;ggml_backend_buffer_t weights=nullptr;ggml_context* device_context=nullptr;ggml_backend_buffer_t device_weights=nullptr;std::string err;
};
Descriptor::Descriptor():impl_(std::make_unique<Impl>()){} Descriptor::~Descriptor()=default;
bool Descriptor::Load(const std::string&p,const DescriptorOptions&o){return impl_->Load(p,o);} bool Descriptor::Describe(const aicore_loma_rgb_image&i,const aicore_loma_keypoint*k,int32_t n,int32_t w,int32_t h,std::vector<float>*o){return impl_->Run(i,k,n,w,h,o);} int32_t Descriptor::descriptor_dim()const{return impl_->Dim();} const std::string& Descriptor::error()const{return impl_->err;}
}  // namespace aicore::loma
