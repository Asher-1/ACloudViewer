// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "aicore/relation_capi.h"
namespace {
const aicore_relation_model_entry catalog[] = {
        {"relsgg-vitb16-f16.gguf",
         "https://huggingface.co/Asher-1/relateanything-ggml-full/resolve/"
         "2377ca3539bf5c4f09161a886a38dd0fc7517bd0/"
         "gguf/relsgg-vitb16-f16.gguf",
         "relsgg-vitb16-f16", "vitb16",
         "fbd6e868e171024182535c281b6ef142b166ea4e67060209d8cfd4b01ac5e911",
         247690848, 0},
        {"relsgg-vitb16-f32.gguf",
         "https://huggingface.co/Asher-1/relateanything-ggml-full/resolve/"
         "2377ca3539bf5c4f09161a886a38dd0fc7517bd0/"
         "gguf/relsgg-vitb16-f32.gguf",
         "relsgg-vitb16-f32", "vitb16",
         "f7d2aa5765877e945d15fc9433c4c947a10e948091692b3f5208c71c24b8e17f",
         494885344, 0},
        {"relsgg-vitb16-q8_0.gguf",
         "https://huggingface.co/Asher-1/relateanything-ggml-full/resolve/"
         "2377ca3539bf5c4f09161a886a38dd0fc7517bd0/"
         "gguf/relsgg-vitb16-q8_0.gguf",
         "relsgg-vitb16-q8_0", "vitb16",
         "44f40492955b493c53d796c7b4ad65d968dce301cd09567920313bf88e4c2b61",
         143419168, 0},
        {"relsgg-vitb16-text-student-f32.gguf",
         "https://huggingface.co/Asher-1/relateanything-ggml-full/resolve/"
         "2377ca3539bf5c4f09161a886a38dd0fc7517bd0/"
         "gguf/relsgg-vitb16-text-student-f32.gguf",
         "relsgg-vitb16-text-student-f32", "vitb16",
         "8cd368fa8692a987a4d8739fcc642dc74fd2902d73ac5544a65f24018656475b",
         44946432, 1},
        {"relsgg-vits16-f16.gguf",
         "https://huggingface.co/Asher-1/relateanything-ggml-full/resolve/"
         "2377ca3539bf5c4f09161a886a38dd0fc7517bd0/"
         "gguf/relsgg-vits16-f16.gguf",
         "relsgg-vits16-f16", "vits16",
         "bd5d0eabc41747af6d0243c755f789917c3f53864f763989bee5a4f0371767cf",
         112374624, 0},
        {"relsgg-vits16-f32.gguf",
         "https://huggingface.co/Asher-1/relateanything-ggml-full/resolve/"
         "2377ca3539bf5c4f09161a886a38dd0fc7517bd0/"
         "gguf/relsgg-vits16-f32.gguf",
         "relsgg-vits16-f32", "vits16",
         "b87e7cff5e5678e2871cbfcf2804be22e5a9b814a89be763c54c95b9c39468f4",
         224252896, 0},
        {"relsgg-vits16-q8_0.gguf",
         "https://huggingface.co/Asher-1/relateanything-ggml-full/resolve/"
         "2377ca3539bf5c4f09161a886a38dd0fc7517bd0/"
         "gguf/relsgg-vits16-q8_0.gguf",
         "relsgg-vits16-q8_0", "vits16",
         "d281ace4ad9a51edc78da26660ffd89a98d073df352de735711812b48e2dee53",
         70450688, 0},
        {"relsgg-vits16-text-student-f32.gguf",
         "https://huggingface.co/Asher-1/relateanything-ggml-full/resolve/"
         "2377ca3539bf5c4f09161a886a38dd0fc7517bd0/"
         "gguf/relsgg-vits16-text-student-f32.gguf",
         "relsgg-vits16-text-student-f32", "vits16",
         "8cd368fa8692a987a4d8739fcc642dc74fd2902d73ac5544a65f24018656475b",
         44946432, 1},
        {"relsgg-vits16plus-f16.gguf",
         "https://huggingface.co/Asher-1/relateanything-ggml-full/resolve/"
         "2377ca3539bf5c4f09161a886a38dd0fc7517bd0/"
         "gguf/relsgg-vits16plus-f16.gguf",
         "relsgg-vits16plus-f16", "vits16plus",
         "36ed9f832324dff4d06edf216782eb692645f8f10c0bd7d1b4c5fa58d44d7d39",
         126569376, 0},
        {"relsgg-vits16plus-f32.gguf",
         "https://huggingface.co/Asher-1/relateanything-ggml-full/resolve/"
         "2377ca3539bf5c4f09161a886a38dd0fc7517bd0/"
         "gguf/relsgg-vits16plus-f32.gguf",
         "relsgg-vits16plus-f32", "vits16plus",
         "24970d851decd61bfdf4b109f5353ff81596ae8d7c9a35635f9de89d17f935e4",
         252640288, 0},
        {"relsgg-vits16plus-q8_0.gguf",
         "https://huggingface.co/Asher-1/relateanything-ggml-full/resolve/"
         "2377ca3539bf5c4f09161a886a38dd0fc7517bd0/"
         "gguf/relsgg-vits16plus-q8_0.gguf",
         "relsgg-vits16plus-q8_0 (recommended)", "vits16plus",
         "c19c25acae196e3a762fcbb4032ba450e716847bc66540988d730e4c75572177",
         78046784, 0},
        {"relsgg-vits16plus-text-student-f32.gguf",
         "https://huggingface.co/Asher-1/relateanything-ggml-full/resolve/"
         "2377ca3539bf5c4f09161a886a38dd0fc7517bd0/"
         "gguf/relsgg-vits16plus-text-student-f32.gguf",
         "relsgg-vits16plus-text-student-f32", "vits16plus",
         "8cd368fa8692a987a4d8739fcc642dc74fd2902d73ac5544a65f24018656475b",
         44946432, 1},
        // ConvNeXt models (DINOv3 backbone, 4 sizes × 3 quantizations)
        {"relsgg-convnext-tiny-f16.gguf",
         "https://huggingface.co/Asher-1/relateanything-ggml-full/resolve/"
         "2377ca3539bf5c4f09161a886a38dd0fc7517bd0/"
         "gguf/relsgg-convnext-tiny-f16.gguf",
         "relsgg-convnext-tiny-f16", "convnext-tiny",
         "785d3f0d324bb1412c69df1b89fe04be7cf3afe92258f1e68a78d89ce079a65f",
         137340320, 0},
        {"relsgg-convnext-tiny-f32.gguf",
         "https://huggingface.co/Asher-1/relateanything-ggml-full/resolve/"
         "2377ca3539bf5c4f09161a886a38dd0fc7517bd0/"
         "gguf/relsgg-convnext-tiny-f32.gguf",
         "relsgg-convnext-tiny-f32", "convnext-tiny",
         "97fb3e7409699188926cf285d714620cc3dd0f89e13cbe2f3975ff1a78ebb272",
         274181024, 0},
        {"relsgg-convnext-tiny-q8_0.gguf",
         "https://huggingface.co/Asher-1/relateanything-ggml-full/resolve/"
         "2377ca3539bf5c4f09161a886a38dd0fc7517bd0/"
         "gguf/relsgg-convnext-tiny-q8_0.gguf",
         "relsgg-convnext-tiny-q8_0", "convnext-tiny",
         "018690fdaf159a9627de71506c93ee85624a77345d1148d3b174ef43a03ec10c",
         96192288, 0},
        {"relsgg-convnext-small-f16.gguf",
         "https://huggingface.co/Asher-1/relateanything-ggml-full/resolve/"
         "2377ca3539bf5c4f09161a886a38dd0fc7517bd0/"
         "gguf/relsgg-convnext-small-f16.gguf",
         "relsgg-convnext-small-f16", "convnext-small",
         "5cc130db41c1a7de0eb4da5d63539c605686eaadbcd649f92ae2e235f953edf0",
         180629120, 0},
        {"relsgg-convnext-small-f32.gguf",
         "https://huggingface.co/Asher-1/relateanything-ggml-full/resolve/"
         "2377ca3539bf5c4f09161a886a38dd0fc7517bd0/"
         "gguf/relsgg-convnext-small-f32.gguf",
         "relsgg-convnext-small-f32", "convnext-small",
         "b5290933a39cff1ef97f064d6965f4d3c5260955a37dbc30728ad21f9344c606",
         360738944, 0},
        {"relsgg-convnext-small-q8_0.gguf",
         "https://huggingface.co/Asher-1/relateanything-ggml-full/resolve/"
         "2377ca3539bf5c4f09161a886a38dd0fc7517bd0/"
         "gguf/relsgg-convnext-small-q8_0.gguf",
         "relsgg-convnext-small-q8_0", "convnext-small",
         "d0d0431f4e1493c58aa4d428484fbae61b6d0aeb8197879a6f7fc3db17a36e84",
         120376320, 0},
        {"relsgg-convnext-base-f16.gguf",
         "https://huggingface.co/Asher-1/relateanything-ggml-full/resolve/"
         "2377ca3539bf5c4f09161a886a38dd0fc7517bd0/"
         "gguf/relsgg-convnext-base-f16.gguf",
         "relsgg-convnext-base-f16", "convnext-base",
         "b7fd03203c87b7203215b4dfe4e58a3d5881368c2015efa6eca23005544d099e",
         267090624, 0},
        {"relsgg-convnext-base-f32.gguf",
         "https://huggingface.co/Asher-1/relateanything-ggml-full/resolve/"
         "2377ca3539bf5c4f09161a886a38dd0fc7517bd0/"
         "gguf/relsgg-convnext-base-f32.gguf",
         "relsgg-convnext-base-f32", "convnext-base",
         "c3ec29b957f6c04d472576dbe052e056925ee56370e852600db7d54bf331a99b",
         533661952, 0},
        {"relsgg-convnext-base-q8_0.gguf",
         "https://huggingface.co/Asher-1/relateanything-ggml-full/resolve/"
         "2377ca3539bf5c4f09161a886a38dd0fc7517bd0/"
         "gguf/relsgg-convnext-base-q8_0.gguf",
         "relsgg-convnext-base-q8_0", "convnext-base",
         "549a4ab4a053d582454126d6e3f3dec70969ab1fcdfb91681cab8c01c5cb7dcc",
         176707744, 0},
        {"relsgg-convnext-large-f16.gguf",
         "https://huggingface.co/Asher-1/relateanything-ggml-full/resolve/"
         "2377ca3539bf5c4f09161a886a38dd0fc7517bd0/"
         "gguf/relsgg-convnext-large-f16.gguf",
         "relsgg-convnext-large-f16", "convnext-large",
         "d0dbd9fea5fd0fe6c728dd6df55b9cb34bb7a5fb9645c5591b7767622b0f3fbc",
         511578944, 0},
        {"relsgg-convnext-large-f32.gguf",
         "https://huggingface.co/Asher-1/relateanything-ggml-full/resolve/"
         "2377ca3539bf5c4f09161a886a38dd0fc7517bd0/"
         "gguf/relsgg-convnext-large-f32.gguf",
         "relsgg-convnext-large-f32", "convnext-large",
         "2fc1d48e2ea8e96ff8429a734205c5915eac73bf774efa2299227b8126d7ecc2",
         1022638592, 0},
        {"relsgg-convnext-large-q8_0.gguf",
         "https://huggingface.co/Asher-1/relateanything-ggml-full/resolve/"
         "2377ca3539bf5c4f09161a886a38dd0fc7517bd0/"
         "gguf/relsgg-convnext-large-q8_0.gguf",
         "relsgg-convnext-large-q8_0", "convnext-large",
         "37a743155aab2390f5d7c8c4dd3d3c595d299870ce951db357dd1ade788ea6e2",
         335619552, 0},
        {"relateanything-tokenizer.json",
         "https://huggingface.co/maelic/relsgg-vits16/resolve/"
         "282652489584b156d533e704bdbf51bca66f3a90/tokenizer.json",
         "Official CLIP tokenizer", "clip",
         "6d9109cc838977f3ca94a379eec36aecc7c807e1785cd729660ca2fc0171fb35",
         3642073, 2},
};
}
int aicore_relation_model_count(void) {
    return sizeof(catalog) / sizeof(catalog[0]);
}
const aicore_relation_model_entry *aicore_relation_model_at(int i) {
    return i >= 0 && i < aicore_relation_model_count() ? &catalog[i] : nullptr;
}
int aicore_relation_model_default_index(void) { return 10; }  // vits16plus-q8_0 (recommended)
