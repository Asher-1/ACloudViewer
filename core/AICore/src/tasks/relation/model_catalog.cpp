// SPDX-License-Identifier: MIT
#include "aicore/relation_capi.h"
namespace {
const aicore_relation_model_entry catalog[] = {
        {"relsgg-vitb16-f16.gguf",
         "https://huggingface.co/Asher-1/relateanything-ggml-full/resolve/main/"
         "gguf/relsgg-vitb16-f16.gguf",
         "relsgg-vitb16-f16", "vitb16",
         "61222e4099c277d5272524507721fc6d65649051bae23984a06fe5cacb8e0806",
         247690880, 0},
        {"relsgg-vitb16-f32.gguf",
         "https://huggingface.co/Asher-1/relateanything-ggml-full/resolve/main/"
         "gguf/relsgg-vitb16-f32.gguf",
         "relsgg-vitb16-f32", "vitb16",
         "1537ba73ae9042b9553774269a4bf44ecfba7ea8776cac44acbea02090fe0934",
         494885312, 0},
        {"relsgg-vitb16-q8_0.gguf",
         "https://huggingface.co/Asher-1/relateanything-ggml-full/resolve/main/"
         "gguf/relsgg-vitb16-q8_0.gguf",
         "relsgg-vitb16-q8_0", "vitb16",
         "61bc2a07233a1d0e189dbcab96736d38afa9f9ade8d8028b327e3aa7a12c3dd5",
         134249664, 0},
        {"relsgg-vitb16-text-student-f32.gguf",
         "https://huggingface.co/Asher-1/relateanything-ggml-full/resolve/main/"
         "gguf/relsgg-vitb16-text-student-f32.gguf",
         "relsgg-vitb16-text-student-f32", "vitb16",
         "8cd368fa8692a987a4d8739fcc642dc74fd2902d73ac5544a65f24018656475b",
         44946432, 1},
        {"relsgg-vits16-f16.gguf",
         "https://huggingface.co/Asher-1/relateanything-ggml-full/resolve/main/"
         "gguf/relsgg-vits16-f16.gguf",
         "relsgg-vits16-f16", "vits16",
         "e00c896a5ad816cdd6af285d63346baab27e5c002935dba59ab0a8e124828a59",
         112374656, 0},
        {"relsgg-vits16-f32.gguf",
         "https://huggingface.co/Asher-1/relateanything-ggml-full/resolve/main/"
         "gguf/relsgg-vits16-f32.gguf",
         "relsgg-vits16-f32", "vits16",
         "ce90fb86c08716a45b9ca85f0bbfe74787966bf17eb267c3ad23f91094ced081",
         224252864, 0},
        {"relsgg-vits16-q8_0.gguf",
         "https://huggingface.co/Asher-1/relateanything-ggml-full/resolve/main/"
         "gguf/relsgg-vits16-q8_0.gguf",
         "relsgg-vits16-q8_0", "vits16",
         "ee168035fa5fa2dc334e159c70ab4785e1bf9c4758427c1264b59878200fa429",
         61281184, 0},
        {"relsgg-vits16-text-student-f32.gguf",
         "https://huggingface.co/Asher-1/relateanything-ggml-full/resolve/main/"
         "gguf/relsgg-vits16-text-student-f32.gguf",
         "relsgg-vits16-text-student-f32", "vits16",
         "8cd368fa8692a987a4d8739fcc642dc74fd2902d73ac5544a65f24018656475b",
         44946432, 1},
        {"relsgg-vits16plus-f16.gguf",
         "https://huggingface.co/Asher-1/relateanything-ggml-full/resolve/main/"
         "gguf/relsgg-vits16plus-f16.gguf",
         "relsgg-vits16plus-f16", "vits16plus",
         "4e5631450416ec2496623631b7f7e2bde9f399570fe747fad1dd9a5f20fbcb37",
         126569408, 0},
        {"relsgg-vits16plus-f32.gguf",
         "https://huggingface.co/Asher-1/relateanything-ggml-full/resolve/main/"
         "gguf/relsgg-vits16plus-f32.gguf",
         "relsgg-vits16plus-f32", "vits16plus",
         "996960494629efd0035379c2fa8ae7ca1ff0039b53d8ab61637e86b7e05ae52e",
         252640256, 0},
        {"relsgg-vits16plus-q8_0.gguf",
         "https://huggingface.co/Asher-1/relateanything-ggml-full/resolve/main/"
         "gguf/relsgg-vits16plus-q8_0.gguf",
         "relsgg-vits16plus-q8_0 (recommended)", "vits16plus",
         "bb70680859abf090bb3942871a5932aa07155f6236e87c6999218ed14400c8f9",
         68877280, 0},
        {"relsgg-vits16plus-text-student-f32.gguf",
         "https://huggingface.co/Asher-1/relateanything-ggml-full/resolve/main/"
         "gguf/relsgg-vits16plus-text-student-f32.gguf",
         "relsgg-vits16plus-text-student-f32", "vits16plus",
         "8cd368fa8692a987a4d8739fcc642dc74fd2902d73ac5544a65f24018656475b",
         44946432, 1},
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
int aicore_relation_model_default_index(void) { return 10; }
