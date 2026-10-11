#include "aicore/tagging_capi.h"
namespace {
const aicore_tagging_model_entry catalog[] = {
        {"ram_plus_swin_large_14m-f16.gguf",
         "https://huggingface.co/Asher-1/RAM_GGUF/resolve/main/"
         "ram_plus_swin_large_14m-f16.gguf",
         "RAM++ (F16) (recommended)", "ram_plus",
         "7c683b1a0306b7b016ba56db3c9f6bcc1ba0935ae177cad5048780a658f37064",
         664922592, 0},
        {"ram_plus_swin_large_14m-f32.gguf",
         "https://huggingface.co/Asher-1/RAM_GGUF/resolve/main/"
         "ram_plus_swin_large_14m-f32.gguf",
         "RAM++ (F32)", "ram_plus",
         "3c540973ad51803c2cf8cb0f383710aa8c8e62c226c9c939f21a19731fc7cd4a",
         1329714464, 0},
        {"ram_plus_swin_large_14m-q8_0.gguf",
         "https://huggingface.co/Asher-1/RAM_GGUF/resolve/main/"
         "ram_plus_swin_large_14m-q8_0.gguf",
         "RAM++ (Q8_0)", "ram_plus",
         "08635cacabbc054d118df0cf932929332aaaaf9901665730f11e5524ce053d9d",
         468805408, 0},
        {"ram_swin_large_14m-f16.gguf",
         "https://huggingface.co/Asher-1/RAM_GGUF/resolve/main/"
         "ram_swin_large_14m-f16.gguf",
         "RAM (F16)", "ram",
         "cdc38f3924b39605970d785bfa4c54ee6e41d4551c49bf5436a44e34331137b7",
         1018021184, 0},
        {"ram_swin_large_14m-f32.gguf",
         "https://huggingface.co/Asher-1/RAM_GGUF/resolve/main/"
         "ram_swin_large_14m-f32.gguf",
         "RAM (F32)", "ram",
         "95f9602087999a4d6ccbbeed8c2cfbc883425759d2b3f29b6cd3fea4a155c837",
         2035851136, 0},
        {"ram_swin_large_14m-q8_0.gguf",
         "https://huggingface.co/Asher-1/RAM_GGUF/resolve/main/"
         "ram_swin_large_14m-q8_0.gguf",
         "RAM (Q8_0)", "ram",
         "0425a0d363958eaa4fb670644a27142e9cba3769c9147d68c37fa2f10231260d",
         560408576, 0},
        {"tag2text_swin_14m-f16.gguf",
         "https://huggingface.co/Asher-1/RAM_GGUF/resolve/main/"
         "tag2text_swin_14m-f16.gguf",
         "Tag2Text (F16) (recommended)", "tag2text",
         "03ce272940f7254374b89ea3a0e5c8c1ce59df15c4e734ad263f42f7f2d54e51",
         827321984, 1},
        {"tag2text_swin_14m-f32.gguf",
         "https://huggingface.co/Asher-1/RAM_GGUF/resolve/main/"
         "tag2text_swin_14m-f32.gguf",
         "Tag2Text (F32)", "tag2text",
         "d242a323cd1463501d5b7e7c582160f6678b89347c3b3f5e4ceac9caca28407b",
         1654014720, 1},
        {"tag2text_swin_14m-q8_0.gguf",
         "https://huggingface.co/Asher-1/RAM_GGUF/resolve/main/"
         "tag2text_swin_14m-q8_0.gguf",
         "Tag2Text (Q8_0)", "tag2text",
         "70227c4bd98dd269077e35dd7035d9ef689a9d2264715f895847e6b336484f76",
         813313888, 1},
};
}
int aicore_tagging_model_count(void) {
    return sizeof(catalog) / sizeof(catalog[0]);
}
const aicore_tagging_model_entry* aicore_tagging_model_at(int index) {
    return index >= 0 && index < aicore_tagging_model_count() ? &catalog[index]
                                                              : nullptr;
}
int aicore_tagging_model_default_index(void) { return 6; }
