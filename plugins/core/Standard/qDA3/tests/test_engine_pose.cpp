#include "engine.hpp"
#include "image_io.hpp"
#include "parity.hpp"
#include <cstdlib>
#include <vector>
#include <array>
int main(){
    const char* gguf=std::getenv("DA_TEST_GGUF"); const char* base=std::getenv("DA_TEST_BASELINE");
    if(!gguf||!base) return 77;
    std::vector<float> raw, rext, rintr; std::vector<int64_t> s;
    if(!da_parity::load_baseline(base,"raw_image",raw,s)) return 77;
    da_parity::load_baseline(base,"extrinsics",rext,s);
    da_parity::load_baseline(base,"intrinsics",rintr,s);
    auto eng=da::Engine::load(gguf,0); if(!eng) return 1;
    da::Image img; img.w=224; img.h=224; img.rgb.resize(224*224*3);
    for(size_t i=0;i<img.rgb.size();++i) img.rgb[i]=(unsigned char)(raw[i]+0.5f);
    std::vector<float> depth,conf; std::array<float,12> ext; std::array<float,9> intr; int H,W;
    if(!eng->depth_pose(img,depth,conf,ext,intr,H,W)) return 1;
    std::vector<float> vext(ext.begin(),ext.end()), vintr(intr.begin(),intr.end());
    bool ok=true;
    ok&=da_parity::compare(vext,rext,"engine_extrinsics",5e-3f,5e-3f);
    ok&=da_parity::compare(vintr,rintr,"engine_intrinsics",5e-3f,5e-3f);
    return ok?0:1;
}
