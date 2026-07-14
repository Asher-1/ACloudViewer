// M5 gate: GIANT depth (DualDPT) + pose (CameraDec) via the EXISTING metadata-driven
// DptHead/CamPose on the giant config dims (head dim_in 3072, features 256,
// out_channels [256,512,1024,1024]; cam dim_in 3072). Feeds the dumped feat_g_*
// + cam_g_39 to isolate head/cam from the backbone. SKIP (77) if artifacts absent.
#include "dpt_head.hpp"
#include "cam_pose.hpp"
#include "model_loader.hpp"
#include "backend.hpp"
#include "parity.hpp"
#include <cstdlib>
#include <vector>
#include <array>
#include <string>
int main(){
    const char* gguf=std::getenv("DA_TEST_GGUF_GIANT");
    const char* base=std::getenv("DA_TEST_BASELINE_GIANT");
    if (!gguf||!base) return 77;
    da::ModelLoader ml; if(!ml.load(gguf)) return 1;
    da::Backend be;
    const int H=224, W=224;
    const int Ls[4]={19,27,33,39};
    std::vector<int64_t> s;

    // ---- depth via DptHead on dumped feat_g_* ----
    std::vector<std::vector<float>> feats(4);
    for (int i=0;i<4;++i)
        if (!da_parity::load_baseline(base, std::string("feat_g_")+std::to_string(Ls[i]), feats[i], s)) return 1;
    da::DptHead head(ml, be);
    std::vector<float> depth, conf;
    if (!head.depth(feats, H, W, depth, conf)) { std::fprintf(stderr,"giant depth failed\n"); return 1; }

    bool ok=true;
    {
        std::vector<float> rd, rc;
        if (!da_parity::load_baseline(base, "depth_g", rd, s)) return 1;
        ok &= da_parity::compare(depth, rd, "depth_g", 2e-3f, 2e-3f);
        if (da_parity::load_baseline(base, "depth_conf_g", rc, s))
            ok &= da_parity::compare(conf, rc, "depth_conf_g", 2e-3f, 2e-3f);
    }

    // ---- pose via CamPose on dumped cam_g_39 (last out-layer camera token) ----
    std::vector<float> ct;
    if (!da_parity::load_baseline(base, "cam_g_39", ct, s)) return 1;
    da::CamPose cp(ml, be);
    std::array<float,9> pe; std::array<float,12> ext; std::array<float,9> K;
    if (!cp.pose(ct, H, W, pe, ext, K)) { std::fprintf(stderr,"giant pose failed\n"); return 1; }
    std::vector<float> rext, rK;
    if (!da_parity::load_baseline(base, "extrinsics_g", rext, s)) return 1;
    if (!da_parity::load_baseline(base, "intrinsics_g", rK, s)) return 1;
    std::vector<float> vext(ext.begin(),ext.end()), vK(K.begin(),K.end());
    ok &= da_parity::compare(vext, rext, "extrinsics_g", 2e-3f, 2e-3f);
    ok &= da_parity::compare(vK,   rK,   "intrinsics_g", 2e-3f, 2e-3f);
    return ok?0:1;
}
