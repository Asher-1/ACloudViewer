#!/usr/bin/env python3
"""Seeded, deterministic GOLD reference for the ray->pose SOLVER (use_ray_pose part B).

The reference RANSAC (ray_utils.ransac_find_homography_weighted_fast_batch) samples
candidate point groups via torch.randperm -> nondeterministic. To make the C++ parity
gate rigorous we REMOVE that nondeterminism: we SEED torch, capture the exact
`rand_sample_iters_idx` the solver used (by monkeypatching get_params_for_ransac), and
dump it alongside the input ray field and the final extrinsics/intrinsics. The C++ solver
CONSUMES the same indices (no RNG on the gated path) so only SVD/QR algorithm differences
remain -> compare within a tight tolerance.

Input ray field is loaded VERBATIM from dumps/reference_rays.gguf (the Part A gold the C++
aux head is already parity-verified against), so the solver gate is decoupled from any head
drift: both reference and C++ consume identical ray bytes.

Writes dumps/reference_ray_pose.gguf + dumps/manifest_ray_pose.json.
"""
import os, sys, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gguf
from scripts.da3_reference import FIX_H, FIX_W

RAYS_GGUF = "dumps/reference_rays.gguf"
OUT = "dumps/reference_ray_pose.gguf"
MANIFEST = "dumps/manifest_ray_pose.json"
SEED = 1234


def load_ray_field(path, Hy, Wx):
    # dump_rays.py writes the tensors FLATTENED (row-major); reshape with known dims.
    r = gguf.GGUFReader(path)
    flat = {t.name: np.array(t.data, dtype=np.float32).reshape(-1) for t in r.tensors}
    ray_np = flat["ray"].reshape(Hy, Wx, 6)
    conf_np = flat["ray_conf"].reshape(Hy, Wx)
    return ray_np, conf_np


def main():
    os.makedirs("dumps", exist_ok=True)
    sys.path.insert(0, "/tmp/da3-src/src")
    import depth_anything_3.utils.ray_utils as RU
    from depth_anything_3.utils.geometry import affine_inverse

    with open("dumps/manifest_rays.json") as f:
        rmani = json.load(f)
    Hy, Wx = int(rmani["H_aux"]), int(rmani["W_aux"])
    ray_np, conf_np = load_ray_field(RAYS_GGUF, Hy, Wx)
    assert ray_np.shape == (Hy, Wx, 6), ray_np.shape

    ray = torch.from_numpy(ray_np).float().reshape(1, 1, Hy, Wx, 6)
    conf = torch.from_numpy(conf_np).float().reshape(1, 1, Hy, Wx, 1)

    cap = {}

    # --- monkeypatch to capture the sampling indices actually used ---
    orig_params = RU.get_params_for_ransac
    def patched_params(N, device):
        n_iter, num_sample, n_sample, idx = orig_params(N, device)
        cap["rand_sample_iters_idx"] = idx.clone()
        cap["_n_iter"] = n_iter
        cap["_num_sample"] = num_sample
        cap["_n_sample"] = n_sample
        return n_iter, num_sample, n_sample, idx
    RU.get_params_for_ransac = patched_params

    # The non-batch homography fit is ONLY called for the final inlier refit (per b);
    # capture its exact inputs (the consensus inlier set actually used).
    orig_fit = RU.find_homography_least_squares_weighted_torch
    fit_calls = []
    def patched_fit(src, dst, w):
        fit_calls.append((src.clone(), dst.clone(), w.clone()))
        return orig_fit(src, dst, w)
    RU.find_homography_least_squares_weighted_torch = patched_fit

    with torch.no_grad():
        torch.manual_seed(SEED)
        ext4, focal, pp = RU.get_extrinsic_from_camray(
            ray, conf, ray.shape[-3], ray.shape[-2], training=False
        )  # ext4: (1,1,4,4) w2c, focal=1/f (1,1,2), pp=pp_raw+1 (1,1,2)

        # _process_ray_pose_estimation post-processing (da3.py:181)
        ext_c2w = affine_inverse(ext4)        # w2c -> c2w
        ext3x4 = ext_c2w[:, :, :3, :]         # (1,1,3,4)
        W, H = FIX_W, FIX_H
        K = torch.eye(3)[None, None].repeat(1, 1, 1, 1).clone()
        K[:, :, 0, 0] = focal[:, :, 0] / 2 * W
        K[:, :, 1, 1] = focal[:, :, 1] / 2 * H
        K[:, :, 0, 2] = pp[:, :, 0] * W * 0.5
        K[:, :, 1, 2] = pp[:, :, 1] * H * 0.5

    RU.get_params_for_ransac = orig_params
    RU.find_homography_least_squares_weighted_torch = orig_fit

    ext3x4 = ext3x4.reshape(3, 4)
    K = K.reshape(3, 3)
    focal2 = focal.reshape(2)
    pp2 = pp.reshape(2)

    assert len(fit_calls) == 1, f"expected exactly one refit, got {len(fit_calls)}"
    refit_src, refit_dst, refit_w = fit_calls[0]
    n_inlier = int(refit_w.shape[0])

    # --- Reconstruct the post-z-normalization full point cloud EXACTLY as
    # compute_optimal_rotation_intrinsics_batch does, so the C++ gate can both
    # (a) validate its own grid+z-norm against src_full/dst_full/w_full and
    # (b) map the consensus refit points back to absolute indices in [0,N) ---
    from depth_anything_3.utils.geometry import unproject_depth
    I_K = torch.eye(3).clone(); I_K[0, 2] = 1.0; I_K[1, 2] = 1.0
    I_K = I_K[None, None].expand(1, 1, -1, -1)
    cam_plane_depth = torch.ones(1, 1, Hy, Wx, 1)
    I_unproj = unproject_depth(cam_plane_depth, I_K, c2w=None, ixt_normalized=True,
                               num_patches_x=Wx, num_patches_y=Hy)  # (1,1,Hy,Wx,3)
    rays_origin = I_unproj.flatten(0, 1).flatten(1, 2)[0]   # (N,3)
    rays_target = ray.flatten(0, 1).flatten(1, 2)[0][:, :3]  # (N,3)  direction half
    conf_flat = conf.squeeze(-1).flatten(0, 1).flatten(1, 2)[0]  # (N,)
    N = Hy * Wx
    zt = rays_target.clone(); zo = rays_origin.clone()
    z_mask = (zt[:, 2].abs() > 1e-4) & (zo[:, 2].abs() > 1e-4)
    zo[z_mask, 0] /= zo[z_mask, 2]; zo[z_mask, 1] /= zo[z_mask, 2]
    zt[z_mask, 0] /= zt[z_mask, 2]; zt[z_mask, 1] /= zt[z_mask, 2]
    src_full = zo[:, :2].contiguous()   # (N,2)
    dst_full = zt[:, :2].contiguous()   # (N,2)
    w_full = conf_flat.clone(); w_full[~z_mask] = 0.0  # (N,)
    # torch.argsort(w, descending=True): the candidate ordering RANSAC samples from.
    # Dumped so the C++ scoring picks the SAME candidate points per rand_sample index.
    sorted_idx = torch.argsort(w_full, descending=True).to(torch.int32)  # (N,)

    # Map consensus refit points back to absolute indices via exact f32 key match.
    key_to_idx = {}
    sf = src_full.numpy()
    for i in range(N):
        key_to_idx.setdefault((sf[i, 0].item(), sf[i, 1].item()), i)
    rs = refit_src.numpy()
    refit_idx = np.empty(n_inlier, dtype=np.int32)
    for k in range(n_inlier):
        refit_idx[k] = key_to_idx[(rs[k, 0].item(), rs[k, 1].item())]
    # verify the mapping reproduces the refit inputs exactly
    assert np.array_equal(sf[refit_idx], rs), "refit_idx reconstruction mismatch (src)"
    assert np.array_equal(dst_full.numpy()[refit_idx], refit_dst.numpy()), "refit_idx mismatch (dst)"
    assert np.array_equal(w_full.numpy()[refit_idx], refit_w.numpy()), "refit_idx mismatch (w)"

    # Sanity: rotation orthonormal, positive focal.
    R = ext3x4[:3, :3].numpy().astype(np.float64)
    orth = float(np.max(np.abs(R @ R.T - np.eye(3))))
    detR = float(np.linalg.det(R))
    f00, f11 = float(K[0, 0]), float(K[1, 1])
    assert orth < 1e-4, f"R not orthonormal: {orth}"
    assert f00 > 0 and f11 > 0, f"focal not positive: {f00},{f11}"

    idx = cap["rand_sample_iters_idx"].cpu().numpy().astype(np.int32)  # (n_iter, num_sample)

    w = gguf.GGUFWriter(OUT, "reference_ray_pose")
    w.add_uint32("ray_pose.seed", SEED)
    w.add_uint32("ray_pose.n_iter", int(cap["_n_iter"]))
    w.add_uint32("ray_pose.num_sample", int(cap["_num_sample"]))
    w.add_uint32("ray_pose.n_sample", int(cap["_n_sample"]))
    w.add_uint32("ray_pose.Hy", Hy)
    w.add_uint32("ray_pose.Wx", Wx)
    w.add_uint32("ray_pose.img_h", FIX_H)
    w.add_uint32("ray_pose.img_w", FIX_W)
    w.add_uint32("ray_pose.n_inlier", n_inlier)

    def addf(name, arr):
        w.add_tensor(name, np.ascontiguousarray(arr.reshape(-1).astype(np.float32)))

    # the ray field (echoed so the C++ gate reads everything from one file)
    addf("ray", ray_np)
    addf("ray_conf", conf_np)
    # the captured sampling indices (int32)
    w.add_tensor("rand_sample_iters_idx", np.ascontiguousarray(idx.reshape(-1)))
    # final outputs
    addf("extrinsics", ext3x4.numpy())     # (3,4) c2w
    addf("intrinsics", K.numpy())          # (3,3)
    addf("focal", focal2.numpy())          # (2,) = 1/f
    addf("pp", pp2.numpy())                # (2,) = pp_raw + 1
    # the post-z-norm full cloud (lets C++ validate its grid+z-norm construction)
    addf("src_full", src_full.numpy())     # (N, 2)
    addf("dst_full", dst_full.numpy())     # (N, 2)
    addf("w_full", w_full.numpy())         # (N,)
    # torch argsort order of weights (candidate ordering)
    w.add_tensor("sorted_idx", np.ascontiguousarray(sorted_idx.numpy().reshape(-1)))
    # absolute indices of the consensus refit points (RNG-free input for the gate)
    w.add_tensor("refit_idx", np.ascontiguousarray(refit_idx.reshape(-1)))
    # debug aids: the exact consensus refit inputs (z-normalized 2D pts + weights)
    addf("refit_src", refit_src.numpy())   # (n_inlier, 2)
    addf("refit_dst", refit_dst.numpy())   # (n_inlier, 2)
    addf("refit_w", refit_w.numpy())       # (n_inlier,)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()

    manifest = {
        "seed": SEED, "n_iter": int(cap["_n_iter"]),
        "num_sample": int(cap["_num_sample"]), "n_sample": int(cap["_n_sample"]),
        "Hy": Hy, "Wx": Wx, "N": Hy * Wx, "img_h": FIX_H, "img_w": FIX_W,
        "n_inlier": n_inlier, "max_inlier_num_refit": 8000,
        # n_inlier == 8000 (==max_inlier_num) => the reference's consensus inlier set
        # exceeded 8000 and was RANDOMLY subsampled (ransac tail randperm). The C++
        # gated path is RNG-free: it consumes the captured absolute refit_idx instead.
        "refit_randperm_subsample_applied": bool(n_inlier >= 8000),
        "R_orthonormality": orth, "det_R": detR,
        "focal_px": [f00, f11], "extrinsics": ext3x4.numpy().tolist(),
        "intrinsics": K.numpy().tolist(),
        "atol_rot": 1e-3, "rtol_intr": 1e-3,
    }
    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)

    print("wrote", OUT)
    print(f"  N={Hy*Wx} n_sample={cap['_n_sample']} n_iter={cap['_n_iter']} "
          f"num_sample={cap['_num_sample']}")
    print(f"  n_inlier={n_inlier} (>8000 randperm path hit: {n_inlier>8000})")
    print(f"  R orthonormality={orth:.2e} det(R)={detR:.6f}")
    print(f"  focal px=({f00:.4f},{f11:.4f}) pp_norm=({float(pp2[0]):.4f},{float(pp2[1]):.4f})")
    print(f"  rand_sample_iters_idx shape={idx.shape} dtype={idx.dtype}")
    print("  extrinsics(c2w 3x4)=\n", np.round(ext3x4.numpy(), 6))
    print("  intrinsics=\n", np.round(K.numpy(), 4))


if __name__ == "__main__":
    main()
