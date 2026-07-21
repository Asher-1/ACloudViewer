"""Single source of truth for GGUF arch, KV keys, and tensor renames.
Both the C++ loader (via generated include/aicore/depth_gguf_keys.h) and the Python
converter import from here, so they cannot drift."""
import re

ARCH = "depthanything3"

# short-key -> full GGUF KV string
KV = {
    "arch":                 f"{ARCH}.arch",
    "checkpoint_name":      f"{ARCH}.checkpoint_name",
    "patch_size":           f"{ARCH}.patch_size",
    "image_size":           f"{ARCH}.image_size",
    "task_caps":            f"{ARCH}.task_caps",          # bitmask of heads present
    # backbone (DINOv2)
    "vit.embed_dim":        f"{ARCH}.vit.embed_dim",
    "vit.depth":            f"{ARCH}.vit.depth",
    "vit.num_heads":        f"{ARCH}.vit.num_heads",
    "vit.head_dim":         f"{ARCH}.vit.head_dim",
    "vit.mlp_hidden":       f"{ARCH}.vit.mlp_hidden",
    "vit.ffn_type":         f"{ARCH}.vit.ffn_type",        # "mlp" (fc1/fc2) or "swiglu" (w12/w3)
    "vit.num_register":     f"{ARCH}.vit.num_register_tokens",
    "vit.init_values":      f"{ARCH}.vit.init_values",
    "vit.alt_start":        f"{ARCH}.vit.alt_start",
    "vit.rope_start":       f"{ARCH}.vit.rope_start",
    "vit.qknorm_start":     f"{ARCH}.vit.qknorm_start",
    "vit.rope_freq":        f"{ARCH}.vit.rope_freq",
    "vit.cat_token":        f"{ARCH}.vit.cat_token",
    "vit.qkv_bias":         f"{ARCH}.vit.qkv_bias",
    "vit.ln_eps":           f"{ARCH}.vit.ln_eps",
    "vit.interp_offset":    f"{ARCH}.vit.interpolate_offset",
    "vit.interp_antialias": f"{ARCH}.vit.interpolate_antialias",
    "vit.pos_embed_grid":   f"{ARCH}.vit.pos_embed_grid",  # M where pos_embed has M*M+1 rows
    "vit.out_layers":       f"{ARCH}.vit.out_layers",
    # preprocessing
    "img.mean":             f"{ARCH}.img.mean",
    "img.std":              f"{ARCH}.img.std",
    "img.resize_mode":      f"{ARCH}.img.resize_mode",
    "img.resize_target":    f"{ARCH}.img.resize_target",   # target long/short side, multiple of patch
    # DualDPT depth head (main path)
    "head.features":        f"{ARCH}.head.features",
    "head.out_channels":    f"{ARCH}.head.out_channels",
    "head.output_dim":      f"{ARCH}.head.output_dim",
    "head.pos_embed":       f"{ARCH}.head.pos_embed",
    "head.down_ratio":      f"{ARCH}.head.down_ratio",
    "head.activation":      f"{ARCH}.head.activation",
    "head.conf_activation": f"{ARCH}.head.conf_activation",
    "head.sky_activation":  f"{ARCH}.head.sky_activation",   # standalone mono DPT + sky head
    "head.norm_type":       f"{ARCH}.head.norm_type",        # "idt" | "layer"
    "head.max_depth":       f"{ARCH}.head.max_depth",      # 0/absent=relative; 20/80=metric (DA2)
    # DualDPT auxiliary ray head (opt-in via converter --with-aux)
    "head.has_aux":         f"{ARCH}.head.has_aux",          # bool: aux ray tensors present
    "head.aux_ray_dim":     f"{ARCH}.head.aux_ray_dim",      # 6 (ray channels; +1 conf = 7)
    "head.aux_levels":      f"{ARCH}.head.aux_levels",       # aux pyramid level count (4)
    # camera pose decoder (cam_dec MLP)
    "cam.dim_in":           f"{ARCH}.cam.dim_in",
    # GSDPT 3D-Gaussian head (giant only)
    "gs.output_dim":        f"{ARCH}.gs.output_dim",        # 38 = raw_gs(37) + conf(1)
    "gs.features":          f"{ARCH}.gs.features",
    "gs.out_channels":      f"{ARCH}.gs.out_channels",
    "gs.sh_degree":         f"{ARCH}.gs.sh_degree",
    "gs.scale_min":         f"{ARCH}.gs.scale_min",
    "gs.scale_max":         f"{ARCH}.gs.scale_max",
    "gs.pred_offset_depth": f"{ARCH}.gs.pred_offset_depth",
    "gs.pred_offset_xy":    f"{ARCH}.gs.pred_offset_xy",
    "gs.pred_color":        f"{ARCH}.gs.pred_color",
    # ===== nested metric branch (DA3NESTED) =====
    # metric ViT-L backbone (24 layers, embed 1024, classic MLP fc1/fc2)
    "m_vit.embed_dim":        f"{ARCH}.m_vit.embed_dim",
    "m_vit.depth":            f"{ARCH}.m_vit.depth",
    "m_vit.num_heads":        f"{ARCH}.m_vit.num_heads",
    "m_vit.head_dim":         f"{ARCH}.m_vit.head_dim",
    "m_vit.mlp_hidden":       f"{ARCH}.m_vit.mlp_hidden",
    "m_vit.ffn_type":         f"{ARCH}.m_vit.ffn_type",
    "m_vit.num_register":     f"{ARCH}.m_vit.num_register_tokens",
    "m_vit.init_values":      f"{ARCH}.m_vit.init_values",
    "m_vit.alt_start":        f"{ARCH}.m_vit.alt_start",
    "m_vit.rope_start":       f"{ARCH}.m_vit.rope_start",
    "m_vit.qknorm_start":     f"{ARCH}.m_vit.qknorm_start",
    "m_vit.rope_freq":        f"{ARCH}.m_vit.rope_freq",
    "m_vit.cat_token":        f"{ARCH}.m_vit.cat_token",
    "m_vit.qkv_bias":         f"{ARCH}.m_vit.qkv_bias",
    "m_vit.ln_eps":           f"{ARCH}.m_vit.ln_eps",
    "m_vit.interp_offset":    f"{ARCH}.m_vit.interpolate_offset",
    "m_vit.interp_antialias": f"{ARCH}.m_vit.interpolate_antialias",
    "m_vit.pos_embed_grid":   f"{ARCH}.m_vit.pos_embed_grid",
    "m_vit.out_layers":       f"{ARCH}.m_vit.out_layers",
    # metric DPT head (single-head) + sky head
    "m_head.features":        f"{ARCH}.m_head.features",
    "m_head.out_channels":    f"{ARCH}.m_head.out_channels",
    "m_head.output_dim":      f"{ARCH}.m_head.output_dim",
    "m_head.down_ratio":      f"{ARCH}.m_head.down_ratio",
    "m_head.activation":      f"{ARCH}.m_head.activation",
    "m_head.conf_activation": f"{ARCH}.m_head.conf_activation",
    "m_head.sky_activation":  f"{ARCH}.m_head.sky_activation",
    "m_head.norm_type":       f"{ARCH}.m_head.norm_type",
}

def rename_backbone(name: str):
    """HF backbone param name (with 'pretrained.' prefix already stripped) ->
    GGUF tensor name, or None if not a backbone tensor."""
    n = name
    if n == "patch_embed.proj.weight": return "vit.patch_embed.weight"
    if n == "patch_embed.proj.bias":   return "vit.patch_embed.bias"
    if n == "cls_token":               return "vit.cls_token"
    if n == "camera_token":            return "vit.camera_token"
    if n == "pos_embed":               return "vit.pos_embed"
    if n == "norm.weight":             return "vit.norm.weight"
    if n == "norm.bias":               return "vit.norm.bias"
    m = re.match(r"^blocks\.(\d+)\.(.+)$", n)
    if m:
        i, rest = m.group(1), m.group(2)
        table = {
            "norm1.weight": "norm1.weight", "norm1.bias": "norm1.bias",
            "norm2.weight": "norm2.weight", "norm2.bias": "norm2.bias",
            "attn.qkv.weight": "attn_qkv.weight", "attn.qkv.bias": "attn_qkv.bias",
            "attn.proj.weight": "attn_proj.weight", "attn.proj.bias": "attn_proj.bias",
            "attn.q_norm.weight": "attn_qnorm.weight", "attn.q_norm.bias": "attn_qnorm.bias",
            "attn.k_norm.weight": "attn_knorm.weight", "attn.k_norm.bias": "attn_knorm.bias",
            "ls1.gamma": "ls1", "ls2.gamma": "ls2",
            "mlp.fc1.weight": "mlp_fc1.weight", "mlp.fc1.bias": "mlp_fc1.bias",
            "mlp.fc2.weight": "mlp_fc2.weight", "mlp.fc2.bias": "mlp_fc2.bias",
            # SwiGLU FFN (giant): w12 = Linear(C->2*hidden), w3 = Linear(hidden->C)
            "mlp.w12.weight": "mlp_w12.weight", "mlp.w12.bias": "mlp_w12.bias",
            "mlp.w3.weight": "mlp_w3.weight", "mlp.w3.bias": "mlp_w3.bias",
        }
        if rest in table:
            return f"vit.blk.{i}.{table[rest]}"
    return None


# Aux (ray/sky) head tensors live in the same module as the main depth path but
# belong to M3. M2 intentionally skips them; this matches the prefixes for
# scratch.refinenet{i}_aux.*, scratch.output_conv1_aux.*, scratch.output_conv2_aux.*
_HEAD_AUX_RE = re.compile(r"(refinenet\d+_aux|output_conv1_aux|output_conv2_aux)")


def is_head_aux(name: str) -> bool:
    """True if `name` is a DualDPT auxiliary-head param (intentionally skipped in M2)."""
    return _HEAD_AUX_RE.search(name) is not None


def rename_head(name: str):
    """HF DualDPT head param name (already without 'head.' prefix, e.g. 'norm.weight',
    'projects.0.weight', 'scratch.refinenet1.resConfUnit1.conv1.weight') ->
    GGUF tensor name, or None if it is an aux/unknown tensor (caller decides whether
    a None is an intentional aux skip via is_head_aux, or a hard error)."""
    n = name
    if n in ("norm.weight", "norm.bias"):
        return f"head.{n}"
    m = re.match(r"^projects\.(\d+)\.(weight|bias)$", n)
    if m:
        return f"head.proj.{m.group(1)}.{m.group(2)}"
    m = re.match(r"^resize_layers\.(\d+)\.(weight|bias)$", n)
    if m:
        return f"head.resize.{m.group(1)}.{m.group(2)}"
    m = re.match(r"^scratch\.layer(\d+)_rn\.(weight|bias)$", n)
    if m:
        return f"head.scratch.layer{m.group(1)}_rn.{m.group(2)}"
    # refinenet{i} (main only; aux is handled by is_head_aux and returns None here)
    m = re.match(
        r"^scratch\.refinenet(\d+)\.resConfUnit(\d+)\.conv(\d+)\.(weight|bias)$", n)
    if m and "_aux" not in n:
        i, unit, conv, wb = m.groups()
        return f"head.scratch.rn{i}.rc{unit}.c{conv}.{wb}"
    m = re.match(r"^scratch\.refinenet(\d+)\.out_conv\.(weight|bias)$", n)
    if m and "_aux" not in n:
        return f"head.scratch.rn{m.group(1)}.out.{m.group(2)}"
    if re.match(r"^scratch\.output_conv1\.(weight|bias)$", n):
        return "head.scratch.out1." + n.rsplit(".", 1)[-1]
    m = re.match(r"^scratch\.output_conv2\.(\d+)\.(weight|bias)$", n)
    if m:
        sub = {"0": "out2a", "2": "out2b"}.get(m.group(1))
        if sub is not None:
            return f"head.scratch.{sub}.{m.group(2)}"
    return None


def rename_head_aux(name: str, last_level: int):
    """HF DualDPT AUX head param name (already without 'head.' prefix) -> stable GGUF
    tensor name under 'head.scratch.*_aux', or None if it is an unused aux level / not
    an aux tensor. Only the FINEST pyramid level (`last_level` = aux_levels-1) of the
    per-level output_conv1_aux / output_conv2_aux ModuleLists is emitted (the only one
    used downstream); all 4 refinenet_aux blocks ARE emitted (the pyramid is sequential).

    output_conv1_aux[last] = Sequential of 5 Conv2d (aux_out1_conv_num=5), indices 0..4
        -> head.scratch.out1_aux.{0..4}.{w,b}
    output_conv2_aux[last] = Sequential[Conv(0), Permute, LayerNorm(2), Permute, ReLU,
        Conv(5)] -> conv0=out2a_aux, LayerNorm(2)=out2_aux_ln, conv5=out2b_aux."""
    n = name
    m = re.match(
        r"^scratch\.refinenet(\d+)_aux\.resConfUnit(\d+)\.conv(\d+)\.(weight|bias)$", n)
    if m:
        i, unit, conv, wb = m.groups()
        return f"head.scratch.rn{i}_aux.rc{unit}.c{conv}.{wb}"
    m = re.match(r"^scratch\.refinenet(\d+)_aux\.out_conv\.(weight|bias)$", n)
    if m:
        return f"head.scratch.rn{m.group(1)}_aux.out.{m.group(2)}"
    m = re.match(r"^scratch\.output_conv1_aux\.(\d+)\.(\d+)\.(weight|bias)$", n)
    if m:
        lvl, idx, wb = int(m.group(1)), m.group(2), m.group(3)
        if lvl != last_level:
            return None
        return f"head.scratch.out1_aux.{idx}.{wb}"
    m = re.match(r"^scratch\.output_conv2_aux\.(\d+)\.(\d+)\.(weight|bias)$", n)
    if m:
        lvl, idx, wb = int(m.group(1)), m.group(2), m.group(3)
        # The channels-last LayerNorm at index 2 is a SINGLE module instance shared
        # by ALL aux levels (ln_seq is built once outside the per-level loop in the
        # reference), so named_parameters() dedupes it to level 0 only — emit it
        # regardless of level. The two convs (idx 0,5) are per-level: finest only.
        if idx == "2":
            return f"head.scratch.out2_aux_ln.{wb}"
        if lvl != last_level:
            return None
        sub = {"0": "out2a_aux", "5": "out2b_aux"}.get(idx)
        if sub is not None:
            return f"head.scratch.{sub}.{wb}"
    return None


def rename_cam(name: str):
    """HF CameraDec param name (already without 'cam_dec.' prefix, e.g.
    'backbone.0.weight', 'fc_t.bias') -> GGUF tensor name, or None if unknown.

    Maps the default cam_dec MLP:
        backbone.0.{weight,bias} -> cam.bb0.{weight,bias}   (Linear 1536->1536)
        backbone.2.{weight,bias} -> cam.bb2.{weight,bias}   (Linear 1536->1536)
        fc_t.{weight,bias}       -> cam.fc_t.{weight,bias}  (Linear 1536->3)
        fc_qvec.{weight,bias}    -> cam.fc_q.{weight,bias}   (Linear 1536->4)
        fc_fov.0.{weight,bias}   -> cam.fc_fov.{weight,bias} (Linear 1536->2)
    """
    m = re.match(r"^backbone\.(0|2)\.(weight|bias)$", name)
    if m:
        return f"cam.bb{m.group(1)}.{m.group(2)}"
    m = re.match(r"^fc_t\.(weight|bias)$", name)
    if m:
        return f"cam.fc_t.{m.group(1)}"
    m = re.match(r"^fc_qvec\.(weight|bias)$", name)
    if m:
        return f"cam.fc_q.{m.group(1)}"
    m = re.match(r"^fc_fov\.0\.(weight|bias)$", name)
    if m:
        return f"cam.fc_fov.{m.group(1)}"
    return None


def rename_gs(name: str):
    """HF GSDPT (gs_head) param name (already without 'gs_head.' prefix, e.g.
    'images_merger.0.weight', 'projects.0.weight',
    'scratch.refinenet1.resConfUnit1.conv1.weight') -> GGUF tensor name, or None
    if unknown. GSDPT is the single-head DPT plus an images_merger; structure
    mirrors rename_head but under the 'gs.' prefix (refinenet4 has no rc1)."""
    n = name
    # images_merger = Sequential(Conv2d, GELU, Conv2d, GELU, Conv2d, GELU); the
    # learned convs live at indices 0, 2, 4.
    m = re.match(r"^images_merger\.(0|2|4)\.(weight|bias)$", n)
    if m:
        i = {"0": "0", "2": "1", "4": "2"}[m.group(1)]
        return f"gs.merger.{i}.{m.group(2)}"
    m = re.match(r"^projects\.(\d+)\.(weight|bias)$", n)
    if m:
        return f"gs.proj.{m.group(1)}.{m.group(2)}"
    m = re.match(r"^resize_layers\.(\d+)\.(weight|bias)$", n)
    if m:
        return f"gs.resize.{m.group(1)}.{m.group(2)}"
    m = re.match(r"^scratch\.layer(\d+)_rn\.(weight|bias)$", n)
    if m:
        return f"gs.scratch.layer{m.group(1)}_rn.{m.group(2)}"
    m = re.match(
        r"^scratch\.refinenet(\d+)\.resConfUnit(\d+)\.conv(\d+)\.(weight|bias)$", n)
    if m:
        i, unit, conv, wb = m.groups()
        return f"gs.scratch.rn{i}.rc{unit}.c{conv}.{wb}"
    m = re.match(r"^scratch\.refinenet(\d+)\.out_conv\.(weight|bias)$", n)
    if m:
        return f"gs.scratch.rn{m.group(1)}.out.{m.group(2)}"
    if re.match(r"^scratch\.output_conv1\.(weight|bias)$", n):
        return "gs.scratch.out1." + n.rsplit(".", 1)[-1]
    m = re.match(r"^scratch\.output_conv2\.(\d+)\.(weight|bias)$", n)
    if m:
        sub = {"0": "out2a", "2": "out2b"}.get(m.group(1))
        if sub is not None:
            return f"gs.scratch.{sub}.{m.group(2)}"
    return None


# ===== standalone monocular branch (DA3MONO-LARGE) ===========================
def rename_mono_head(name: str):
    """HF standalone monocular DPT-head param name (already without 'head.' prefix)
    -> GGUF tensor name under the 'head.*' prefix the C++ reads directly, or None if
    unknown. Identical to rename_head (single-head DPT main path) PLUS the parallel
    sky head scratch.sky_output_conv2.{0,2} -> head.scratch.sky_out2{a,b} (which
    rename_head skips). Unlike rename_metric_head this keeps the head.* prefix (the
    mono net is standalone, not nested under m_head.*)."""
    m = re.match(r"^scratch\.sky_output_conv2\.(\d+)\.(weight|bias)$", name)
    if m:
        sub = {"0": "sky_out2a", "2": "sky_out2b"}.get(m.group(1))
        if sub is not None:
            return f"head.scratch.{sub}.{m.group(2)}"
    return rename_head(name)


# ===== nested metric branch (DA3NESTED-GIANT-LARGE) ==========================
def rename_metric_backbone(name: str):
    """HF metric backbone param name (pretrained.* prefix already stripped) ->
    GGUF tensor name under the 'm_vit.*' prefix, or None if not a backbone tensor.
    The metric ViT-L uses the classic MLP FFN (fc1/fc2), so this is the same
    mapping as rename_backbone, just re-prefixed m_vit.* instead of vit.*."""
    g = rename_backbone(name)
    if g is not None and g.startswith("vit."):
        return "m_vit." + g[len("vit."):]
    return None


def rename_metric_head(name: str):
    """HF metric DPT-head param name (already without 'head.' prefix) -> GGUF
    tensor name under the 'm_head.*' prefix, or None if unknown. Mirrors
    rename_head (single-head DPT main path: norm/projects/resize/scratch.layer_rn/
    refinenet/output_conv1/output_conv2.{0,2}) PLUS the parallel sky head
    scratch.sky_output_conv2.{0,2} -> m_head.scratch.sky_out2{a,b}."""
    n = name
    m = re.match(r"^scratch\.sky_output_conv2\.(\d+)\.(weight|bias)$", n)
    if m:
        sub = {"0": "sky_out2a", "2": "sky_out2b"}.get(m.group(1))
        if sub is not None:
            return f"m_head.scratch.{sub}.{m.group(2)}"
    g = rename_head(n)
    if g is not None and g.startswith("head."):
        return "m_head." + g[len("head."):]
    return None
