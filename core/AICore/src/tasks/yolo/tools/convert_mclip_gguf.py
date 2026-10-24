#!/usr/bin/env python
"""Convert sentence-transformers/clip-ViT-B-32-multilingual-v1 (DistilBERT
multilingual text encoder + linear projection into the OpenAI CLIP ViT-B/32
text space) into a single GGUF for AICore's mclip text graph. The released
model encodes prompts in 100+ languages and lands them in the same 512-d
space the YOLO-World detection head was trained against.

Usage:
  convert_mclip_gguf.py pack <model_dir> <out.gguf>
  convert_mclip_gguf.py ref  <model_dir> <text> <out.json>   # fp32 reference
                             embedding from a hand-written forward pass
                             (DistilBERT + mean pooling + projection + L2),
                             used to verify the C++ graph bit-for-bit.
Requires: torch, numpy, gguf (pip).
"""
import json
import struct
import sys
import unicodedata

import numpy as np
import torch
import gguf

PAD, UNK, CLS, SEP = "[PAD]", "[UNK]", "[CLS]", "[SEP]"


def read_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for k, v in header.items():
            if k == "__metadata__":
                continue
            dt, shape = v["dtype"], v["shape"]
            begin, end = v["data_offsets"]
            f.seek(base + begin)
            raw = f.read(end - begin)
            assert dt == "F32", f"unexpected dtype {dt} for {k}"
            out[k] = torch.from_numpy(
                np.frombuffer(raw, dtype=np.float32).reshape(shape).copy())
        return out


# ---------------------------------------------------------------------------
# WordPiece tokenizer (BertTokenizer: do_lower_case=false, CJK per-char)
# ---------------------------------------------------------------------------

def _is_whitespace(ch):
    if ch in (" ", "\t", "\n", "\r"):
        return True
    return unicodedata.category(ch) == "Zs"


def _is_control(ch):
    if ch in ("\t", "\n", "\r"):
        return False
    return unicodedata.category(ch).startswith("C")


def _is_punctuation(ch):
    cp = ord(ch)
    if (33 <= cp <= 47) or (58 <= cp <= 64) or (91 <= cp <= 96) or (123 <= cp <= 126):
        return True
    return unicodedata.category(ch).startswith("P")


def _is_cjk(cp):
    return any(lo <= cp <= hi for lo, hi in (
        (0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0xF900, 0xFAFF),
        (0x20000, 0x2A6DF), (0x2A700, 0x2B73F), (0x2B740, 0x2B81F),
        (0x2B820, 0x2CEAF), (0x2CEB0, 0x2EBEF), (0x30000, 0x3134F)))


def basic_tokenize(text):
    out = []
    for ch in text:
        if _is_control(ch):
            continue
        if _is_whitespace(ch):
            out.append(" ")
        elif _is_cjk(ord(ch)):
            out.append(" " + ch + " ")
        else:
            out.append(ch)
    words = []
    for tok in "".join(out).split():
        buf = []
        for c in tok:
            if _is_punctuation(c):
                if buf:
                    words.append("".join(buf))
                    buf = []
                words.append(c)
            else:
                buf.append(c)
        if buf:
            words.append("".join(buf))
    return words


def wordpiece(word, vocab, max_chars=100):
    if len(word) > max_chars:
        return [UNK]
    pieces, start = [], 0
    while start < len(word):
        end = len(word)
        chosen = None
        while start < end:
            sub = word[start:end]
            if start > 0:
                sub = "##" + sub
            if sub in vocab:
                chosen = sub
                break
            end -= 1
        if chosen is None:
            return [UNK]
        pieces.append(chosen)
        start = end
    return pieces


def tokenize(text, vocab, max_len=250):
    tokens = [CLS]
    for w in basic_tokenize(text):
        tokens.extend(wordpiece(w, vocab))
        if len(tokens) >= max_len:
            break
    tokens.append(SEP)
    return tokens[:max_len]


# ---------------------------------------------------------------------------
# Hand-written DistilBERT forward (fp32 reference)
# ---------------------------------------------------------------------------

def layer_norm(x, w, b, eps=1e-12):
    mu = x.mean(-1, keepdim=True)
    var = ((x - mu) ** 2).mean(-1, keepdim=True)
    return (x - mu) / torch.sqrt(var + eps) * w + b


def softmax_last(x):
    e = torch.exp(x - x.max(-1, keepdim=True).values)
    return e / e.sum(-1, keepdim=True)


def distilbert_forward(W, ids):
    """W: state dict; ids: token ids. Returns mean-pooled hidden [768]."""
    x = W["embeddings.word_embeddings.weight"][ids] + \
        W["embeddings.position_embeddings.weight"][:len(ids)]
    x = layer_norm(x, W["embeddings.LayerNorm.weight"],
                   W["embeddings.LayerNorm.bias"]).unsqueeze(0)  # [1,S,D]
    n_layers = len({k.split(".")[2] for k in W
                    if k.startswith("transformer.layer")})
    n_heads, d_head = 12, x.shape[-1] // 12
    S = x.shape[1]
    for L in range(n_layers):
        p = f"transformer.layer.{L}."
        q = x @ W[p + "attention.q_lin.weight"].T + W[p + "attention.q_lin.bias"]
        k = x @ W[p + "attention.k_lin.weight"].T + W[p + "attention.k_lin.bias"]
        v = x @ W[p + "attention.v_lin.weight"].T + W[p + "attention.v_lin.bias"]
        q = q.view(1, S, n_heads, d_head).transpose(1, 2)
        k = k.view(1, S, n_heads, d_head).transpose(1, 2)
        v = v.view(1, S, n_heads, d_head).transpose(1, 2)
        att = softmax_last(q @ k.transpose(-1, -2) / (d_head ** 0.5)) @ v
        att = att.transpose(1, 2).reshape(1, S, d_head * n_heads)
        att = att @ W[p + "attention.out_lin.weight"].T + \
            W[p + "attention.out_lin.bias"]
        x = layer_norm(x + att, W[p + "sa_layer_norm.weight"],
                       W[p + "sa_layer_norm.bias"])
        h = torch.nn.functional.gelu(
                x @ W[p + "ffn.lin1.weight"].T + W[p + "ffn.lin1.bias"])
        h = h @ W[p + "ffn.lin2.weight"].T + W[p + "ffn.lin2.bias"]
        x = layer_norm(x + h, W[p + "output_layer_norm.weight"],
                       W[p + "output_layer_norm.bias"])
    return x.squeeze(0).mean(0)


def reference_embedding(model_dir, text):
    W = read_safetensors(model_dir + "/model.safetensors")
    proj = read_safetensors(model_dir + "/2_Dense/model.safetensors")["linear.weight"]
    with open(model_dir + "/vocab.txt", encoding="utf-8") as f:
        vocab = {line.rstrip("\n"): i for i, line in enumerate(f)}
    inv = {v: k for k, v in vocab.items()}
    pieces = tokenize(text, vocab)
    ids = [vocab[t] for t in pieces]
    pooled = distilbert_forward(W, ids)
    emb = torch.nn.functional.normalize(pooled @ proj.T, dim=-1)
    return {"tokens": ids, "pieces": pieces, "embed": emb.tolist()}


# ---------------------------------------------------------------------------
# GGUF packing
# ---------------------------------------------------------------------------

def quantize_q8_0(tensor):
    """GGML Q8_0: blocks of 32, f16 scale = max|w|/127, int8 rounded weights.
    Returns (uint8 block bytes, ggml-ne-ordered logical shape)."""
    f = tensor.to(torch.float32).reshape(-1, 32)
    amax = f.abs().amax(dim=1)
    d = amax.div(127.0).to(torch.float16)
    d_f32 = d.to(torch.float32)
    qs = torch.round(f / d_f32.unsqueeze(1)).clamp_(-127, 127).to(torch.int8)
    d_bytes = d.view(torch.uint8).numpy().reshape(-1, 2)
    qs_bytes = qs.contiguous().numpy().view(np.uint8).reshape(-1, 32)
    blocks = np.concatenate([d_bytes, qs_bytes], axis=1).reshape(-1)
    return blocks, tensor.shape


def pack(model_dir, out_path, dtype="f16", proj_override=None,
         target_space="clipb32", proj_bias=None):
    W = read_safetensors(model_dir + "/model.safetensors")
    proj = read_safetensors(model_dir + "/2_Dense/model.safetensors")["linear.weight"]
    with open(model_dir + "/vocab.txt", encoding="utf-8") as f:
        vocab = [line.rstrip("\n") for line in f]
    assert len(vocab) == 119547, len(vocab)
    assert vocab[0] == PAD and vocab[100] == UNK and vocab[101] == CLS \
        and vocab[102] == SEP, (vocab[0], vocab[100], vocab[101], vocab[102])

    w = gguf.GGUFWriter(out_path, "mclip")
    w.add_string("general.name", "mclip-labse-vitb32")
    w.add_string("mclip.arch", "distilbert")
    w.add_string("mclip.target_space", target_space)
    w.add_uint32("mclip.vocab_size", len(vocab))
    w.add_uint32("mclip.embed_dim", 768)
    w.add_uint32("mclip.out_dim", 512)
    w.add_uint32("mclip.n_layers", 6)
    w.add_uint32("mclip.n_heads", 12)
    w.add_uint32("mclip.max_pos", 512)
    w.add_string("mclip.cls_token", CLS)
    w.add_string("mclip.sep_token", SEP)
    w.add_string("mclip.pad_token", PAD)
    w.add_string("mclip.unk_token", UNK)
    w.add_array("mclip.vocab", vocab)

    def t(name, tensor, f16=True, quant=True):
        if dtype == "q8_0" and f16 and quant:
            # Quantized matrices feed ggml_mul_mat/get_rows directly (no
            # cast op — GGML_OP_CAST does not accept quantized types).
            # gguf-py expects the 2D byte shape (rows, row_bytes) with
            # row_bytes = ne0/32*34; it derives the element shape itself.
            blocks, _shape = quantize_q8_0(tensor)
            row_bytes = (tensor.shape[1] // 32) * 34
            w.add_tensor(name, blocks.reshape(tensor.shape[0], row_bytes),
                         raw_dtype=gguf.GGMLQuantizationType.Q8_0)
            return
        arr = tensor.to(torch.float16).numpy() if f16 else tensor.numpy()
        w.add_tensor(name, arr)

    t("mclip.word_emb", W["embeddings.word_embeddings.weight"])
    t("mclip.pos_emb", W["embeddings.position_embeddings.weight"],
      quant=False)  # f16: added to the F32 stream via cast (CAST rejects
                    # quantized types)
    t("mclip.emb_ln_w", W["embeddings.LayerNorm.weight"], f16=False)
    t("mclip.emb_ln_b", W["embeddings.LayerNorm.bias"], f16=False)
    if proj_bias is not None:
        w.add_tensor("mclip.proj_b", proj_bias.to(torch.float32).numpy())
    for L in range(6):
        p, q = f"transformer.layer.{L}.", f"mclip.l{L}."
        # Fused QKV projection matching clip::clip_self_attention's split
        # order (q | k | v along dim 0).
        t(q + "attn_in_w", torch.cat(
            [W[p + "attention.q_lin.weight"],
             W[p + "attention.k_lin.weight"],
             W[p + "attention.v_lin.weight"]], dim=0))
        t(q + "attn_in_b", torch.cat(
            [W[p + "attention.q_lin.bias"],
             W[p + "attention.k_lin.bias"],
             W[p + "attention.v_lin.bias"]], dim=0), f16=False)
        t(q + "attn_o_w", W[p + "attention.out_lin.weight"])
        t(q + "attn_o_b", W[p + "attention.out_lin.bias"], f16=False)
        t(q + "sa_ln_w", W[p + "sa_layer_norm.weight"], f16=False)
        t(q + "sa_ln_b", W[p + "sa_layer_norm.bias"], f16=False)
        t(q + "ffn_in_w", W[p + "ffn.lin1.weight"])
        t(q + "ffn_in_b", W[p + "ffn.lin1.bias"], f16=False)
        t(q + "ffn_out_w", W[p + "ffn.lin2.weight"])
        t(q + "ffn_out_b", W[p + "ffn.lin2.bias"], f16=False)
        t(q + "out_ln_w", W[p + "output_layer_norm.weight"], f16=False)
        t(q + "out_ln_b", W[p + "output_layer_norm.bias"], f16=False)
    if proj_override is not None:
        # Fitted bridge (e.g. the YOLOE variant): the OpenAI-CLIP-space
        # projection composed with a linear map into the detector's own
        # text space. Same single-matrix GGUF layout as the World bridge.
        t("mclip.proj_w", proj_override)
    else:
        t("mclip.proj_w", proj)  # [512, 768] — y = W @ pooled
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print("packed ->", out_path)


if __name__ == "__main__":
    if sys.argv[1] == "pack":
        pack(sys.argv[2], sys.argv[3],
             sys.argv[4] if len(sys.argv) > 4 else "f16")
    elif sys.argv[1] == "ref":
        r = reference_embedding(sys.argv[2], sys.argv[3])
        with open(sys.argv[4], "w") as f:
            json.dump(r, f)
        print("ref tokens:", r["pieces"])
        print("ref embed[0:6]:", [round(x, 5) for x in r["embed"][:6]])
