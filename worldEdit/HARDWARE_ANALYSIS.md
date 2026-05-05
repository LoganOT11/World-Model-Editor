# Laptop Specifications & LingBot-Map Runtime Estimate

**Date:** 2026-05-05

---

## 1. Hardware Specifications

| Component | Detail |
|-----------|--------|
| **CPU** | Intel Core i7-1260P (12th Gen Alder Lake) |
| | — 12 cores: 4 Performance (P-cores) + 8 Efficiency (E-cores) |
| | — 16 logical threads |
| | — Base: 2.1 GHz, Max Turbo: 4.7 GHz (P-cores) / 3.4 GHz (E-cores) |
| | — TDP: 28W (configurable 20–64W) |
| **GPU** | Intel Iris Xe Graphics (integrated) |
| | — 96 EU (Execution Units) |
| | — ~2 GB shared VRAM (from system RAM) |
| | — **No NVIDIA GPU. No CUDA.** |
| **RAM** | 16 GB LPDDR5 (likely 5200 MHz) |
| **Storage** | 1 TB NVMe SSD (KIOXIA KBG50ZNS1T02) |

---

## 2. LingBot-Map Pipeline — What It Actually Does

The full pipeline for a 30-second video:

```
Video (30s) → Frame extraction → DINOv2 ViT-L encoding → 
Temporal GCT transformer → Camera/depth/point heads → 
Post-process → Point cloud + Viser viewer
```

### Pipeline stages

| Stage | What happens | Hardware required |
|-------|-------------|-------------------|
| **1. Frame extraction** | `--fps 10` → 300 frames extracted from 30s video using OpenCV | CPU only (fast) |
| **2. Image preprocessing** | Resize + crop to 518×378, normalize to [-1, 1] | CPU (fast, ~2s) |
| **3. Sky masking** (optional) | ONNX skyseg model per frame (~300 frames) | CPU (~0.5s/frame on CPU) or CUDA |
| **4. Model inference** | DINOv2 ViT-Large backbone (~300M params) + GCT temporal transformer + 3 prediction heads | **CUDA GPU required for practical use** |
| **5. Post-processing** | Pose encoding → extrinsics/intrinsics, depth → world points, move to CPU | GPU/CPU hybrid |
| **6. Visualization** | Viser point cloud viewer at `localhost:8080` | CPU only |

---

## 3. Model Architecture & Size

| Property | Value |
|----------|-------|
| Backbone | DINOv2 ViT-Large with registers (`dinov2_vitl14_reg`) |
| Backbone params | ~300M (ViT-L: 24 layers, 1024-dim, 16 heads) |
| Checkpoint size | **4.63 GB** (`lingbot-map-long.pt`) |
| Input resolution | 518×378 (enforced by `patch_size=14` → must be divisible by 14) |
| Patches per frame | (518/14) × (378/14) = 37 × 27 = **999 patches** |
| Temporal blocks | Additional GCT transformer blocks over frame sequence |
| KV cache | Stores per-frame intermediate states; grows with sequence length |
| Heads | Camera pose (iterative, 4 refinement steps by default), depth, world points |

---

## 4. Runtime Estimate — 30-Second Video (300 frames at 10 FPS)

### 4A. With CUDA GPU (Reference — NOT this laptop)

The README reports **~20 FPS** at 518×378 on GPU. This would give:

| Stage | Time |
|-------|------|
| Frame extraction + preprocessing | ~5–10 s |
| Sky masking (optional, ONNX GPU) | ~5–10 s |
| Model inference (300 frames @ 20 FPS) | ~15 s |
| Post-processing + visualization | ~5–10 s |
| **Total (GPU)** | **~30–45 s** |

With `torch.compile` + FlashInfer: ~25–30 FPS → ~10–12 s inference → **~25–35 s total**.

### 4B. On This Laptop (Intel Iris Xe, CPU-only)

**Critical problem: No CUDA GPU.** The code falls back to `device='cpu'` and `dtype=float32`, but this is catastrophically slow for a ViT-Large model processing 300 frames:

| Factor | Impact |
|--------|--------|
| **Backbone size** | ViT-Large: 24 transformer layers, 1024-dim, ~300M params. Each forward pass computes 999 patches × 999 patches self-attention = ~1M attention pairs per layer per frame. |
| **CPU inference** | No GPU parallelism. Self-attention O(p²·d) per layer. 24 layers × 1M pairs × 1024-dim ≈ billions of FLOPs per frame. |
| **No bfloat16** | CPU runs fp32; no hardware bf16 acceleration. 2× memory, 2× compute. |
| **No FlashInfer** | Falls back to SDPA (PyTorch native attention), which is memory-bound on CPU. |
| **Temporal blocks** | Additional cross-frame attention grows with sequence length. |
| **KV cache** | Stores 300 frames × 999 patches × 1024-dim × fp32 × (K+V) ≈ 2.4 GB just for KV cache of one layer. 24 layers → impossible in 16GB RAM with model weights. |

**Estimated per-frame CPU time:**

| Component | Per-frame estimate |
|-----------|-------------------|
| ViT-L backbone forward pass (fp32, CPU) | 5–15 seconds |
| GCT temporal attention + heads | 2–5 seconds |
| **Total per frame** | **7–20 seconds** |

> This is based on: DINOv2 ViT-L takes ~3–5s/frame on a modern x86 CPU at 518×518 (reported in various benchmarks). The GCT temporal blocks add cross-frame KV cache operations which are also O(N²) in cached frames, growing linearly with sequence length. At 300 frames, KV cache overhead becomes dominant.

| Stage | Time |
|-------|------|
| Frame extraction + preprocessing | ~5 s |
| Sky masking (ONNX CPU, 300 frames) | ~150 s (2.5 min) |
| Model inference (300 frames × 7–20 s) | **2,100–6,000 s (35–100 min)** |
| Post-processing + visualization | ~10–30 s |
| **Total (CPU)** | **~40–105 minutes** |

### 4C. Realistic Assessment

**Running LingBot-Map on this laptop is not practical.** The CPU-only path exists in code but is designed for testing, not production use. Even if it completes:

- **Memory:** 16 GB RAM is insufficient. The model alone is 4.63 GB. KV cache for 300 frames in fp32 needs 10+ GB. System + other processes consume 4-6 GB. Total: ~20 GB required, exceeding 16 GB → **will swap to disk, making it even slower**.
- **Time:** 40–105 minutes for a 30-second video is 80–210× slower than real-time.
- **Reliability:** The KV cache growth with frame count may cause memory allocation failures on CPU.

---

## 5. What Would Work?

### Option A: Cloud GPU (Recommended)

| Provider | GPU | Cost | Time estimate |
|----------|-----|------|---------------|
| Google Colab Pro+ | T4 / L4 / A100 | ~$10–50/mo | 30–45 s |
| Lambda Labs | A10 / A100 | ~$0.50–1.10/hr | 30–45 s |
| RunPod | RTX 4090 / A6000 | ~$0.44–0.79/hr | 25–35 s |
| HuggingFace Spaces | Free tier (T4) | Free | 45–60 s |

**Action:** Upload video + model to cloud GPU, run `demo.py`, download point cloud. This is the recommended path.

### Option B: External eGPU

Thunderbolt 4 eGPU enclosure + RTX 3060/4060 (~$400–600). Would require driver setup on Windows but provides local CUDA access.

### Option C: Reduce Scope for CPU

If cloud/eGPU is not possible:
- **Extract far fewer frames:** `--fps 1` → 30 frames instead of 300 → ~4–10 min CPU
- **Use `--first_k 20`** to test on just 20 frames → ~2–4 min
- **Skip sky masking** (`--mask_sky` adds significant CPU overhead)
- **Use `--camera_num_iterations 1`** to reduce head computation
- **Use `--use_sdpa`** (already forced on CPU)
- **Use `--offload_to_cpu`** (already forced on CPU)

Even with these optimizations, CPU inference is measured in minutes, not seconds.

---

## 6. Verification — Quick CPU Test

To confirm these estimates, run a micro-benchmark on 5 frames:

```bash
cd lingbot-map
python demo.py \
  --model_path /path/to/lingbot-map-long.pt \
  --image_folder example/courthouse \
  --first_k 5 \
  --camera_num_iterations 1 \
  --mask_sky
```

This should complete in ~1–3 minutes on CPU and give a per-frame baseline. Multiply by 60 for 300-frame estimate.

---

## 7. Summary

| Metric | Value |
|--------|-------|
| Laptop GPU | Intel Iris Xe — **no CUDA** |
| LingBot-Map GPU requirement | CUDA GPU (NVIDIA) — **not met** |
| CPU fallback available? | Yes, but impractically slow |
| 30s video, CPU estimate | **40–105 minutes** (if it doesn't OOM) |
| 30s video, GPU estimate (cloud) | **30–45 seconds** |
| Recommendation | Use **Google Colab** (free T4) or **RunPod** ($0.50/hr) |
| LingBot-Map checkpoint | ~4.63 GB (`lingbot-map-long.pt` from HuggingFace) |
| 300 frames @ 10 FPS | ~300 frames to process |

---

*Analysis based on: actual hardware probing + LingBot-Map README + DINOv2 ViT-L benchmarks + transformer FLOP estimation.*
