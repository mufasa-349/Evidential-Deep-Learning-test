"""
Evidential AnomalyDINO (few-shot, unsupervised) - PyTorch reference implementation.

Core idea:
- Extract DINOv2 patch embeddings from a few NORMAL images
- Build a memory bank of patch features
- For a test image: nearest-neighbor distance per patch -> evidence via RBF
- Map evidence to Dirichlet parameters for Normal vs Unknown/Anomaly
- Produce belief(normal) and epistemic uncertainty maps

Based on the architecture described in:
"Evidential Deep Learning for Few-Shot Unsupervised Anomaly Detection" (uploaded PDF).
"""

import os
# Fix OpenMP conflict on macOS (multiple OpenMP runtimes)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

# -----------------------------
# Config
# -----------------------------

@dataclass
class Config:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    image_size: int = 518  # DINOv2 commonly uses 518, but any multiple-ish works
    patch_size: int = 14   # dinov2_vits14 -> 14x14 patches
    dinov2_repo: str = "facebookresearch/dinov2"
    dinov2_model: str = "dinov2_vits14"  # ViT-S/14
    use_faiss: bool = True               # auto-fallback if faiss not available

    # Evidence mapping: evidence = exp(-gamma * d2)
    # d2 is squared L2 distance from FAISS IndexFlatL2 (or computed manually)
    gamma: float = 1.0

    # Scoring aggregation for image-level score (optional)
    agg: str = "max"  # "max" or "mean" on uncertainty map

    # Output
    out_dir: str = "outputs"
    save_heatmaps: bool = True


# -----------------------------
# Utilities
# -----------------------------

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def list_images(folder: str) -> List[str]:
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
    paths = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(exts):
                paths.append(os.path.join(root, f))
    return sorted(paths)

def pil_load_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")

def normalize_map(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # x: (H, W)
    xmin = x.min()
    xmax = x.max()
    return (x - xmin) / (xmax - xmin + eps)

def to_uint8_img(x01: torch.Tensor) -> Image.Image:
    # x01: (H, W) in [0,1]
    arr = (x01.clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr)

def overlay_heatmap_on_image(img: Image.Image, heat01: torch.Tensor, alpha: float = 0.45) -> Image.Image:
    """
    Simple grayscale overlay (keeps dependencies minimal).
    If you want fancy colormaps, we can add matplotlib later.
    """
    base = np.array(img).astype(np.float32)
    h = np.array(to_uint8_img(heat01)).astype(np.float32)  # 0..255
    h3 = np.stack([h, h, h], axis=-1)
    out = (1 - alpha) * base + alpha * h3
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


# -----------------------------
# Feature Extractor (DINOv2)
# -----------------------------

class FeatureExtractor(torch.nn.Module):
    """
    Extract patch tokens:
      model.forward_features(x)["x_norm_patchtokens"] -> (B, N_patches, D)

    PDF emphasizes using normalized patch tokens for stable Euclidean distances.
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        # Preprocess: Resize + CenterCrop-ish + ImageNet normalization
        self.tf = transforms.Compose([
            transforms.Resize(cfg.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(cfg.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
        ])

        # Load DINOv2 backbone
        # NOTE: requires internet on first run unless cached.
        self.backbone = torch.hub.load(cfg.dinov2_repo, cfg.dinov2_model)
        self.backbone.eval().to(self.device)

        for p in self.backbone.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def extract_patches(self, pil_img: Image.Image) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Returns:
          patches: (N_patches, D)
          grid_hw: (grid_h, grid_w) where grid_h = H/patch_size
        """
        x = self.tf(pil_img).unsqueeze(0).to(self.device)  # (1,3,H,W)

        feats = self.backbone.forward_features(x)
        patch_tokens = feats["x_norm_patchtokens"]  # (1, N, D)
        patch_tokens = patch_tokens.squeeze(0)      # (N, D)

        # Infer grid size from image_size and patch_size
        grid = self.cfg.image_size // self.cfg.patch_size
        grid_hw = (grid, grid)

        return patch_tokens, grid_hw


# -----------------------------
# Memory Bank (FAISS optional)
# -----------------------------

class EvidentialMemoryBank:
    """
    Stores patch embeddings from normal images.
    Provides nearest-neighbor squared L2 distance for query patch embeddings.
    """
    def __init__(self, dim: int, use_faiss: bool = True, device: str = "cpu"):
        self.dim = dim
        self.use_faiss = use_faiss
        self.device = device

        self._faiss = None
        self.index = None
        self._features = None  # fallback tensor store

        if use_faiss:
            try:
                import faiss
                self._faiss = faiss
                self.index = faiss.IndexFlatL2(dim)  # returns squared L2 distances
            except Exception as e:
                print(f"[WARN] faiss unavailable or failed to import ({e}). Falling back to torch NN.")
                self.use_faiss = False

        if not self.use_faiss:
            self._features = torch.empty((0, dim), dtype=torch.float32, device=self.device)

    def add(self, feats: torch.Tensor):
        """
        feats: (N, D) float tensor (on any device)
        """
        feats = feats.detach().float()
        if self.use_faiss:
            feats_np = feats.cpu().numpy().astype(np.float32)
            self.index.add(feats_np)
        else:
            feats = feats.to(self._features.device)
            self._features = torch.cat([self._features, feats], dim=0)

    def search_nn_dist2(self, query_feats: torch.Tensor) -> torch.Tensor:
        """
        query_feats: (N, D)
        returns dist2: (N,) squared L2 to nearest neighbor
        """
        query_feats = query_feats.detach().float()

        if self.use_faiss:
            q = query_feats.cpu().numpy().astype(np.float32)
            dist2, _ = self.index.search(q, k=1)  # (N,1)
            return torch.from_numpy(dist2[:, 0]).to(query_feats.device)
        else:
            # torch fallback: compute NN distance with chunking for memory safety
            feats = self._features  # (M,D)
            if feats.shape[0] == 0:
                raise RuntimeError("Memory bank is empty. Add normal features first.")

            q = query_feats.to(feats.device)
            # chunked cdist to avoid O(N*M) memory blowups
            chunk = 2048
            out = []
            for i in range(0, q.shape[0], chunk):
                qi = q[i:i+chunk]  # (c,D)
                # cdist -> (c,M), squared L2:
                d = torch.cdist(qi, feats, p=2) ** 2
                out.append(d.min(dim=1).values)
            dist2 = torch.cat(out, dim=0)
            return dist2.to(query_feats.device)


# -----------------------------
# Evidential inference (Dirichlet logic)
# -----------------------------

class EDLInference(torch.nn.Module):
    """
    Maps NN distances to evidence and then to belief/uncertainty.

    Binary-like setup:
      alpha_normal = evidence + 1
      alpha_anomaly = 1 (fixed prior; "unknown" bucket)
      S = alpha_normal + alpha_anomaly = evidence + 2

      belief_normal = evidence / S
      uncertainty = 2 / S
    """
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = float(gamma)

    @torch.no_grad()
    def forward(self, dist2: torch.Tensor, grid_hw: Tuple[int, int], out_hw: Tuple[int, int]) -> dict:
        """
        dist2: (N_patches,) squared L2 distance
        grid_hw: (gh, gw)
        out_hw: (H, W) target size for upsampled maps
        """
        # Evidence
        evidence = torch.exp(-self.gamma * dist2)  # (N,)

        # Dirichlet
        alpha_norm = evidence + 1.0
        S = alpha_norm + 1.0  # + alpha_anom where alpha_anom=1
        belief = evidence / S
        uncertainty = 2.0 / S

        gh, gw = grid_hw
        belief_map = belief.view(gh, gw)
        unc_map = uncertainty.view(gh, gw)

        # Upsample to image resolution
        belief_map_up = F.interpolate(belief_map[None, None, ...], size=out_hw, mode="bilinear", align_corners=False)[0, 0]
        unc_map_up = F.interpolate(unc_map[None, None, ...], size=out_hw, mode="bilinear", align_corners=False)[0, 0]

        return {
            "evidence": evidence,
            "belief_map": belief_map_up,
            "uncertainty_map": unc_map_up,
            "belief_map_lowres": belief_map,
            "uncertainty_map_lowres": unc_map,
        }


# -----------------------------
# Pipeline
# -----------------------------

def build_memory_bank(cfg: Config, extractor: FeatureExtractor, normal_dir: str) -> EvidentialMemoryBank:
    paths = list_images(normal_dir)
    if len(paths) == 0:
        raise RuntimeError(f"No images found in: {normal_dir}")

    print(f"[INFO] Building memory bank from {len(paths)} normal images...")
    first_feats, _ = extractor.extract_patches(pil_load_rgb(paths[0]))
    dim = first_feats.shape[-1]

    bank = EvidentialMemoryBank(dim=dim, use_faiss=cfg.use_faiss, device=cfg.device)
    bank.add(first_feats)

    for p in paths[1:]:
        feats, _ = extractor.extract_patches(pil_load_rgb(p))
        bank.add(feats)

    print("[INFO] Memory bank ready.")
    return bank

@torch.no_grad()
def infer_one(cfg: Config, extractor: FeatureExtractor, bank: EvidentialMemoryBank, edl: EDLInference, img_path: str):
    img = pil_load_rgb(img_path)

    # We need the post-crop size for upsampling
    # Extractor always center-crops to image_size x image_size
    out_hw = (cfg.image_size, cfg.image_size)

    patches, grid_hw = extractor.extract_patches(img)          # (N,D)
    dist2 = bank.search_nn_dist2(patches)                      # (N,)
    out = edl(dist2=dist2, grid_hw=grid_hw, out_hw=out_hw)     # maps

    # Image-level score (optional)
    unc = out["uncertainty_map"]
    if cfg.agg == "max":
        score = float(unc.max().item())
    elif cfg.agg == "mean":
        score = float(unc.mean().item())
    else:
        raise ValueError("cfg.agg must be 'max' or 'mean'")

    return img, out, score

def run(cfg: Config, normal_dir: str, test_dir: str):
    ensure_dir(cfg.out_dir)

    extractor = FeatureExtractor(cfg)
    bank = build_memory_bank(cfg, extractor, normal_dir)
    edl = EDLInference(gamma=cfg.gamma).to(cfg.device).eval()

    test_paths = list_images(test_dir)
    if len(test_paths) == 0:
        raise RuntimeError(f"No images found in: {test_dir}")

    print(f"[INFO] Running inference on {len(test_paths)} test images...")
    results = []
    for p in test_paths:
        img, out, score = infer_one(cfg, extractor, bank, edl, p)
        results.append((p, score))

        if cfg.save_heatmaps:
            # normalize maps for saving
            unc01 = normalize_map(out["uncertainty_map"])
            bel01 = normalize_map(out["belief_map"])

            unc_img = to_uint8_img(unc01)
            bel_img = to_uint8_img(bel01)

            # overlay
            # NOTE: overlay uses the cropped/resized view; for exact original-size overlay,
            # we’d need to keep transform parameters and map back.
            base_cropped = transforms.Compose([
                transforms.Resize(cfg.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(cfg.image_size),
            ])(img)

            overlay_unc = overlay_heatmap_on_image(base_cropped, unc01, alpha=0.45)

            stem = os.path.splitext(os.path.basename(p))[0]
            unc_img.save(os.path.join(cfg.out_dir, f"{stem}_uncertainty.png"))
            bel_img.save(os.path.join(cfg.out_dir, f"{stem}_belief.png"))
            overlay_unc.save(os.path.join(cfg.out_dir, f"{stem}_overlay_uncertainty.png"))

        print(f"  - {os.path.basename(p)} | uncertainty_score({cfg.agg}) = {score:.4f}")

    # Save a simple ranking text
    results.sort(key=lambda x: x[1], reverse=True)
    with open(os.path.join(cfg.out_dir, "ranking.txt"), "w", encoding="utf-8") as f:
        for p, s in results:
            f.write(f"{s:.6f}\t{p}\n")

    print("[INFO] Done. Outputs saved to:", cfg.out_dir)


if __name__ == "__main__":
    # Example usage:
    #   python evidential_anomalydino.py --normal_dir path/to/normal --test_dir path/to/test
    #
    # Minimal CLI-free demo (edit these):
    normal_dir = "data/normal"
    test_dir = "data/test"

    cfg = Config(
        device="cuda" if torch.cuda.is_available() else "cpu",
        image_size=518,
        patch_size=14,
        dinov2_model="dinov2_vits14",
        use_faiss=False,  # Disabled due to macOS OpenMP conflicts; using PyTorch fallback
        gamma=0.01,  # Adjusted: distance^2 values are ~100-400, so smaller gamma needed
        agg="max",
        out_dir="outputs",
        save_heatmaps=True,
    )

    run(cfg, normal_dir=normal_dir, test_dir=test_dir)
