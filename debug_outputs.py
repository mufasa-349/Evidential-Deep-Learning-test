"""
Debug script to check actual uncertainty and belief values before normalization.
"""
import os
import torch
import numpy as np
from PIL import Image
import sys
sys.path.insert(0, os.path.dirname(__file__))

from evidential_anomalydino import (
    Config, FeatureExtractor, EvidentialMemoryBank, 
    EDLInference, pil_load_rgb, list_images
)

# Load config
cfg = Config(
    device="cuda" if torch.cuda.is_available() else "cpu",
    image_size=518,
    patch_size=14,
    dinov2_model="dinov2_vits14",
    use_faiss=False,
    gamma=0.01,
)

print("Loading model and building memory bank...")
extractor = FeatureExtractor(cfg)

normal_paths = list_images("data/normal")
first_feats, _ = extractor.extract_patches(pil_load_rgb(normal_paths[0]))
dim = first_feats.shape[-1]
bank = EvidentialMemoryBank(dim=dim, use_faiss=False, device=cfg.device)
bank.add(first_feats)
for p in normal_paths[1:]:
    feats, _ = extractor.extract_patches(pil_load_rgb(p))
    bank.add(feats)

edl = EDLInference(gamma=cfg.gamma).to(cfg.device).eval()

# Test one normal and one anomaly image
test_normal = "data/test/test_normal_00.png"
test_anom = "data/test/test_anom_00.png"

for name, path in [("NORMAL", test_normal), ("ANOMALI", test_anom)]:
    print(f"\n{'='*60}")
    print(f"{name} GÖRÜNTÜ: {os.path.basename(path)}")
    print(f"{'='*60}")
    
    img = pil_load_rgb(path)
    patches, grid_hw = extractor.extract_patches(img)
    dist2 = bank.search_nn_dist2(patches)
    
    print(f"Distance^2 stats:")
    print(f"  Min: {dist2.min().item():.6f}")
    print(f"  Max: {dist2.max().item():.6f}")
    print(f"  Mean: {dist2.mean().item():.6f}")
    print(f"  Median: {dist2.median().item():.6f}")
    
    out = edl(dist2=dist2, grid_hw=grid_hw, out_hw=(cfg.image_size, cfg.image_size))
    
    evidence = out["evidence"]
    unc_map = out["uncertainty_map"]
    bel_map = out["belief_map"]
    
    print(f"\nEvidence stats:")
    print(f"  Min: {evidence.min().item():.6f}")
    print(f"  Max: {evidence.max().item():.6f}")
    print(f"  Mean: {evidence.mean().item():.6f}")
    
    print(f"\nUncertainty map stats (BEFORE normalization):")
    print(f"  Min: {unc_map.min().item():.6f}")
    print(f"  Max: {unc_map.max().item():.6f}")
    print(f"  Mean: {unc_map.mean().item():.6f}")
    print(f"  Unique values count: {len(torch.unique(unc_map))}")
    
    print(f"\nBelief map stats (BEFORE normalization):")
    print(f"  Min: {bel_map.min().item():.6f}")
    print(f"  Max: {bel_map.max().item():.6f}")
    print(f"  Mean: {bel_map.mean().item():.6f}")
    
    # Image-level score
    score_max = float(unc_map.max().item())
    score_mean = float(unc_map.mean().item())
    print(f"\nImage-level scores:")
    print(f"  Max uncertainty: {score_max:.6f}")
    print(f"  Mean uncertainty: {score_mean:.6f}")

