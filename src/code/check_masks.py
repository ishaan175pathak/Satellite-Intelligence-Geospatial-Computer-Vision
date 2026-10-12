import glob, os, random
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

proc = sorted(glob.glob("data/processed/whu_building/train/*.pt"))
fracs = []
for f in proc:
    _, m = torch.load(f, weights_only=True)
    fracs.append(m.mean().item())
fracs = np.array(fracs)

print("overall mean building fraction:", fracs.mean())
print("tiles with buildings:", (fracs > 0).sum(), "/", len(fracs))
nz = np.nonzero(fracs)[0]
print("first nonempty index:", nz[0], "| last empty-run end near:", nz[:5])

# Compare raw image, raw mask, processed mask for 3 early tiles and 3 nonempty tiles
picks = list(range(3)) + random.sample(list(nz), 3)
fig, ax = plt.subplots(len(picks), 3, figsize=(9, 3 * len(picks)))
for r, i in enumerate(picks):
    name = os.path.basename(proc[i])[:-3]  # strip ".pt"
    img = Image.open(f"data/raw/whu_building/train/Image/{name}")
    msk = Image.open(f"data/raw/whu_building/train/Mask/{name}")
    _, pm = torch.load(proc[i], weights_only=True)
    ax[r, 0].imshow(img); ax[r, 0].set_title(f"raw image {name}", fontsize=7)
    ax[r, 1].imshow(np.asarray(msk), cmap="gray"); ax[r, 1].set_title("raw mask", fontsize=7)
    ax[r, 2].imshow(pm, cmap="gray"); ax[r, 2].set_title(f"processed mask ({fracs[i]:.3f})", fontsize=7)
    for a in ax[r]: a.axis("off")
plt.tight_layout(); plt.savefig("mask_check.png", dpi=120)