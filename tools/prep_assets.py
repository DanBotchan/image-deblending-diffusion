"""Copy paper figures into the project-page static folder as WebP."""
import shutil, sys
from pathlib import Path
from PIL import Image
import fitz

FIG = Path(sys.argv[1])          # PaperOverleaf/figures
OUT = Path(sys.argv[2]) / "static" / "img"
if OUT.exists():
    shutil.rmtree(OUT)

def put(src, dst, q=90):
    dst = OUT / dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.open(src).convert("RGB").save(dst.with_suffix(".webp"), "WEBP", quality=q, method=6)

# Teaser
t = FIG / "1_introduction/teaser/images"
for s, d in [("ground_truth/lemons", "src1"), ("ground_truth/cat", "src2"),
             ("results/lemons", "rec1"), ("results/cat", "rec2"),
             ("input/lemons_cat", "input"), ("input/lemons_cat_output", "recon"),
             ("alpha_maps/gt_alpha", "alpha_gt"), ("alpha_maps/pred_alpha", "alpha_pred")]:
    put(t / f"{s}.png", f"teaser/{d}")

# Main comparison
roots = {"seen": FIG / "4_experiments/baseline_evaluation", "unseen": FIG / "4_experiments/generalization"}
for dom, root in roots.items():
    for k, sc in [(1, "ks"), (2, "us"), (3, "sv")]:
        base = root / f"scenario{k}_images"
        for mdir in base.iterdir():
            for f in mdir.glob("*.png"):
                put(f, f"results/{dom}/{sc}/{mdir.name}/{f.stem}")

# Text guidance ablation
tg = FIG / "0_appendix/text_guidance/text_images"
for m, d in [("text_guidance_cross", "cross"), ("text_guidance_no_neg", "cfg"), ("text_guidance_classes", "classes")]:
    for w in ["0.0", "1.0", "2.0", "3.0"]:
        for n in ["apples", "cat", "fruits", "kangaro"]:
            put(tg / m / f"CFG_{w}" / f"{n}.png", f"text/{d}/{w[0]}/{n}")
for n in ["apples", "cat", "fruits", "kangaro"]:
    put(tg / "ground_truth" / f"{n}.png", f"text/gt/{n}")
put(tg / "input/apples_cat.png", "text/input/apples-cat")
put(tg / "input/fruits_kangaro.png", "text/input/fruits-kangaro")

# Two-stage ablation
g = FIG / "0_appendix/two_stage/Gallery"
for sub, d in [("stable", "stage1"), ("ddpm", "stage2"), ("ground_truth", "gt"), ("input", "input")]:
    for f in (g / sub).glob("*.png"):
        put(f, f"twostage/{d}/{f.stem}")

# Method figure
page = fitz.open(FIG / "3_method/latest_method_figure_pickscore.pdf")[0]
pix = page.get_pixmap(dpi=220)
tmp = OUT / "method.png"
pix.save(tmp)
Image.open(tmp).convert("RGB").save(OUT / "method.webp", "WEBP", quality=92, method=6)
tmp.unlink()
print("done")
