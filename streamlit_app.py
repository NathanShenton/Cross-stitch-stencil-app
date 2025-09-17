# app.py — Photo → Cropped → Cross-stitch pattern (grid + per-cell code + PDF)
import io
import math
from dataclasses import dataclass
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
import streamlit as st
import matplotlib.pyplot as plt

# OPTIONAL: in-app cropper (touch-friendly)
# pip install streamlit-cropper
try:
    from streamlit_cropper import st_cropper
    HAS_CROPPER = True
except Exception:
    HAS_CROPPER = False

# ----------------------------- Page & Intro -----------------------------
st.set_page_config(page_title="Photo → Cross-Stitch Pattern", layout="wide")
st.title("🧵 Photo → Cross-Stitch Pattern")
st.caption("Upload, crop, quantise to stitch-friendly palette, render a counted chart, and export a PDF.")

# ----------------------------- Sidebar Controls -----------------------------
with st.sidebar:
    st.header("1) Input & Crop")
    uploaded = st.file_uploader("Upload a photo (PNG/JPG)", type=["png", "jpg", "jpeg"])
    st.help("Tip: Use a clear, high-contrast image. You can crop in the next step.")

    st.header("2) Sizing (Aida)")
    aida_count = st.number_input("Aida count (stitches / inch)", 6, 36, 14, 1)
    size_mode = st.radio("Set size by…", ["Inches", "Stitches"], index=0)
    if size_mode == "Inches":
        width_in = st.number_input("Width (inches)", 1.0, 60.0, 7.0, 0.5)
        height_in = st.number_input("Height (inches) (0 = keep aspect)", 0.0, 60.0, 0.0, 0.5)
        target_inches = (width_in, height_in)
        target_stitches = None
    else:
        target_w_st = st.number_input("Width (stitches)", 20, 800, 160, 10)
        target_h_st = st.number_input("Height (stitches) (0 = keep aspect)", 0, 800, 0, 10)
        target_stitches = (target_w_st, target_h_st)
        target_inches = None

    st.header("3) Palette / Style")
    colour_mode = st.radio("Style", ["Colour", "Greyscale"], index=0)
    max_colors = st.slider("Max colours (colour mode)", 2, 60, 25, 1)
    use_dither = st.checkbox("Floyd–Steinberg dither", value=False,
                             help="Can smooth gradients but may add speckle.")
    use_thread_palette = st.checkbox("Map to thread palette (DMC subset)", value=True)

    st.header("4) Pre-processing")
    brightness = st.slider("Brightness", 0.2, 2.0, 1.0, 0.05)
    contrast   = st.slider("Contrast",   0.2, 2.0, 1.0, 0.05)

    st.header("5) Grid & Output")
    bold10 = st.checkbox("Bold every 10 stitches", value=True)
    number_every = st.selectbox("Numbering interval", [5, 10], index=1)
    cell_px = st.slider("On-screen cell size (px)", 6, 30, 14, 1)
    label_text = st.checkbox("Show colour code text in each cell", value=True)
    grid_shape = st.radio("Stitching area shape", ["Rectangle", "Square", "Circle"], index=0)
    pad_shape = st.slider("Shape margin (%)", 0, 20, 0, 1, help="Leave margin inside the shape mask")

# ----------------------------- Thread Palette (DMC subset demo) -----------------------------
# Minimal demo palette: add more rows or replace with your full DMC list (RGB + Code)
DMC_SUBSET = pd.DataFrame([
    #  Code,     R,   G,   B,  Name (optional)
    ("310",       0,   0,   0,  "Black"),
    ("762",     236, 236, 236,  "Pearl Gray VLT"),
    ("415",     211, 211, 214,  "Pearl Gray"),
    ("318",     171, 171, 171,  "Steel Gray LT"),
    ("317",     108, 108, 108,  "Pewter Gray"),
    ("B5200",   255, 255, 255,  "Snow White"),
    ("321",     199,  43,  59,  "Red"),
    ("666",     227,  29,  66,  "Bright Red"),
    ("699",       0,  92,   9,  "Green"),
    ("703",     123, 181,  71,  "Chartreuse"),
    ("797",      19,  71, 125,  "Royal Blue"),
    ("996",      55, 164, 199,  "Electric Blue"),
    ("742",     255, 191,  87,  "Tangerine LT"),
    ("727",     255, 241, 193,  "Topaz VLT"),
    ("743",     254, 196,  53,  "Yellow MD")
], columns=["Code", "R", "G", "B", "Name"])

def _rgb_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # Fast Euclidean distance per-row (b is Nx3)
    # a shape: (..., 3)
    a2 = a.reshape(-1, 3).astype(np.float32)
    b2 = b.astype(np.float32)
    # (Npx x 3) - (Npals x 3)
    # compute squared distance: (x-y)^2 across last axis
    dists = np.sqrt(((a2[:, None, :] - b2[None, :, :]) ** 2).sum(axis=2))
    return dists

# ----------------------------- Helpers -----------------------------
def compute_target_wh(im: Image.Image) -> Tuple[int, int]:
    w, h = im.size
    aspect = w / h
    if target_inches is not None:
        win, hin = target_inches
        if hin == 0.0:
            tw = int(round(win * aida_count))
            th = int(round(tw / aspect))
        else:
            tw = int(round(win * aida_count))
            th = int(round(hin * aida_count))
    else:
        tw, th = target_stitches
        if th == 0:
            th = int(round(tw / aspect))
    tw = int(np.clip(tw, 10, 2000))
    th = int(np.clip(th, 10, 2000))
    return tw, th

def apply_preproc(im: Image.Image) -> Image.Image:
    im = ImageEnhance.Brightness(im).enhance(brightness)
    im = ImageEnhance.Contrast(im).enhance(contrast)
    if colour_mode == "Greyscale":
        im = im.convert("L").convert("RGB")
    return im

def adaptive_quantize(im: Image.Image, max_cols: int, dither: bool) -> Image.Image:
    dmode = Image.Dither.FLOYDSTEINBERG if dither else Image.Dither.NONE
    return im.convert("P", palette=Image.ADAPTIVE, colors=max_cols, dither=dmode).convert("RGB")

def map_to_palette(arr_rgb: np.ndarray, palette_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (indices grid HxW, palette_rgb Nx3 used)."""
    H, W = arr_rgb.shape[:2]
    d = _rgb_dist(arr_rgb, palette_rgb)  # (H*W x Npal)
    idx = d.argmin(axis=1).reshape(H, W)
    return idx, palette_rgb

def rgb_to_hex(rgb: List[int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)

def make_shape_mask(H: int, W: int, shape: str, margin_pct: int) -> np.ndarray:
    """1=inside stitching area, 0=masked out."""
    mask = np.ones((H, W), dtype=np.uint8)
    if shape == "Rectangle":
        if margin_pct > 0:
            mH = int(H * margin_pct / 100)
            mW = int(W * margin_pct / 100)
            mask[:mH, :] = 0; mask[-mH:, :] = 0
            mask[:, :mW] = 0; mask[:, -mW:] = 0
        return mask
    if shape == "Square":
        side = min(H, W)
        side = int(side * (100 - margin_pct) / 100)
        top = (H - side)//2; left = (W - side)//2
        mask[:] = 0
        mask[top:top+side, left:left+side] = 1
        return mask
    if shape == "Circle":
        yy, xx = np.ogrid[:H, :W]
        cy, cx = (H-1)/2.0, (W-1)/2.0
        r = min(H, W) * (100 - margin_pct) / 200.0
        dist = np.sqrt((yy - cy)**2 + (xx - cx)**2)
        mask = (dist <= r).astype(np.uint8)
        return mask
    return mask

# ----------------------------- Main Flow -----------------------------
if not uploaded:
    st.info("Upload an image to begin.")
    st.stop()

orig = Image.open(uploaded).convert("RGB")

st.subheader("1) Crop")
if HAS_CROPPER:
    st.caption("Drag to crop. Double-tap handles on mobile.")
    cropped = st_cropper(
    orig,
    realtime_update=True,
    aspect_ratio=None,
    box_color="#00FFAA",
    return_type="image",  # ✅ valid: "image" or "array"
    key="cropper",
)
else:
    st.warning("`streamlit-cropper` not installed — showing original. "
               "Install with `pip install streamlit-cropper` for interactive cropping.")
    cropped = orig

# Preprocess
pre = apply_preproc(cropped)

# Compute target size in stitches
target_w, target_h = compute_target_wh(pre)

# Resize so that each pixel = 1 stitch
resized = pre.resize((target_w, target_h), Image.Resampling.LANCZOS)

# Build palette/indices
arr = np.array(resized, dtype=np.uint8)  # HxWx3

if colour_mode == "Greyscale":
    # fixed greys: 6/12/20 levels depending on max_colors
    levels = max(2, min(max_colors, 20))
    gray = np.dot(arr[..., :3], [0.2126, 0.7152, 0.0722]).astype(np.uint8)
    bins = np.linspace(0, 255, levels, endpoint=True)
    # palette centers at bin edges; compute midpoints for better visual
    mids = ((bins[:-1] + bins[1:]) / 2).astype(np.uint8)
    palette_cols = np.stack([mids, mids, mids], axis=1)
    # map gray → nearest mid
    d = _rgb_dist(np.stack([gray, gray, gray], axis=2), palette_cols)
    idx = d.argmin(axis=1).reshape(gray.shape)
else:
    if use_thread_palette:
        pal = DMC_SUBSET[["R", "G", "B"]].to_numpy()
        idx, palette_cols = map_to_palette(arr, pal)
    else:
        q = adaptive_quantize(Image.fromarray(arr), max_colors=max_colors, dither=use_dither)
        arr_q = np.array(q)
        palette_cols, counts = np.unique(arr_q.reshape(-1, 3), axis=0, return_counts=True)
        order = np.argsort(-counts)
        palette_cols = palette_cols[order]
        # remap to contiguous indices
        lut = {tuple(rgb): i for i, rgb in enumerate(palette_cols.tolist())}
        Hh, Ww = arr_q.shape[:2]
        idx = np.zeros((Hh, Ww), dtype=np.int32)
        for y in range(Hh):
            for x in range(Ww):
                idx[y, x] = lut[tuple(arr_q[y, x])]

H, W = idx.shape

# Shape mask (only show cells inside shape)
mask = make_shape_mask(H, W, grid_shape, pad_shape)

# Legend / codes
if colour_mode == "Greyscale" or not use_thread_palette:
    # Simple A, B, C... / S26+ scheme
    symbols = [chr(65+i) if i < 26 else f"S{i}" for i in range(len(palette_cols))]
    legend = pd.DataFrame({
        "Code": symbols,
        "RGB": [tuple(map(int, c)) for c in palette_cols],
        "Hex": [rgb_to_hex(c) for c in palette_cols],
        "Count": [(idx == i).sum() for i in range(len(palette_cols))]
    })
    # label text grid uses these symbols
    cell_labels = np.vectorize(lambda i: symbols[i])(idx)
else:
    # Thread codes from DMC subset
    pal_codes = DMC_SUBSET["Code"].tolist()
    pal_rgb = DMC_SUBSET[["R","G","B"]].to_numpy()
    # Find which palette entries are actually used
    used = np.unique(idx)
    code_map = {i: pal_codes[i] for i in used}
    hexs = [rgb_to_hex(pal_rgb[i]) for i in used]
    legend = pd.DataFrame({
        "Code": [pal_codes[i] for i in used],
        "Hex": hexs,
        "Count": [(idx == i).sum() for i in used]
    })
    # labels are DMC codes
    cell_labels = np.vectorize(lambda i: pal_codes[i])(idx)

# Apply mask to labels: outside shape → empty
cell_labels_masked = cell_labels.copy()
cell_labels_masked[mask == 0] = ""

# ----------------------------- Preview & Dimensions -----------------------------
st.subheader("2) Preview & Dimensions")
colA, colB = st.columns([1,1])
with colA:
    st.image(cropped, caption="Cropped", use_column_width=True)
with colB:
    st.image(resized, caption=f"Resized to grid: {W}×{H} stitches", use_column_width=True)
st.caption(
    f"Approx stitched size on {aida_count}-count Aida: "
    f"{W/aida_count:.2f} × {H/aida_count:.2f} inches."
)

# ----------------------------- Chart Renderer (PNG in app + PDF export) -----------------------------
def render_chart_png(idx_grid: np.ndarray,
                     palette: np.ndarray,
                     labels: np.ndarray,
                     show_text=True,
                     cell_px=12,
                     bold10=True,
                     number_every=10,
                     mask_area: Optional[np.ndarray]=None) -> Image.Image:
    H, W = idx_grid.shape
    fig_w = W * cell_px / 100
    fig_h = H * cell_px / 100
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)
    # colour image
    img_rgb = palette[idx_grid]
    # mask outside shape to white
    if mask_area is not None:
        img_rgb = img_rgb.copy()
        img_rgb[mask_area == 0] = (255, 255, 255)
    ax.imshow(img_rgb, interpolation="nearest")
    # micro grid
    ax.set_xticks(np.arange(-.5, W, 1), minor=True)
    ax.set_yticks(np.arange(-.5, H, 1), minor=True)
    ax.grid(which="minor", linewidth=0.4, color="0.75")
    # bold every 10
    if bold10:
        ax.set_xticks(np.arange(-.5, W, 10))
        ax.set_yticks(np.arange(-.5, H, 10))
        ax.grid(which="both", linewidth=1.3, color="0.25")

    # numbers
    ax.set_xlim([-0.5, W-0.5]); ax.set_ylim([H-0.5, -0.5])
    ax.set_xticklabels([]); ax.set_yticklabels([])
    ax.tick_params(left=False, bottom=False)
    for x in range(0, W, number_every):
        ax.text(x, -1.1, f"{x}", ha="center", va="top", fontsize=6)
        ax.text(x, H,   f"{x}", ha="center", va="bottom", fontsize=6)
    for y in range(0, H, number_every):
        ax.text(-1.1, y, f"{y}", ha="right", va="center", fontsize=6)
        ax.text(W,   y, f"{y}", ha="left",  va="center", fontsize=6)

    # border
    ax.add_patch(plt.Rectangle((-0.5, -0.5), W, H, fill=False, lw=1.5, ec="0.1"))

    # per-cell labels
    if show_text:
        # pick contrasting text colour
        text_color = "black"
        for y in range(H):
            for x in range(W):
                if mask_area is not None and mask_area[y, x] == 0:
                    continue
                lab = labels[y, x]
                if not lab:
                    continue
                # choose white text on dark squares
                r, g, b = palette[idx_grid[y, x]]
                lum = 0.2126*r + 0.7152*g + 0.0722*b
                tc = "white" if lum < 140 else "black"
                ax.text(x, y, lab, ha="center", va="center", fontsize=cell_px*0.55, color=tc)

    ax.set_axis_off()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig); buf.seek(0)
    return Image.open(buf)

def export_pdf(idx_grid: np.ndarray,
               palette: np.ndarray,
               labels: np.ndarray,
               show_text=True,
               bold10=True,
               number_every=10,
               mask_area: Optional[np.ndarray]=None) -> bytes:
    H, W = idx_grid.shape
    # high-DPI export
    dpi = 300
    cell_px = 10  # render density
    fig_w = W * cell_px / 100
    fig_h = H * cell_px / 100
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    img_rgb = palette[idx_grid]
    if mask_area is not None:
        img_rgb = img_rgb.copy()
        img_rgb[mask_area == 0] = (255, 255, 255)
    ax.imshow(img_rgb, interpolation="nearest")

    # micro grid
    ax.set_xticks(np.arange(-.5, W, 1), minor=True)
    ax.set_yticks(np.arange(-.5, H, 1), minor=True)
    ax.grid(which="minor", linewidth=0.25, color="0.75")

    # bold every 10
    if bold10:
        ax.set_xticks(np.arange(-.5, W, 10))
        ax.set_yticks(np.arange(-.5, H, 10))
        ax.grid(which="both", linewidth=0.9, color="0.25")

    # numbers
    ax.set_xlim([-0.5, W-0.5]); ax.set_ylim([H-0.5, -0.5])
    ax.set_xticklabels([]); ax.set_yticklabels([])
    ax.tick_params(left=False, bottom=False)
    for x in range(0, W, number_every):
        ax.text(x, -1.2, f"{x}", ha="center", va="top", fontsize=5)
        ax.text(x, H,   f"{x}", ha="center", va="bottom", fontsize=5)
    for y in range(0, H, number_every):
        ax.text(-1.2, y, f"{y}", ha="right", va="center", fontsize=5)
        ax.text(W,   y, f"{y}", ha="left",  va="center", fontsize=5)

    ax.add_patch(plt.Rectangle((-0.5, -0.5), W, H, fill=False, lw=1.2, ec="0.1"))

    # per-cell labels
    if show_text:
        for y in range(H):
            for x in range(W):
                if mask_area is not None and mask_area[y, x] == 0:
                    continue
                lab = labels[y, x]
                if not lab:
                    continue
                r, g, b = palette[idx_grid[y, x]]
                lum = 0.2126*r + 0.7152*g + 0.0722*b
                tc = "white" if lum < 140 else "black"
                ax.text(x, y, lab, ha="center", va="center", fontsize=4.5, color=tc)

    ax.set_axis_off()
    buf = io.BytesIO()
    plt.savefig(buf, format="pdf", bbox_inches="tight", pad_inches=0.25)
    plt.close(fig); buf.seek(0)
    return buf.getvalue()

# Palette as numpy array
palette_np = palette_cols.astype(np.uint8)

# On-screen preview (PNG)
chart_img = render_chart_png(
    idx_grid=idx,
    palette=palette_np,
    labels=cell_labels_masked if label_text else np.full_like(cell_labels_masked, "", dtype=object),
    show_text=label_text,
    cell_px=cell_px,
    bold10=bold10,
    number_every=number_every,
    mask_area=mask
)

st.subheader("3) Pattern Preview")
st.image(chart_img, use_column_width=True)

st.subheader("4) Legend")
st.dataframe(legend, use_container_width=True)

# Downloads
pdf_bytes = export_pdf(
    idx_grid=idx,
    palette=palette_np,
    labels=cell_labels_masked if label_text else np.full_like(cell_labels_masked, "", dtype=object),
    show_text=label_text,
    bold10=bold10,
    number_every=number_every,
    mask_area=mask
)

c1, c2 = st.columns(2)
with c1:
    # PNG export for quick sharing
    out_png = io.BytesIO()
    chart_img.save(out_png, format="PNG")
    st.download_button("⬇️ Download Chart (PNG)", out_png.getvalue(),
                       file_name="crossstitch_chart.png", mime="image/png")
with c2:
    st.download_button("⬇️ Download Pattern (PDF)", pdf_bytes,
                       file_name="crossstitch_pattern.pdf", mime="application/pdf")

st.markdown(
    """
**Notes**
- Each pixel in the resized image equals **one full cross**.
- For **colour** charts, you can use an adaptive palette (uncheck “thread palette”) or map to a **thread set** (DMC subset included here—swap in your full list for production).
- For **greyscale** patterns, we quantise to a fixed number of grey steps (set by *Max colours*).
- **Aida count** + your chosen size controls the stitch grid; the app shows the stitched size in inches.
- The **shape mask** (rectangle / square / circle) is useful for hoop patterns or centred designs.
"""
)