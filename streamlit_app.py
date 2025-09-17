import io
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(page_title="Cross-Stitch Stencil Grid", layout="wide")
st.title("🧵 Cross-Stitch Grid with Stencil (touch friendly)")

# Sidebar
rows = st.sidebar.number_input("Rows", 5, 200, 40)
cols = st.sidebar.number_input("Cols", 5, 200, 60)
cell_size = st.sidebar.slider("Cell size (px)", 10, 30, 20)

st.sidebar.markdown("---")
st.sidebar.subheader("Stencil")
stencil_file = st.sidebar.file_uploader("Upload stencil (PNG/JPG)", type=["png","jpg","jpeg"])
stencil_threshold = st.sidebar.slider("Threshold (darker→black)", 0.0, 1.0, 0.5, 0.01)
stencil_opacity = st.sidebar.slider("Opacity", 0.0, 1.0, 0.3, 0.05)
invert = st.sidebar.checkbox("Invert stencil", value=False)

# Process stencil
stencil_img = None
if stencil_file:
    im = Image.open(stencil_file).convert("L")
    im = im.resize((cols, rows), Image.Resampling.NEAREST)
    arr = np.array(im, dtype=np.float32)/255.0
    if invert:
        arr = 1-arr
    mask = (arr < stencil_threshold).astype(np.uint8)*255
    stencil_img = Image.fromarray(mask).convert("RGBA")
    # Apply opacity
    alpha = int(stencil_opacity*255)
    stencil_img.putalpha(alpha)
    st.image(stencil_img.resize((cols*cell_size, rows*cell_size)), caption="Stencil preview")

# Canvas for drawing
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=cell_size,
    stroke_color="black",
    background_color="white",
    background_image=stencil_img.resize((cols*cell_size, rows*cell_size)) if stencil_img else None,
    update_streamlit=True,
    height=rows*cell_size,
    width=cols*cell_size,
    drawing_mode="freedraw",
    key="canvas",
)

# Convert canvas to grid
if canvas_result.image_data is not None:
    data = np.array(canvas_result.image_data)  # RGBA
    # Treat black pixels as stitches
    gray = data[:,:,:3].mean(axis=2)
    grid = (gray < 128).astype(np.uint8)
else:
    grid = np.zeros((rows, cols), dtype=np.uint8)

# Export PDF
def export_pdf(g: np.ndarray) -> bytes:
    H, W = g.shape
    fig, ax = plt.subplots(figsize=(W/5, H/5), dpi=100)
    ax.imshow(g, cmap="gray_r", interpolation="nearest")
    ax.set_xticks(np.arange(-.5,W,1), minor=True)
    ax.set_yticks(np.arange(-.5,H,1), minor=True)
    ax.grid(which="minor", color="0.8", linewidth=0.3)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_axis_off()
    buf = io.BytesIO()
    plt.savefig(buf, format="pdf", bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

st.download_button("⬇️ Download PDF", data=export_pdf(grid), file_name="crossstitch.pdf", mime="application/pdf")