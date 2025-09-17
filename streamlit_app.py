import io
from dataclasses import dataclass
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(page_title="Cross-Stitch Grid + Stencil", layout="wide")
st.title("🧵 Clickable Cross-Stitch Grid + Stencil")
st.caption("Click squares to toggle stitches. Optional translucent stencil overlay. Export to PDF.")

# ---------------- Session state ----------------
@dataclass
class RectBuffer:
    start: tuple | None = None  # (x, y)

def init_state(h=40, w=60):
    st.session_state.grid = np.zeros((h, w), dtype=np.uint8)
    st.session_state.history = []
    st.session_state.rect_buf = RectBuffer()

if "grid" not in st.session_state:
    init_state()

grid = st.session_state.grid
H, W = grid.shape

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Grid")
    c1, c2 = st.columns(2)
    with c1:
        new_h = st.number_input("Rows", min_value=5, max_value=200, value=H, step=1)
    with c2:
        new_w = st.number_input("Cols", min_value=5, max_value=200, value=W, step=1)
    if (new_h, new_w) != (H, W):
        # keep a snapshot for undo and re-init grid with new size
        st.session_state.history.append(grid.copy())
        init_state(new_h, new_w)
        grid = st.session_state.grid
        H, W = grid.shape

    st.subheader("Tools")
    tool = st.radio("Tool", ["Paint", "Erase", "Rectangle Fill"], index=0, horizontal=True)
    multi_rect = st.checkbox("Multi-rectangle mode", value=False)

    st.subheader("Grid look")
    show_10_grid = st.checkbox("Bold every 10 stitches", value=True)
    number_every = st.selectbox("Numbering interval", [5, 10], index=1)

    st.subheader("Stencil (optional)")
    stencil_file = st.file_uploader("Image (PNG/JPG)", type=["png", "jpg", "jpeg"])
    stencil_opacity = st.slider("Stencil opacity", 0.0, 1.0, 0.35, 0.01)
    stencil_threshold = st.slider("Threshold (darker → black)", 0.0, 1.0, 0.55, 0.01)
    stencil_invert = st.checkbox("Invert stencil", value=False)
    include_stencil_pdf = st.checkbox("Include stencil in PDF", value=False)

    st.subheader("Actions")
    a1, a2, a3 = st.columns(3)
    with a1:
        if st.button("Undo"):
            if st.session_state.history:
                st.session_state.grid = st.session_state.history.pop()
                grid = st.session_state.grid
                H, W = grid.shape
    with a2:
        if st.button("Clear"):
            st.session_state.history.append(grid.copy())
            st.session_state.grid[:] = 0
    with a3:
        if st.button("Invert Grid"):
            st.session_state.history.append(grid.copy())
            st.session_state.grid[:] = 1 - grid

# ---------------- Stencil processing ----------------
def load_stencil_mask(file, target_h, target_w, thr: float, invert: bool):
    if not file:
        return None
    im = Image.open(file).convert("L")  # grayscale
    im = im.resize((target_w, target_h), Image.Resampling.LANCZOS)
    arr = np.asarray(im, dtype=np.float32) / 255.0  # 0=black, 1=white
    if invert:
        arr = 1.0 - arr
    mask = (arr < thr).astype(np.float32)  # 1=black, 0=white
    return mask

stencil = load_stencil_mask(stencil_file, H, W, stencil_threshold, stencil_invert)

# ---------------- Figure ----------------
def make_fig(g: np.ndarray, stencil_mask: np.ndarray | None):
    layers = []
    if stencil_mask is not None:
        # Draw stencil UNDER the grid (so clicks reach the grid trace)
        layers.append(
            go.Heatmap(
                z=stencil_mask[::-1, :],
                showscale=False,
                colorscale=[[0.0, "rgba(0,0,0,0)"], [1.0, "rgba(0,0,0,1)"]],
                opacity=stencil_opacity,
                x=np.arange(W), y=np.arange(H),
                hoverinfo="skip"
            )
        )
    layers.append(
        go.Heatmap(
            z=g[::-1, :],
            showscale=False,
            colorscale=[[0.0, "#FFFFFF"], [1.0, "#000000"]],
            x=np.arange(W), y=np.arange(H),
            hovertemplate="x=%{x}, y=%{y}<extra></extra>",
        )
    )
    fig = go.Figure(data=layers)
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        clickmode="event+select",     # IMPORTANT for click events
        dragmode=False,
        uirevision="keep",            # keep zoom on rerender
        plot_bgcolor="white",
        height=min(800, 14 * H),
        xaxis=dict(constrain="domain", tickmode="linear", dtick=1, showgrid=False),
        yaxis=dict(constrain="domain", tickmode="linear", dtick=1, showgrid=False, scaleanchor="x", scaleratio=1),
    )
    # Bold 10×10
    shapes = []
    step = 10 if show_10_grid else 10_000
    for x in range(0, W + 1, step):
        shapes.append(dict(type="line", x0=x-0.5, x1=x-0.5, y0=-0.5, y1=H-0.5, line=dict(width=2, color="rgba(0,0,0,0.35)")))
    for y in range(0, H + 1, step):
        shapes.append(dict(type="line", x0=-0.5, x1=W-0.5, y0=y-0.5, y1=y-0.5, line=dict(width=2, color="rgba(0,0,0,0.35)")))
    shapes.append(dict(type="rect", x0=-0.5, y0=-0.5, x1=W-0.5, y1=H-0.5, line=dict(width=2.5, color="rgba(0,0,0,0.6)"),
                       fillcolor="rgba(0,0,0,0)"))
    fig.update_layout(shapes=shapes)
    # Edge numbers
    ann = []
    for x in range(0, W, number_every):
        ann.append(dict(x=x, y=-1.1, text=str(x), showarrow=False, font=dict(size=10)))
        ann.append(dict(x=x, y=H,   text=str(x), showarrow=False, font=dict(size=10)))
    for y in range(0, H, number_every):
        ann.append(dict(x=-1.1, y=y, text=str(y), showarrow=False, font=dict(size=10)))
        ann.append(dict(x=W,   y=y, text=str(y), showarrow=False, font=dict(size=10)))
    fig.update_layout(annotations=ann)
    return fig

st.subheader("Editor")
fig = make_fig(grid, stencil)

# Use a unique key so Streamlit knows this interactive widget instance
events = plotly_events(
    fig,
    click_event=True,
    select_event=False,
    hover_event=False,
    override_height=None,
    override_width=None,
    key=f"grid_{H}x{W}",  # rerender when size changes
)

# ---------------- Edit functions ----------------
def apply_point(px, py):
    gy = H - 1 - int(py)  # plotly y is bottom-up on our heatmap
    gx = int(px)
    if 0 <= gx < W and 0 <= gy < H:
        before = st.session_state.grid.copy()
        if tool == "Paint":
            st.session_state.grid[gy, gx] = 1
        elif tool == "Erase":
            st.session_state.grid[gy, gx] = 0
        elif tool == "Rectangle Fill":
            rb = st.session_state.rect_buf
            if rb.start is None:
                rb.start = (gx, gy)
                return False  # wait for second point
            else:
                x0, y0 = rb.start
                x1, y1 = gx, gy
                xlo, xhi = sorted([x0, x1]); ylo, yhi = sorted([y0, y1])
                st.session_state.grid[ylo:yhi+1, xlo:xhi+1] = 1
                st.session_state.rect_buf.start = None
        if not np.array_equal(before, st.session_state.grid):
            st.session_state.history.append(before)
        return True
    return False

# Apply clicks
if events:
    for e in events:
        x = int(e["x"]); y = int(e["y"])
        finished = apply_point(x, y)
        if tool == "Rectangle Fill" and finished and not multi_rect:
            break

st.write("Tip: In **Rectangle Fill**, click once to start and once to finish. Enable *Multi-rectangle mode* to draw several without refresh.")

# ---------------- PDF export ----------------
def export_pdf(g: np.ndarray, stencil_mask: np.ndarray | None,
               bold_every=10, number_every=10, include_stencil=False) -> bytes:
    H, W = g.shape
    cell_px = 10
    fig_w = W * cell_px / 100
    fig_h = H * cell_px / 100
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)

    if include_stencil and stencil_mask is not None:
        ax.imshow(stencil_mask, cmap="gray_r", interpolation="nearest", alpha=0.28)

    ax.imshow(g, cmap="gray_r", interpolation="nearest")

    ax.set_xticks(np.arange(-.5, W, 1), minor=True)
    ax.set_yticks(np.arange(-.5, H, 1), minor=True)
    ax.grid(which="minor", linewidth=0.25, color="0.75")

    if bold_every:
        ax.set_xticks(np.arange(-.5, W, bold_every))
        ax.set_yticks(np.arange(-.5, H, bold_every))
        ax.grid(which="both", linewidth=0.9, color="0.25")

    ax.set_xlim([-0.5, W-0.5]); ax.set_ylim([H-0.5, -0.5])
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.tick_params(left=False, bottom=False)
    for x in range(0, W, number_every):
        ax.text(x, -1.2, f"{x}", ha="center", va="top", fontsize=5)
        ax.text(x, H,   f"{x}", ha="center", va="bottom", fontsize=5)
    for y in range(0, H, number_every):
        ax.text(-1.2, y, f"{y}", ha="right", va="center", fontsize=5)
        ax.text(W,   y, f"{y}", ha="left",  va="center", fontsize=5)

    ax.add_patch(plt.Rectangle((-0.5, -0.5), W, H, fill=False, lw=1.2, ec="0.1"))
    ax.set_axis_off()
    buf = io.BytesIO()
    plt.savefig(buf, format="pdf", bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

st.subheader("Export")
pdf_bytes = export_pdf(
    st.session_state.grid,
    stencil,
    bold_every=10 if show_10_grid else 0,
    number_every=number_every,
    include_stencil=include_stencil_pdf
)
st.download_button("⬇️ Download PDF", data=pdf_bytes, file_name="cross_stitch_grid.pdf", mime="application/pdf")