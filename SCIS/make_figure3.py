import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Circle, Rectangle, FancyArrow
from matplotlib import font_manager as fm
cond = [f.fname for f in fm.fontManager.ttflist if f.name == 'DejaVu Sans' and 'Condensed' in f.fname]
FB = fm.FontProperties(fname=[c for c in cond if 'Bold' in c and 'Oblique' not in c][0])
FR = fm.FontProperties(fname=[c for c in cond if 'Bold' not in c and 'Oblique' not in c][0])
OUT = sys.argv[1] if len(sys.argv) > 1 else '/tmp/figure3_v3.png'

W_IN, H_IN = 5.4, 3.2
fig = plt.figure(figsize=(W_IN, H_IN)); ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W_IN); ax.set_ylim(0, H_IN); ax.axis('off')
HEAD, BODY, BAND, EDGE, DARK, ARR, MID = '#6B6E73', '#F4F5F6', '#D5D7DA', '#A3A6AB', '#1F2937', '#A9B1BC', '#4B5563'
BLUE, PURP, GREEN, SLATE, ORANGE = '#2E75B6', '#7A4EAB', '#2E9E6B', '#4A5568', '#D9772B'
from matplotlib.colors import to_rgb
def tint(c, f):
    r, g, b = to_rgb(c); return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)
TEXTS, ZONES = [], []          # (artist, container rect) ; icon keep-out rects

def rbox(x, y, w, h, fc, ec=EDGE, lw=0.9, r=0.07, z=2, shadow=True):
    if shadow:
        ax.add_patch(FancyBboxPatch((x + 0.025, y - 0.03), w, h, boxstyle=f'round,pad=0,rounding_size={r}', fc='k', ec='none', alpha=0.07, zorder=z - 0.5))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', fc=fc, ec=ec, lw=lw, zorder=z))

def txt(x, y, s, size, box, bold=False, color=DARK, ha='center', va='center', z=6, **kw):
    t = ax.text(x, y, s, fontproperties=FB if bold else FR, fontsize=size, color=color, ha=ha, va=va, zorder=z, **kw)
    TEXTS.append((t, box, s))
    return t

def card(x, y, w, h, title, col=HEAD):
    rbox(x, y, w, h, tint(col, 0.93), ec=tint(col, 0.45))
    ax.add_patch(FancyBboxPatch((x, y + h - 0.3), w, 0.3, boxstyle='round,pad=0,rounding_size=0.07', fc=col, ec=col, lw=0.9, zorder=3))
    ax.add_patch(Rectangle((x, y + h - 0.3), w, 0.1, fc=col, ec='none', zorder=3))
    txt(x + w / 2, y + h - 0.15, title, 9.0, (x, y + h - 0.3, w, 0.3), True, 'white')

def band(x, y, w, h, s, col=HEAD):
    rbox(x, y, w, h, tint(col, 0.72), ec=tint(col, 0.35), z=4)
    txt(x + w / 2, y + h / 2, s, 8.0, (x, y, w, h), True)

def bullets(x, y, items, box, size=7.1, dy=0.17):
    for k, s in enumerate(items):
        txt(x, y - k * dy, '•  ' + s, size, box, ha='left')

def arrow(x0, y0, x1, y1, w=0.07):
    ax.add_patch(FancyArrow(x0, y0, x1 - x0, y1 - y0, width=w, head_width=w * 2.3, head_length=0.1,
                            length_includes_head=True, fc=ARR, ec='#8E97A3', lw=0.4, zorder=1))

# ---------------- icons (each registers a keep-out zone) ----------------
def icon_chart(cx, cy, s=0.17, MID=BLUE):
    ZONES.append((cx - s, cy - s, 2 * s, 2 * s))
    ax.plot([cx - s, cx - s, cx + s], [cy + s, cy - s, cy - s], color=MID, lw=1.0, zorder=6)
    for i, hgt in enumerate([0.35, 0.55, 0.75, 0.95]):
        ax.add_patch(Rectangle((cx - s + s * (0.22 + i * 0.44), cy - s), s * 0.26, hgt * s * 1.5, fc=tint(BLUE, 0.35), ec='none', zorder=6))
    ax.plot([cx - s * 0.8, cx - s * 0.2, cx + s * 0.15, cx + s * 0.9], [cy - s * 0.1, cy + s * 0.45, cy + s * 0.15, cy + s * 0.9], color=MID, lw=1.1, zorder=7)
    ax.add_patch(Polygon([[cx + s, cy + s], [cx + s * 0.62, cy + s * 0.92], [cx + s * 0.92, cy + s * 0.62]], fc=MID, ec=MID, zorder=7))

def icon_route(cx, cy, s=0.17, MID=PURP):
    ZONES.append((cx - s, cy - s, 2 * s, 2 * s))
    ax.add_patch(Polygon([[cx - s, cy - s * 0.8], [cx - s, cy + s * 0.8], [cx - s / 3, cy + s], [cx + s / 3, cy + s * 0.8], [cx + s, cy + s],
                          [cx + s, cy - s * 0.8], [cx + s / 3, cy - s], [cx - s / 3, cy - s * 0.8]], fc=tint(PURP, 0.85), ec=MID, lw=0.8, zorder=6))
    ax.plot([cx - s * 0.65, cx - s * 0.1, cx + s * 0.2, cx + s * 0.6], [cy - s * 0.4, cy + s * 0.3, cy - s * 0.25, cy + s * 0.5], color=MID, lw=0.9, ls=(0, (2, 1.2)), zorder=7)
    for px, py in ((cx - s * 0.65, cy - s * 0.4), (cx + s * 0.2, cy - s * 0.25), (cx + s * 0.6, cy + s * 0.5)):
        ax.add_patch(Circle((px, py), 0.03, fc='white', ec=MID, lw=0.9, zorder=8))

def icon_chip(cx, cy, s=0.21):
    ZONES.append((cx - s - 0.1, cy - s - 0.1, 2 * s + 0.2, 2 * s + 0.2))
    for i in range(7):
        o = -s + 0.05 + i * (2 * s - 0.1) / 6
        for (x0, y0, x1, y1) in ((cx + o, cy + s, cx + o, cy + s + 0.07), (cx + o, cy - s, cx + o, cy - s - 0.07),
                                 (cx - s, cy + o, cx - s - 0.07, cy + o), (cx + s, cy + o, cx + s + 0.07, cy + o)):
            ax.plot([x0, x1], [y0, y1], color=tint(PURP, 0.2), lw=0.9, zorder=5)
            ax.add_patch(Circle((x1, y1), 0.013, fc=tint(PURP, 0.2), ec='none', zorder=5))
    rbox(cx - s, cy - s, 2 * s, 2 * s, tint(PURP, 0.35), ec=PURP, lw=1.0, r=0.05, z=6, shadow=False)
    rbox(cx - s * 0.72, cy - s * 0.72, 1.44 * s, 1.44 * s, '#4B2E7A', ec=tint(PURP, 0.5), lw=1.2, r=0.04, z=7, shadow=False)
    ax.text(cx, cy, 'AI', fontproperties=FB, fontsize=12, color='white', ha='center', va='center', zorder=8)

def icon_cubes(cx, cy, s=0.085):
    ZONES.append((cx - 2 * s, cy - 1.1 * s, 4 * s, 3.6 * s))
    def cube(x, y):
        top = [[x, y + s], [x + s, y + s * 1.5], [x, y + 2 * s], [x - s, y + s * 1.5]]
        left = [[x - s, y + s * 1.5], [x, y + s], [x, y - s * 0.2], [x - s, y + s * 0.3]]
        right = [[x, y + s], [x + s, y + s * 1.5], [x + s, y + s * 0.3], [x, y - s * 0.2]]
        for poly, col in ((top, tint(GREEN, 0.55)), (left, GREEN), (right, '#237A52')):
            ax.add_patch(Polygon(poly, fc=col, ec='#1B5E3F', lw=0.5, zorder=7))
    cube(cx - s, cy - s * 0.9); cube(cx + s, cy - s * 0.9); cube(cx, cy + s * 0.55)

def icon_truck(cx, cy):
    ZONES.append((cx - 0.2, cy - 0.11, 0.39, 0.22))
    ax.add_patch(Rectangle((cx - 0.2, cy - 0.05), 0.24, 0.15, fc=tint(SLATE, 0.3), ec=SLATE, lw=0.7, zorder=6))
    ax.add_patch(Polygon([[cx + 0.04, cy - 0.05], [cx + 0.04, cy + 0.07], [cx + 0.13, cy + 0.07], [cx + 0.18, cy], [cx + 0.18, cy - 0.05]], fc=tint(SLATE, 0.7), ec=SLATE, lw=0.7, zorder=6))
    for px in (cx - 0.13, cx + 0.11):
        ax.add_patch(Circle((px, cy - 0.065), 0.032, fc='#3F4246', ec='white', lw=0.5, zorder=7))

# ---------------- layout grid (inches, origin bottom-left) ----------------
LX, CX, RX = 0.10, 1.96, 3.70          # column x
LW, CW, RW = 1.60, 1.48, 1.60          # column widths
TOP_Y, TOP_H = 1.72, 1.40              # top cards
BOT_Y, BOT_H = 0.10, 1.22              # bottom cards
AI_Y, AI_H = 0.96, 1.16                # centre card

# Demand Model
card(LX, TOP_Y, LW, TOP_H, 'Demand Model', BLUE)
inner = (LX, TOP_Y + 0.25, LW, TOP_H - 0.55)
icon_chart(LX + 0.28, TOP_Y + 0.87)
txt(LX + 0.52, TOP_Y + 0.87, 'Observed Demand', 7.5, inner, True, ha='left')
bullets(LX + 0.14, TOP_Y + 0.58, ['IoT sensor readings', 'EWMA fills missing data'], inner)
band(LX + 0.10, TOP_Y - 0.15, LW - 0.30, 0.30, 'Demand State', BLUE)

# Routing Model
card(RX, TOP_Y, RW, TOP_H, 'Routing Model', PURP)
inner = (RX, TOP_Y + 0.25, RW, TOP_H - 0.55)
icon_route(RX + 0.28, TOP_Y + 0.87)
txt(RX + 0.52, TOP_Y + 0.87, 'Route Planning', 7.5, inner, True, ha='left')
bullets(RX + 0.14, TOP_Y + 0.58, ['Capacity and shift limits', 'One visit per customer'], inner)
band(RX + 0.20, TOP_Y - 0.15, RW - 0.30, 0.30, 'Daily Routing Plan', PURP)

# AI chip + AI Optimization
icon_chip(CX + CW / 2, 2.62)
card(CX, AI_Y, CW, AI_H, 'AI Optimization', PURP)
ax.add_patch(Rectangle((CX + 0.01, AI_Y + AI_H - 0.56), CW - 0.02, 0.26, fc=tint(PURP, 0.8), ec='none', zorder=3))
txt(CX + CW / 2, AI_Y + AI_H - 0.43, 'Reinforcement Learning', 7.9, (CX, AI_Y + AI_H - 0.56, CW, 0.26), True)
inner = (CX, AI_Y, CW, AI_H - 0.56)
bullets(CX + 0.10, AI_Y + 0.46, ['Tabular Q-learning', 'Candidate-scoring DQN', 'Reward: cost, delay, demand'], inner, size=6.6, dy=0.16)

# Blockchain Validation
rbox(RX, BOT_Y, RW, BOT_H, tint(GREEN, 0.93), ec=tint(GREEN, 0.45))
ax.add_patch(Rectangle((RX + 0.005, BOT_Y + BOT_H - 0.07), RW - 0.01, 0.0, fc=GREEN))
bx = (RX, BOT_Y, RW, BOT_H)
icon_cubes(RX + 0.30, BOT_Y + BOT_H - 0.33)
txt(RX + 0.58, BOT_Y + BOT_H - 0.22, 'Blockchain', 8.4, bx, True, color='#1B5E3F', ha='left')
txt(RX + 0.58, BOT_Y + BOT_H - 0.40, 'Validation', 8.4, bx, True, color='#1B5E3F', ha='left')
bullets(RX + 0.12, BOT_Y + 0.50, ['Capacity', 'Unique visits'], (RX, BOT_Y, RW / 2 + 0.02, BOT_H), size=6.9, dy=0.16)
bullets(RX + 0.84, BOT_Y + 0.50, ['Shift length', 'Valid loads'], (RX + RW / 2 - 0.02, BOT_Y, RW / 2 + 0.02, BOT_H), size=6.9, dy=0.16)
txt(RX + RW / 2, BOT_Y + 0.12, '5 validators · majority vote (τ = 0.7)', 6.3, bx)

# Plan Execution
rbox(LX, BOT_Y, LW, BOT_H, tint(SLATE, 0.93), ec=tint(SLATE, 0.45))
ex = (LX, BOT_Y + 0.28, LW, BOT_H - 0.28)
icon_truck(LX + 0.28, BOT_Y + BOT_H - 0.26)
txt(LX + 0.55, BOT_Y + BOT_H - 0.26, 'Plan Execution', 8.0, ex, True, ha='left')
bullets(LX + 0.14, BOT_Y + 0.66, ['Accepted: executed', 'Rejected: reactive fallback'], ex, size=6.9, dy=0.17)
ax.add_patch(Rectangle((LX + 0.005, BOT_Y + 0.005), LW - 0.01, 0.26, fc=tint(SLATE, 0.72), ec='none', zorder=3))
ax.plot([LX, LX + LW], [BOT_Y + 0.265, BOT_Y + 0.265], color=tint(SLATE, 0.45), lw=0.7, zorder=3)
txt(LX + LW / 2, BOT_Y + 0.135, 'Executed Routes', 8.0, (LX, BOT_Y, LW, 0.26), True)

# ---------------- arrows and flow labels ----------------
ym = TOP_Y
arrow(LX + LW - 0.20, ym, CX, ym)                          # Demand State -> AI Optimization
arrow(CX + CW, ym, RX + 0.20, ym)                          # AI Optimization -> Daily Routing Plan
arrow(RX + RW / 2 + 0.05, TOP_Y - 0.15, RX + RW / 2 + 0.05, BOT_Y + BOT_H)   # plan -> validation
yv = 0.42
arrow(RX, yv, LX + LW, yv)                                 # validation -> execution
gap = (LX + LW, 0.10, RX - LX - LW, 0.84)
txt((LX + LW + RX) / 2, yv + 0.13, 'validated plan', 6.8, gap, color=MID)
# training feedback: execution -> AI Optimization (dashed elbow below the AI card)
fy = 0.78
ax.plot([LX + LW, CX + 0.32], [fy, fy], color=ORANGE, lw=1.0, ls=(0, (3, 1.6)), zorder=4)
ax.annotate('', xy=(CX + 0.32, AI_Y), xytext=(CX + 0.32, fy), arrowprops=dict(arrowstyle='-|>', color=ORANGE, lw=1.0, mutation_scale=8), zorder=4)
txt(CX + 0.40, fy, 'training feedback', 6.6, gap, color='#B35E1C', ha='left')

# ---------------- overlap / containment check ----------------
fig.canvas.draw()
rend = fig.canvas.get_renderer()
def inches(t):
    b = t.get_window_extent(rend); d = fig.dpi
    return (b.x0 / d, b.y0 / d, b.x1 / d, b.y1 / d)
M = 0.03
problems = []
boxes = [(inches(t), box, s) for t, box, s in TEXTS]
for (x0, y0, x1, y1), (bx0, by0, bw, bh), s in boxes:
    if x0 < bx0 + M or x1 > bx0 + bw - M or y0 < by0 + 0.005 or y1 > by0 + bh - 0.005:
        problems.append(f'outside container: {s!r}  text=({x0:.2f},{y0:.2f},{x1:.2f},{y1:.2f}) box=({bx0:.2f},{by0:.2f},{bx0+bw:.2f},{by0+bh:.2f})')
for i in range(len(boxes)):
    for j in range(i + 1, len(boxes)):
        a, b = boxes[i][0], boxes[j][0]
        if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
            problems.append(f'text overlap: {boxes[i][2]!r} / {boxes[j][2]!r}')
    for zx, zy, zw, zh in ZONES:
        a = boxes[i][0]
        if a[0] < zx + zw and zx < a[2] and a[1] < zy + zh and zy < a[3]:
            problems.append(f'text over icon: {boxes[i][2]!r}')
print('PROBLEMS:', len(problems))
for p in problems: print('  ', p)
fig.savefig(OUT, dpi=600, facecolor='white')
