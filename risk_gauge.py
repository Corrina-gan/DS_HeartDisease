"""Total Risk Level gauge for the live prediction result."""

import math

import streamlit as st


def _polar(cx, cy, deg, r):
    rad = math.radians(deg)
    return cx + r * math.cos(rad), cy - r * math.sin(rad)


def _arc(cx, cy, start_deg, end_deg, r):
    """Clockwise upper arc (decreasing degrees: left → top → right)."""
    x0, y0 = _polar(cx, cy, start_deg, r)
    x1, y1 = _polar(cx, cy, end_deg, r)
    return f"M {x0:.2f} {y0:.2f} A {r:.2f} {r:.2f} 0 0 1 {x1:.2f} {y1:.2f}"


def _needle_poly(cx, cy, deg, inner, outer, half_w=8.5):
    """Filled arrow from hub toward the arc, pointing at `deg`."""
    rad = math.radians(deg)
    ux, uy = math.cos(rad), -math.sin(rad)
    px, py = -uy, ux
    tx, ty = _polar(cx, cy, deg, outer)
    bx, by = _polar(cx, cy, deg, inner)
    lx, ly = bx + px * half_w, by + py * half_w
    rx, ry = bx - px * half_w, by - py * half_w
    # slightly flared tail so the pointer reads as an arrow, not a line
    return (
        f"M {tx:.2f} {ty:.2f} L {lx:.2f} {ly:.2f} "
        f"L {bx - ux * 10:.2f} {by - uy * 10:.2f} L {rx:.2f} {ry:.2f} Z"
    )


def _build_gauge_svg(risk_pct):
    pct = max(0.0, min(100.0, float(risk_pct)))
    shown = int(round(pct))
    pointer = "#0f766e"

    if pct < 30:
        tip = "#0f766e"
    elif pct >= 70:
        tip = "#be123c"
    else:
        tip = "#ca8a04"

    cx, cy, radius, stroke = 220, 168, 118, 30
    needle = 180.0 - pct * 1.8
    lx, ly = _polar(cx, cy, needle, radius + 36)
    arc_d = _arc(cx, cy, 180, 90, radius) + " " + _arc(cx, cy, 90, 0, radius)
    arrow = _needle_poly(cx, cy, needle, 26, radius - 2, 9)

    return f"""
<svg class="rg-svg" viewBox="0 0 440 230" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="rgFill" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#22c55e"/>
      <stop offset="22%" stop-color="#84cc16"/>
      <stop offset="42%" stop-color="#eab308"/>
      <stop offset="62%" stop-color="#f97316"/>
      <stop offset="82%" stop-color="#e11d48"/>
      <stop offset="100%" stop-color="#be123c"/>
    </linearGradient>
  </defs>
  <path d="{arc_d}" fill="none" stroke="#e8eef0" stroke-width="{stroke + 8}" stroke-linecap="round"/>
  <path d="{arc_d}" fill="none" stroke="url(#rgFill)" stroke-width="{stroke}" stroke-linecap="round"/>
  <path d="{arrow}" fill="{pointer}" stroke="#ffffff" stroke-width="1.6" stroke-linejoin="round"/>
  <circle cx="{cx}" cy="{cy}" r="11" fill="{pointer}" stroke="#ffffff" stroke-width="2.4"/>
  <circle cx="{cx}" cy="{cy}" r="4.2" fill="#ffffff"/>
  <text x="{lx:.2f}" y="{ly:.2f}" text-anchor="middle" dominant-baseline="middle"
        fill="{tip}" font-size="14" font-weight="800">{shown}%</text>
  <text x="{cx}" y="{cy - 22}" text-anchor="middle" fill="#16232b"
        font-size="46" font-weight="800">{shown}%</text>
</svg>
"""


_GAUGE_CSS = """
:host {
  display: flex;
  justify-content: center;
  align-items: flex-start;
  width: 100%;
  height: 100%;
}
:host > div {
  width: 100%;
  flex: 0 0 auto;
}
.rg-wrap {
  width: 100%;
  box-sizing: border-box;
  background: #ffffff;
  border-radius: 22px;
  padding: 16px 22px 18px;
  border: 1px solid #d5e3e0;
  box-shadow: 0 12px 32px rgba(22, 35, 43, 0.08);
  text-align: center;
  font-family: Inter, system-ui, sans-serif;
}
.rg-title {
  color: #16232b;
  font-size: 1rem;
  font-weight: 700;
  letter-spacing: 0.03em;
  margin-bottom: 2px;
}
.rg-status {
  display: none;
  margin: 6px auto 4px;
  padding: 3px 10px;
  border-radius: 999px;
  font-size: 0.72rem;
  font-weight: 800;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  width: fit-content;
}
.rg-status.on { display: inline-block; }
.rg-status.ok { background: #e5f6f1; color: #0f766e; border: 1px solid #9fd6c9; }
.rg-status.bad { background: #fde8ea; color: #be123c; border: 1px solid #f3b4b8; }
.rg-svg { width: min(520px, 100%); height: auto; display: block; margin: 0 auto 8px; }
.rg-bar {
  width: min(360px, 72%);
  margin: 6px auto 0;
}
.rg-bar-row {
  position: relative;
  width: 100%;
  height: 14px;
}
.rg-bar-track {
  width: 100%;
  height: 14px;
  border-radius: 999px;
  background: linear-gradient(90deg, #22c55e 0%, #84cc16 22%, #eab308 42%, #f97316 62%, #e11d48 82%, #be123c 100%);
}
.rg-bar-knob {
  position: absolute;
  top: 50%;
  width: 16px;
  height: 16px;
  margin-left: -8px;
  border-radius: 50%;
  background: #ffffff;
  border: 2.5px solid #0f766e;
  box-shadow: 0 2px 6px rgba(22, 35, 43, 0.2);
  transform: translateY(-50%);
  pointer-events: none;
}
.rg-bar-pct {
  position: absolute;
  top: -20px;
  transform: translateX(-50%);
  font-size: 13px;
  font-weight: 800;
  color: #15803d;
  pointer-events: none;
  white-space: nowrap;
}
.rg-bar-labels {
  display: flex;
  justify-content: space-between;
  margin-top: 8px;
  font-size: 13px;
  font-weight: 800;
}
.rg-bar-labels .low { color: #15803d; }
.rg-bar-labels .high { color: #be123c; }
"""

_GAUGE_HTML = """
<div class="rg-wrap">
  <div class="rg-title">Total Risk Level</div>
  <div class="rg-status" id="rg-status"></div>
  <div id="rg-mount"></div>
  <div class="rg-bar">
    <div class="rg-bar-row">
      <div class="rg-bar-pct" id="rg-bar-pct">0%</div>
      <div class="rg-bar-track"></div>
      <div class="rg-bar-knob" id="rg-bar-knob"></div>
    </div>
    <div class="rg-bar-labels"><span class="low">LOW</span><span class="high">HIGH</span></div>
  </div>
</div>
"""

_GAUGE_JS = """
export default function (component) {
  const { data, parentElement } = component;
  const mount = parentElement.querySelector("#rg-mount");
  const status = parentElement.querySelector("#rg-status");
  const knob = parentElement.querySelector("#rg-bar-knob");
  const pctEl = parentElement.querySelector("#rg-bar-pct");
  if (!mount) return;

  if (data && data.svg) {
    mount.innerHTML = data.svg;
  }

  const raw = data && data.risk_pct;
  const pct = Math.max(0, Math.min(100, Number(raw)));
  const shown = Number.isFinite(pct) ? Math.round(pct) : 0;
  const left = Number.isFinite(pct) ? pct : 0;
  if (knob) knob.style.left = left + "%";
  if (pctEl) {
    pctEl.textContent = shown + "%";
    pctEl.style.left = left + "%";
    pctEl.style.color = shown >= 70 ? "#be123c" : shown >= 30 ? "#ca8a04" : "#15803d";
  }

  const label = data && data.label ? String(data.label) : "";
  if (status) {
    status.className = "rg-status";
    if (label === "Yes") {
      status.textContent = "Heart · Bad";
      status.classList.add("on", "bad");
    } else if (label === "No") {
      status.textContent = "Heart · No";
      status.classList.add("on", "ok");
    }
  }
}
"""

_gauge_component = st.components.v2.component(
    "total_risk_gauge_v10",
    html=_GAUGE_HTML,
    css=_GAUGE_CSS,
    js=_GAUGE_JS,
)


def render_total_risk_gauge(risk_pct, pred_label=None, key="total_risk_gauge"):
    """Mount the Total Risk Level gauge. `pred_label` is Yes/No."""
    return _gauge_component(
        data={
            "risk_pct": float(risk_pct),
            "label": pred_label,
            "svg": _build_gauge_svg(risk_pct),
        },
        key=key,
        width="stretch",
        height=400,
    )
