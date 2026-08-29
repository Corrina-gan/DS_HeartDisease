"""Interactive 3D anatomical heart (Three.js) and Plotly 3D helpers."""

import os

import plotly.graph_objects as go
import streamlit as st

_HEART_GLB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "human_heart.glb")
_HEART_GLB_BYTES = None


def _heart_model_url():
    """Serve the GLB through Streamlit's media endpoint (same-origin, works in the browser)."""
    global _HEART_GLB_BYTES
    try:
        if _HEART_GLB_BYTES is None:
            with open(_HEART_GLB_PATH, "rb") as f:
                _HEART_GLB_BYTES = f.read()
        from streamlit.runtime import get_instance

        return get_instance().media_file_mgr.add(
            _HEART_GLB_BYTES,
            "model/gltf-binary",
            "heart_3d.human_heart.glb",
            file_name="human_heart.glb",
        )
    except Exception:
        return "/app/static/human_heart.glb"

_HEART_CSS = """
:host { display: block; width: 100%; height: 100%; }
.stage {
  --glow: rgba(10, 92, 86, 0.18);
  position: relative;
  width: 100%;
  height: 100%;
  min-height: 440px;
  border-radius: 22px;
  overflow: hidden;
  background:
    linear-gradient(180deg, rgba(255,255,255,0.22) 0%, rgba(232,244,242,0.12) 100%);
  border: 1px solid rgba(255, 255, 255, 0.55);
  box-shadow:
    inset 0 1px 0 rgba(255,255,255,0.55),
    0 0 0 1px rgba(10, 92, 86, 0.08),
    0 16px 36px rgba(22, 35, 43, 0.10),
    0 0 28px var(--glow);
  backdrop-filter: blur(16px);
}
.stage.yes { --glow: rgba(225, 29, 72, 0.38); }
.stage.no { --glow: rgba(10, 92, 86, 0.36); }
canvas { display: block; width: 100%; height: 100%; cursor: grab; }
canvas:active { cursor: grabbing; }
.hint {
  position: absolute;
  left: 12px;
  bottom: 56px;
  color: #64748b;
  font: 600 10px/1.4 Inter, system-ui, sans-serif;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  pointer-events: none;
}
.load {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #64748b;
  font: 600 0.92rem/1.4 Inter, system-ui, sans-serif;
  background: rgba(255, 255, 255, 0.55);
  pointer-events: none;
}
.load.hidden { display: none; }
.pin {
  position: absolute;
  width: 16px;
  height: 16px;
  margin: -8px 0 0 -8px;
  border-radius: 50%;
  border: 2px solid #ffffff;
  background: #0a5c56;
  box-shadow: 0 0 0 4px rgba(10, 92, 86, 0.22), 0 6px 14px rgba(0,0,0,0.28);
  cursor: pointer;
  z-index: 3;
  padding: 0;
}
.stage.yes .pin { background: #e11d48; box-shadow: 0 0 0 4px rgba(225, 29, 72, 0.25), 0 6px 14px rgba(0,0,0,0.28); }
.pin:hover, .pin.active { transform: scale(1.2); }
.card {
  position: absolute;
  top: 12px;
  left: 12px;
  width: min(250px, 72%);
  padding: 0.7rem 0.8rem 0.75rem;
  border-radius: 14px;
  background: rgba(255,255,255,0.88);
  border: 1px solid rgba(255,255,255,0.7);
  box-shadow: 0 12px 28px rgba(15, 23, 32, 0.16);
  backdrop-filter: blur(14px);
  z-index: 4;
  color: #16232b;
  font: 500 0.82rem/1.45 Inter, system-ui, sans-serif;
}
.card.hidden { display: none; }
.card strong { display: block; font-size: 0.86rem; margin-bottom: 0.25rem; }
.card p { margin: 0; color: #475569; }
.card button {
  margin-top: 0.5rem;
  border: 0;
  background: transparent;
  color: #0a5c56;
  font: 700 0.75rem/1 Inter, system-ui, sans-serif;
  cursor: pointer;
  padding: 0;
}
.bar {
  position: absolute;
  left: 10px;
  right: 10px;
  bottom: 10px;
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  z-index: 4;
}
.bar button {
  flex: 1 1 auto;
  min-width: 72px;
  border: 1px solid #d5e3e0;
  background: rgba(255, 255, 255, 0.88);
  color: #334155;
  border-radius: 10px;
  padding: 0.38rem 0.45rem;
  font: 700 0.68rem/1.1 Inter, system-ui, sans-serif;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  cursor: pointer;
}
.bar button:hover, .bar button.on {
  background: #0a5c56;
  color: #ffffff;
  border-color: #0a5c56;
}
.stage.yes .bar button.on { background: rgba(190, 18, 60, 0.9); }
"""

_HEART_HTML = """
<div class="stage" id="heart-stage">
  <canvas id="heart-canvas"></canvas>
  <div class="load" id="heart-load">Loading anatomical heart…</div>
  <div class="hint" id="heart-hint">Drag to orbit · scroll to zoom</div>
  <div id="heart-pins"></div>
  <aside class="card hidden" id="heart-card">
    <strong id="heart-card-title"></strong>
    <p id="heart-card-body"></p>
    <button type="button" id="heart-card-close">Close</button>
  </aside>
  <div class="bar" id="heart-bar">
    <button type="button" data-cam="anterior">Anterior</button>
    <button type="button" data-cam="posterior">Posterior</button>
    <button type="button" data-cam="side">Side</button>
    <button type="button" data-cam="reset">Reset</button>
    <button type="button" data-action="clip">Cross-section</button>
    <button type="button" data-action="explode">Layered view</button>
  </div>
</div>
"""

_HEART_JS = r"""
export default function (component) {
  const { data, parentElement } = component;
  const canvas = parentElement.querySelector("#heart-canvas");
  const stage = parentElement.querySelector("#heart-stage");
  const hint = parentElement.querySelector("#heart-hint");
  const loadEl = parentElement.querySelector("#heart-load");
  const pinsEl = parentElement.querySelector("#heart-pins");
  const cardEl = parentElement.querySelector("#heart-card");
  const cardTitle = parentElement.querySelector("#heart-card-title");
  const cardBody = parentElement.querySelector("#heart-card-body");
  const barEl = parentElement.querySelector("#heart-bar");
  if (!canvas) return;

  parentElement.__heartLatest = data;
  if (typeof parentElement.__heartUpdate === "function") {
    parentElement.__heartUpdate(parentElement.__heartLatest);
    return;
  }
  if (parentElement.__heartBooting) return;
  parentElement.__heartBooting = true;

  function applyStatus(next) {
    const label = next && next.label ? String(next.label) : "";
    stage.classList.toggle("yes", label === "Yes");
    stage.classList.toggle("no", label === "No");
    if (label === "Yes") return "yes";
    if (label === "No") return "no";
    return "idle";
  }

  const HOTSPOTS = [
    { id: "lv", name: "Left ventricle", x: -0.42, y: -0.18, z: 0.28, text: "Main pumping chamber. Blood pressure and BMI load this wall the most." },
    { id: "aorta", name: "Aorta", x: 0.02, y: 0.92, z: -0.12, text: "Largest artery leaving the heart. High pressure stresses this vessel first." },
    { id: "coronary", name: "Coronary arteries", x: 0.22, y: 0.18, z: 0.52, text: "Feed the heart muscle. Cholesterol and LDL sit on this pathway." },
    { id: "ra", name: "Right atrium", x: 0.48, y: 0.48, z: 0.18, text: "Receives returning blood. Sleep and stress change how hard the heart works." },
  ];

  const THREE_URL = "https://cdn.jsdelivr.net/npm/three@0.160.1/+esm";
  const LOADER_URL = "https://cdn.jsdelivr.net/npm/three@0.160.1/examples/jsm/loaders/GLTFLoader.js/+esm";

  Promise.all([import(THREE_URL), import(LOADER_URL)])
    .then(([THREE, loaderMod]) => start(THREE, loaderMod.GLTFLoader))
    .catch(() => import("https://cdn.jsdelivr.net/npm/three@0.160.1/build/three.module.js")
      .then((THREE) => start(THREE, null))
      .catch(() => {
        if (hint) hint.textContent = "Could not load the 3D engine (network blocked)";
        if (loadEl) loadEl.textContent = "3D engine blocked";
      }));

  function start(THREE, GLTFLoader) {
    const renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: false,
      preserveDrawingBuffer: true,
      powerPreference: "high-performance",
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.NoToneMapping;
    renderer.setClearColor(0xf4f8f8, 1);
    renderer.toneMappingExposure = 0.88;
    renderer.localClippingEnabled = true;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(40, 1, 0.05, 80);
    camera.position.set(0.1, 0.05, 3.15);

    scene.add(new THREE.HemisphereLight(0xfff6f0, 0xc5d4d0, 0.88));
    const key = new THREE.DirectionalLight(0xffffff, 1.28);
    key.position.set(2.8, 3.6, 4.2);
    scene.add(key);
    const fill = new THREE.DirectionalLight(0xc5ddd8, 0.38);
    fill.position.set(-3.6, 0.4, 1.6);
    scene.add(fill);
    const rim = new THREE.DirectionalLight(0xffc9d0, 0.42);
    rim.position.set(-0.4, -1.2, -3.4);
    scene.add(rim);

    const grid = new THREE.GridHelper(12, 28, 0x9fd6c9, 0xd5e3e0);
    grid.position.y = -1.9;
    scene.add(grid);
    const pCount = 160;
    const pPos = new Float32Array(pCount * 3);
    for (let i = 0; i < pCount; i += 1) {
      pPos[i * 3] = (Math.random() - 0.5) * 9;
      pPos[i * 3 + 1] = (Math.random() - 0.5) * 6;
      pPos[i * 3 + 2] = (Math.random() - 0.5) * 9;
    }
    const pGeo = new THREE.BufferGeometry();
    pGeo.setAttribute("position", new THREE.BufferAttribute(pPos, 3));
    scene.add(new THREE.Points(pGeo, new THREE.PointsMaterial({
      color: 0x0a5c56, size: 0.022, transparent: true, opacity: 0.22, depthWrite: false,
    })));

    const clipPlane = new THREE.Plane(new THREE.Vector3(1, 0.08, 0.04), 0.02);
    const heart = new THREE.Group();
    scene.add(heart);
    const tintMats = [];
    const explodeMeshes = [];

    function kindFromName(label) {
      if (/vein|vena|cava/.test(label)) return "vein";
      if (/arter|aort|carotid|subclavian|brachio|coronar/.test(label)) return "artery";
      if (/valve/.test(label)) return "valve";
      if (/atrium|auricle/.test(label)) return "atrium";
      return "muscle";
    }

    function prepareMats(root) {
      root.traverse((o) => {
        if (!o.isMesh) return;
        o.frustumCulled = false;
        const label = ((o.name || "") + " " + ((o.parent && o.parent.name) || "")).toLowerCase();
        const kind = kindFromName(label);
        const hex = { vein: 0x2e6aa6, artery: 0xd21f3c, valve: 0xc96b6b, atrium: 0xb03d3d, muscle: 0xa83232 }[kind];
        const std = new THREE.MeshLambertMaterial({
          color: hex,
          emissive: hex,
          emissiveIntensity: 0.28,
          side: THREE.DoubleSide,
          clippingPlanes: [],
        });
        std.userData.base = std.color.clone();
        std.userData.kind = kind;
        tintMats.push(std);
        o.material = std;
        o.userData.origin = o.position.clone();
        explodeMeshes.push(o);
      });
    }

    function heat(status, riskPct, pulse) {
      const t = (riskPct == null || Number.isNaN(riskPct)) ? -1 : Math.max(0, Math.min(1, Number(riskPct) / 100));
      tintMats.forEach((m) => {
        if (!m.color || !m.userData.base) return;
        const kind = m.userData.kind || "muscle";
        if (t < 0) {
          m.color.copy(m.userData.base);
          if (m.emissive) {
            m.emissive.copy(m.userData.base);
            m.emissiveIntensity = 0.26;
          }
          return;
        }
        if (t < 0.4) {
          m.color.copy(m.userData.base).lerp(new THREE.Color(0x1f8a72), 0.22);
          if (m.emissive) {
            m.emissive.setHex(kind === "vein" ? 0x1d4ed8 : 0x0a5c56);
            m.emissiveIntensity = 0.18 + pulse * 0.08;
          }
          return;
        }
        const dark = 1 - t * 0.38;
        m.color.copy(m.userData.base).multiplyScalar(dark);
        if (m.emissive) {
          if (kind === "artery" || kind === "muscle") {
            m.emissive.setHex(0x9f1239);
            m.emissiveIntensity = 0.22 + pulse * (0.18 + t * 0.2);
          } else {
            m.emissive.setHex(0x5a080c);
            m.emissiveIntensity = 0.16 + pulse * 0.08;
          }
        }
      });
    }

    function setExploded(on) {
      explodeMeshes.forEach((o) => {
        if (!o.userData.origin) return;
        if (!o.userData.explodeDir) {
          const dir = o.position.clone();
          if (dir.lengthSq() < 1e-6) dir.set(0, 1, 0);
          o.userData.explodeDir = dir.normalize();
        }
        o.position.copy(o.userData.origin);
        if (on) o.position.addScaledVector(o.userData.explodeDir, 0.42);
      });
    }

    function setClipped(on) {
      tintMats.forEach((m) => {
        m.clippingPlanes = on ? [clipPlane] : [];
        m.needsUpdate = true;
      });
    }

    function mountFitted(root) {
      tintMats.length = 0;
      explodeMeshes.length = 0;
      prepareMats(root);
      while (heart.children.length) heart.remove(heart.children[0]);
      root.updateMatrixWorld(true);
      const box = new THREE.Box3().setFromObject(root);
      const size = box.getSize(new THREE.Vector3());
      const center = box.getCenter(new THREE.Vector3());
      const holder = new THREE.Group();
      holder.position.copy(center).multiplyScalar(-1);
      holder.add(root);
      const inner = new THREE.Group();
      inner.add(holder);
      const maxDim = Math.max(size.x, size.y, size.z, 0.001);
      inner.scale.setScalar(4.7 / maxDim);
      inner.rotation.set(0.42, 0.55, 0.06);
      heart.add(inner);
      HOTSPOTS.forEach((h) => {
        const marker = new THREE.Object3D();
        marker.position.set(h.x, h.y, h.z);
        marker.userData.hotspot = h;
        heart.add(marker);
        h.obj = marker;
      });
      return { size, maxDim };
    }

    function buildFallback() {
      const muscle = new THREE.MeshStandardMaterial({ color: 0xc0392b, roughness: 0.48, metalness: 0.06, clippingPlanes: [] });
      const darkMuscle = new THREE.MeshStandardMaterial({ color: 0x9b2c23, roughness: 0.52, metalness: 0.05, clippingPlanes: [] });
      const artery = new THREE.MeshStandardMaterial({ color: 0xe74c3c, roughness: 0.38, metalness: 0.08, clippingPlanes: [] });
      const vein = new THREE.MeshStandardMaterial({ color: 0x7f3b4a, roughness: 0.46, metalness: 0.05, clippingPlanes: [] });
      const fat = new THREE.MeshStandardMaterial({ color: 0xd4a59a, roughness: 0.62, metalness: 0.02, clippingPlanes: [] });
      [[muscle, "muscle"], [darkMuscle, "muscle"], [artery, "artery"], [vein, "vein"], [fat, "muscle"]].forEach(([m, kind]) => {
        m.userData.base = m.color.clone();
        m.userData.kind = kind;
        tintMats.push(m);
      });
      function tube(points, radius, mat, segs) {
        const curve = new THREE.CatmullRomCurve3(points.map((p) => new THREE.Vector3(...p)));
        return new THREE.Mesh(new THREE.TubeGeometry(curve, segs || 40, radius, 14, false), mat);
      }
      const g = new THREE.Group();
      const lv = new THREE.Mesh(new THREE.SphereGeometry(1.0, 48, 36), muscle);
      lv.scale.set(1.08, 1.32, 0.92);
      lv.position.set(-0.18, -0.05, 0.02);
      lv.rotation.z = 0.42;
      g.add(lv);
      const rv = new THREE.Mesh(new THREE.SphereGeometry(0.78, 40, 32), darkMuscle);
      rv.scale.set(0.92, 1.12, 1.08);
      rv.position.set(0.38, 0.12, 0.32);
      rv.rotation.z = 0.18;
      g.add(rv);
      const apex = new THREE.Mesh(new THREE.ConeGeometry(0.62, 0.95, 28), muscle);
      apex.position.set(-0.42, -1.18, 0.06);
      apex.rotation.z = 0.48;
      g.add(apex);
      const la = new THREE.Mesh(new THREE.SphereGeometry(0.46, 28, 22), darkMuscle);
      la.scale.set(1.15, 0.85, 0.95);
      la.position.set(-0.62, 0.78, -0.32);
      g.add(la);
      const ra = new THREE.Mesh(new THREE.SphereGeometry(0.5, 28, 22), muscle);
      ra.scale.set(1.05, 0.9, 1.0);
      ra.position.set(0.42, 0.82, 0.08);
      g.add(ra);
      g.add(tube([
        [0.02, 0.52, 0.12], [0.0, 1.02, 0.02], [0.08, 1.38, -0.18],
        [-0.12, 1.48, -0.48], [-0.38, 1.22, -0.62], [-0.42, 0.62, -0.58],
      ], 0.145, artery, 56));
      g.add(tube([[0.48, 0.72, 0.08], [0.52, 1.12, 0.06], [0.5, 1.55, 0.04], [0.48, 1.92, 0.02]], 0.12, vein, 24));
      mountFitted(g);
    }

    function loadGltb(url) {
      return new Promise((resolve, reject) => {
        if (!GLTFLoader || !url) {
          reject(new Error("no loader"));
          return;
        }
        const loader = new GLTFLoader();
        loader.load(url, resolve, undefined, reject);
      });
    }

    const modelUrl = (parentElement.__heartLatest && parentElement.__heartLatest.model_url)
      || "/app/static/human_heart.glb";

    loadGltb(modelUrl)
      .then((gltf) => {
        mountFitted(gltf.scene);
        if (loadEl) loadEl.classList.add("hidden");
        let meshes = 0;
        gltf.scene.traverse((o) => { if (o.isMesh) meshes += 1; });
        if (hint) hint.textContent = meshes ? "Drag to orbit · tap a pin" : "Heart model loaded with no meshes";
      })
      .catch(() => {
        buildFallback();
        if (loadEl) loadEl.classList.add("hidden");
        if (hint) hint.textContent = "Simplified heart · drag to orbit";
      });

    const state = {
      status: "idle",
      riskPct: null,
      bpm: 72,
      rotY: 0.12,
      rotX: 0.06,
      dist: 3.15,
      dragging: false,
      lastX: 0,
      lastY: 0,
      auto: true,
      lastMove: 0,
      exploded: false,
      clipped: false,
      activePin: null,
      pulse: 0,
    };

    if (pinsEl) {
      pinsEl.innerHTML = HOTSPOTS.map((h) => (
        `<button type="button" class="pin" data-pin="${h.id}" title="${h.name}"></button>`
      )).join("");
    }

    function showCard(h) {
      state.activePin = h ? h.id : null;
      if (!cardEl) return;
      if (!h) {
        cardEl.classList.add("hidden");
        return;
      }
      const extra = state.status === "yes"
        ? " This region is highlighted because the model called higher risk."
        : state.status === "no"
          ? " The current estimate does not flag this as a high-risk pattern."
          : " Calculate risk to colour this region by the live estimate.";
      cardTitle.textContent = h.name;
      cardBody.textContent = h.text + extra;
      cardEl.classList.remove("hidden");
      pinsEl.querySelectorAll(".pin").forEach((b) => b.classList.toggle("active", b.dataset.pin === h.id));
    }

    function projectPins() {
      if (!pinsEl) return;
      const rect = stage.getBoundingClientRect();
      const w = rect.width || 1;
      const h = rect.height || 1;
      HOTSPOTS.forEach((spot) => {
        const btn = pinsEl.querySelector(`[data-pin="${spot.id}"]`);
        if (!btn || !spot.obj) return;
        const v = new THREE.Vector3();
        spot.obj.getWorldPosition(v);
        v.project(camera);
        const x = (v.x * 0.5 + 0.5) * w;
        const y = (-v.y * 0.5 + 0.5) * h;
        const behind = v.z > 1 || v.z < -1;
        btn.style.left = `${x}px`;
        btn.style.top = `${y}px`;
        btn.style.display = behind ? "none" : "block";
      });
    }

    function resize() {
      const rect = stage.getBoundingClientRect();
      const w = Math.max(320, Math.floor(rect.width || 520));
      const h = Math.max(320, Math.floor(rect.height || 430));
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    }

    const onDown = (e) => {
      state.dragging = true;
      state.auto = false;
      state.lastX = e.clientX;
      state.lastY = e.clientY;
      state.lastMove = performance.now();
      canvas.setPointerCapture(e.pointerId);
    };
    const onMove = (e) => {
      if (!state.dragging) return;
      state.rotY += (e.clientX - state.lastX) * 0.008;
      state.rotX = Math.max(-0.7, Math.min(0.7, state.rotX + (e.clientY - state.lastY) * 0.008));
      state.lastX = e.clientX;
      state.lastY = e.clientY;
      state.lastMove = performance.now();
    };
    const onUp = () => { state.dragging = false; };
    const onWheel = (e) => {
      e.preventDefault();
      state.dist = Math.max(2.2, Math.min(6.5, state.dist + e.deltaY * 0.008));
      state.auto = false;
      state.lastMove = performance.now();
    };
    canvas.addEventListener("pointerdown", onDown);
    canvas.addEventListener("pointermove", onMove);
    canvas.addEventListener("pointerup", onUp);
    canvas.addEventListener("pointercancel", onUp);
    canvas.addEventListener("wheel", onWheel, { passive: false });

    if (pinsEl) {
      pinsEl.addEventListener("click", (e) => {
        const btn = e.target.closest("[data-pin]");
        if (!btn) return;
        const spot = HOTSPOTS.find((h) => h.id === btn.dataset.pin);
        showCard(state.activePin === (spot && spot.id) ? null : spot);
        state.auto = false;
      });
    }
    const closeBtn = parentElement.querySelector("#heart-card-close");
    if (closeBtn) closeBtn.addEventListener("click", () => showCard(null));

    const presets = {
      anterior: { rotX: 0.08, rotY: 0.12, dist: 3.15 },
      posterior: { rotX: 0.08, rotY: 3.26, dist: 3.2 },
      side: { rotX: 0.04, rotY: 1.58, dist: 3.25 },
      reset: { rotX: 0.06, rotY: 0.12, dist: 3.15 },
    };
    if (barEl) {
      barEl.addEventListener("click", (e) => {
        const btn = e.target.closest("button");
        if (!btn) return;
        if (btn.dataset.cam && presets[btn.dataset.cam]) {
          Object.assign(state, presets[btn.dataset.cam]);
          state.auto = false;
          state.lastMove = performance.now();
        }
        if (btn.dataset.action === "explode") {
          state.exploded = !state.exploded;
          setExploded(state.exploded);
          btn.classList.toggle("on", state.exploded);
        }
        if (btn.dataset.action === "clip") {
          state.clipped = !state.clipped;
          setClipped(state.clipped);
          btn.classList.toggle("on", state.clipped);
        }
      });
    }

    const t0 = performance.now();
    let raf = 0;
    function frame(now) {
      const dt = (now - t0) / 1000;
      if (!state.dragging && now - state.lastMove > 2200) state.auto = true;
      if (state.auto) state.rotY += 0.0026;
      resize();
      const bpm = Math.max(48, Math.min(140, Number(state.bpm) || 72));
      const beatHz = bpm / 60;
      const pulse = Math.pow(Math.abs(Math.sin(dt * beatHz * Math.PI)), 8);
      state.pulse = pulse;
      const amp = 0.028 + Math.min(0.04, (bpm - 60) / 900) + (state.status === "yes" ? 0.018 : 0);
      heart.scale.setScalar(1 + pulse * amp);
      heart.rotation.set(state.rotX, state.rotY, 0);
      camera.position.set(0.08, 0.04, state.dist);
      camera.lookAt(0, 0.02, 0);
      heat(state.status, state.riskPct, pulse);
      projectPins();
      renderer.render(scene, camera);
      raf = requestAnimationFrame(frame);
    }
    raf = requestAnimationFrame(frame);

    parentElement.__heartUpdate = (next) => {
      state.status = applyStatus(next);
      state.riskPct = next && next.risk_pct != null ? Number(next.risk_pct) : null;
      state.bpm = next && next.bpm != null ? Number(next.bpm) : 72;
      heat(state.status, state.riskPct, state.pulse);
      if (state.activePin) {
        const spot = HOTSPOTS.find((h) => h.id === state.activePin);
        if (spot) showCard(spot);
      }
    };
    parentElement.__heartUpdate(parentElement.__heartLatest);

    parentElement.__heartCleanup = () => {
      cancelAnimationFrame(raf);
      canvas.removeEventListener("pointerdown", onDown);
      canvas.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("pointerup", onUp);
      canvas.removeEventListener("pointercancel", onUp);
      canvas.removeEventListener("wheel", onWheel);
      renderer.dispose();
    };
  }
}
"""

_heart_component = st.components.v2.component(
    "human_heart_glb_v14",
    html=_HEART_HTML,
    css=_HEART_CSS,
    js=_HEART_JS,
)


def render_beating_heart(risk_pct=None, label=None, caption=None, bpm=72, key="beating_heart", height=520):
    """Mount the interactive anatomical heart. `label` is Yes/No after a prediction."""
    return _heart_component(
        data={
            "risk_pct": risk_pct,
            "label": label,
            "caption": caption,
            "bpm": bpm,
            "model_url": _heart_model_url(),
        },
        key=key,
        width="stretch",
        height=height,
    )


def plot_patient_cloud_3d(raw_df, target_col, sample_n=1600, seed=42):
    """Interactive 3D scatter of Age × BMI × cholesterol, colored by diagnosis."""
    cols = ["Age", "BMI", "Cholesterol Level", target_col]
    df = raw_df.dropna(subset=cols).copy()
    if len(df) > sample_n:
        df = df.sample(sample_n, random_state=seed)

    yes = df[target_col] == "Yes"
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=df.loc[~yes, "Age"],
        y=df.loc[~yes, "BMI"],
        z=df.loc[~yes, "Cholesterol Level"],
        mode="markers",
        name="No heart disease",
        marker=dict(size=3.2, color="#0f766e", opacity=0.55),
        hovertemplate="Age: %{x:.0f}<br>BMI: %{y:.1f}<br>Cholesterol: %{z:.0f}<extra>No</extra>",
    ))
    fig.add_trace(go.Scatter3d(
        x=df.loc[yes, "Age"],
        y=df.loc[yes, "BMI"],
        z=df.loc[yes, "Cholesterol Level"],
        mode="markers",
        name="Heart disease",
        marker=dict(size=3.6, color="#e11d48", opacity=0.7),
        hovertemplate="Age: %{x:.0f}<br>BMI: %{y:.1f}<br>Cholesterol: %{z:.0f}<extra>Yes</extra>",
    ))
    fig.update_layout(
        height=520,
        margin=dict(l=0, r=0, t=8, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        scene=dict(
            xaxis_title="Age",
            yaxis_title="BMI",
            zaxis_title="Cholesterol",
            bgcolor="rgba(248,250,250,0.35)",
            xaxis=dict(backgroundcolor="rgba(255,255,255,0.55)", gridcolor="#d5e3e0", showspikes=False),
            yaxis=dict(backgroundcolor="rgba(255,255,255,0.55)", gridcolor="#d5e3e0", showspikes=False),
            zaxis=dict(backgroundcolor="rgba(255,255,255,0.55)", gridcolor="#d5e3e0", showspikes=False),
            camera=dict(eye=dict(x=1.55, y=1.35, z=1.15)),
        ),
    )
    return fig
