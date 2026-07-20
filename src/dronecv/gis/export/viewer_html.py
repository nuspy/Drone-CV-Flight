"""Realtime WebGL viewer for the exported scene.glb (three.js).

Advanced PBR rendering: real sky + sun at the scene's solar position, soft
shadows, ACES tone mapping, environment reflections. Free-fly controls:

    mouse drag     look around (orientation)
    click          center the view on the clicked point (pan)
    W/S            forward / back      A/D  strafe left / right
    Q/E            down / up
    G              toggle GLOBAL vs VIEW-RELATIVE movement
    wheel          zoom in / out (dolly)
    Tab            reset view / zoom
    Shift          move faster

The page loads `scene.glb` + `scene_meta.json` from its own directory, so it
must be SERVED (browsers block file:// fetches) — `dronecv gis view` does
that. It also accepts a drag-and-dropped .glb as a fallback.
"""

VIEWER_HTML = """<!doctype html>
<html><head>
<meta charset="utf-8"/>
<title>DroneCV scene viewer</title>
<style>
  html,body{height:100%;margin:0;background:#0b0e13;overflow:hidden;font:13px system-ui,sans-serif}
  #c{display:block;width:100%;height:100%}
  #hud{position:fixed;left:10px;top:10px;color:#dfe6ee;background:rgba(10,14,20,.62);
       padding:10px 12px;border-radius:8px;line-height:1.5;pointer-events:none;max-width:320px}
  #hud b{color:#7fd1ff}
  #mode{color:#ffd479}
  #err{position:fixed;left:50%;top:50%;transform:translate(-50%,-50%);color:#ff8a8a;
       background:rgba(10,14,20,.9);padding:16px 20px;border-radius:8px;display:none}
  #drop{position:fixed;inset:0;display:none;align-items:center;justify-content:center;
        background:rgba(10,14,20,.7);color:#dfe6ee;font-size:18px}
</style>
</head><body>
<canvas id="c"></canvas>
<div id="hud">
  <b>DroneCV scene</b><br/>
  mouse drag: look &nbsp; click: center<br/>
  W/S/A/D move &nbsp; Q/E down/up<br/>
  wheel zoom &nbsp; Tab reset &nbsp; Shift faster<br/>
  G: movement = <span id="mode">view-relative</span>
</div>
<div id="err"></div>
<div id="drop">drop a scene.glb here</div>
<script type="importmap">
{ "imports": {
  "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
  "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
} }
</script>
<script type="module">
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { Sky } from 'three/addons/objects/Sky.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

const canvas = document.getElementById('c');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const scene = new THREE.Scene();
scene.fog = new THREE.Fog(0x9fb4cc, 800, 6000);
const camera = new THREE.PerspectiveCamera(60, 1, 0.5, 40000);
camera.position.set(0, 300, 600);

// ---- sky + sun ----
const sky = new Sky(); sky.scale.setScalar(45000); scene.add(sky);
const sun = new THREE.Vector3();
const sunLight = new THREE.DirectionalLight(0xfff2d8, 3.0);
sunLight.castShadow = true;
sunLight.shadow.mapSize.set(2048, 2048);
sunLight.shadow.bias = -0.0004;
scene.add(sunLight, sunLight.target);
scene.add(new THREE.HemisphereLight(0xbcd3ff, 0x556070, 0.7));

function setSun(azDeg, elDeg) {
  const az = THREE.MathUtils.degToRad(azDeg), el = THREE.MathUtils.degToRad(elDeg);
  // three world: up=+Y, north=-Z, east=+X (export maps Z-up ENU -> Y-up).
  sun.set(Math.sin(az) * Math.cos(el), Math.sin(el), -Math.cos(az) * Math.cos(el));
  sky.material.uniforms['sunPosition'].value.copy(sun);
  sky.material.uniforms['turbidity'].value = 4;
  sky.material.uniforms['rayleigh'].value = 2;
  sky.material.uniforms['mieCoefficient'].value = 0.005;
  sky.material.uniforms['mieDirectionalG'].value = 0.8;
  sunLight.position.copy(sun).multiplyScalar(3000);
  sunLight.target.position.set(0, 0, 0);
}
setSun(160, 55);

const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;

// ---- load the model ----
const loader = new GLTFLoader();
function frame(obj) {
  const box = new THREE.Box3().setFromObject(obj);
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  target.copy(center);
  const r = Math.max(size.x, size.z) * 0.6 + size.y;
  camera.position.set(center.x, center.y + size.y * 0.8 + r * 0.4, center.z + r);
  camera.near = Math.max(0.5, r / 5000); camera.far = r * 40; camera.updateProjectionMatrix();
  scene.fog.near = r * 0.4; scene.fog.far = r * 4;
  // size the sun shadow frustum to the scene
  const s = sunLight.shadow.camera; const e = Math.max(size.x, size.z) * 0.7;
  s.left = -e; s.right = e; s.top = e; s.bottom = -e; s.near = 1; s.far = 8000; s.updateProjectionMatrix();
  home.pos = camera.position.clone(); home.target = target.clone();
}
function addModel(gltf) {
  if (window._model) scene.remove(window._model);
  const m = gltf.scene; window._model = m;
  m.traverse(o => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
  scene.add(m); frame(m);
}
function loadUrl(url) {
  loader.load(url, addModel, undefined, e => showErr('could not load scene.glb — ' + e));
}
function showErr(msg) { const d = document.getElementById('err'); d.textContent = msg; d.style.display = 'block'; }

fetch('scene_meta.json').then(r => r.ok ? r.json() : null).then(meta => {
  if (meta && meta.sun_azimuth_deg != null) setSun(meta.sun_azimuth_deg, meta.sun_elevation_deg);
}).catch(() => {});
loadUrl('scene.glb');

// ---- free-fly controls ----
const target = new THREE.Vector3();
const home = { pos: camera.position.clone(), target: target.clone() };
let globalMode = false;
const keys = new Set();
let dragging = false, moved = false, px = 0, py = 0;
let yaw = 0, pitch = -0.25;
function syncAngles() {
  const dir = new THREE.Vector3().subVectors(target, camera.position).normalize();
  yaw = Math.atan2(dir.x, dir.z); pitch = Math.asin(THREE.MathUtils.clamp(dir.y, -1, 1));
}
syncAngles();

canvas.addEventListener('mousedown', e => { dragging = true; moved = false; px = e.clientX; py = e.clientY; });
addEventListener('mouseup', e => {
  if (dragging && !moved) centerOn(e);
  dragging = false;
});
addEventListener('mousemove', e => {
  if (!dragging) return;
  const dx = e.clientX - px, dy = e.clientY - py; px = e.clientX; py = e.clientY;
  if (Math.abs(dx) + Math.abs(dy) > 2) moved = true;
  yaw -= dx * 0.0025; pitch -= dy * 0.0025;
  pitch = THREE.MathUtils.clamp(pitch, -1.5, 1.5);
});
canvas.addEventListener('wheel', e => {
  e.preventDefault();
  const dir = viewDir();
  const dist = camera.position.distanceTo(target);
  camera.position.addScaledVector(dir, -Math.sign(e.deltaY) * dist * 0.12);
}, { passive: false });
addEventListener('keydown', e => {
  if (e.code === 'Tab') { e.preventDefault(); camera.position.copy(home.pos); target.copy(home.target); syncAngles(); return; }
  if (e.code === 'KeyG') { globalMode = !globalMode; document.getElementById('mode').textContent = globalMode ? 'global' : 'view-relative'; return; }
  keys.add(e.code);
});
addEventListener('keyup', e => keys.delete(e.code));

function viewDir() { return new THREE.Vector3(Math.sin(yaw) * Math.cos(pitch), Math.sin(pitch), Math.cos(yaw) * Math.cos(pitch)); }
function centerOn(e) {
  const rect = canvas.getBoundingClientRect();
  const ndc = new THREE.Vector2(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1);
  const ray = new THREE.Raycaster(); ray.setFromCamera(ndc, camera);
  const hit = window._model ? ray.intersectObject(window._model, true)[0] : null;
  if (hit) target.copy(hit.point);
}

const clock = new THREE.Clock();
function moveStep(dt) {
  const speed = (keys.has('ShiftLeft') || keys.has('ShiftRight') ? 4 : 1) *
                camera.position.distanceTo(target) * 0.9 * dt;
  const fwd = globalMode ? new THREE.Vector3(0, 0, -1)
                         : new THREE.Vector3(Math.sin(yaw), 0, Math.cos(yaw)).normalize();
  const right = globalMode ? new THREE.Vector3(1, 0, 0)
                           : new THREE.Vector3(fwd.z, 0, -fwd.x);
  const up = new THREE.Vector3(0, 1, 0);
  const move = new THREE.Vector3();
  if (keys.has('KeyW')) move.add(fwd);
  if (keys.has('KeyS')) move.sub(fwd);
  if (keys.has('KeyD')) move.add(right);
  if (keys.has('KeyA')) move.sub(right);
  if (keys.has('KeyE')) move.add(up);
  if (keys.has('KeyQ')) move.sub(up);
  if (move.lengthSq() > 0) {
    move.normalize().multiplyScalar(speed);
    camera.position.add(move); target.add(move);
  }
}

function resize() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== w || canvas.height !== h) { renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix(); }
}
function tick() {
  const dt = Math.min(clock.getDelta(), 0.1);
  moveStep(dt);
  camera.lookAt(camera.position.clone().add(viewDir()));
  resize(); renderer.render(scene, camera);
  requestAnimationFrame(tick);
}
tick();

// ---- drag & drop fallback (standalone use) ----
const drop = document.getElementById('drop');
addEventListener('dragover', e => { e.preventDefault(); drop.style.display = 'flex'; });
addEventListener('dragleave', () => drop.style.display = 'none');
addEventListener('drop', e => {
  e.preventDefault(); drop.style.display = 'none';
  const f = e.dataTransfer.files[0]; if (!f) return;
  const url = URL.createObjectURL(f);
  loader.load(url, addModel, undefined, err => showErr('load failed: ' + err));
});
</script>
</body></html>"""


def write_viewer(out_dir) -> "object":
    from pathlib import Path

    p = Path(out_dir) / "viewer.html"
    p.write_text(VIEWER_HTML)
    return p
