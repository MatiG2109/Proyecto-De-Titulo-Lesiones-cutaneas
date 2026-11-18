// ================= CONFIGURACIÓN (coincide con get_transforms(train=False) de la tesis) =================
// Resize((224, 224), bilinear, antialias) -> ToTensor -> Normalize(ImageNet)
const MODEL_URL = "model/skincancer.onnx";
const CLASSES = ["AK", "BCC", "BKL", "DF", "MEL", "NV", "SCC", "VASC"];
const SIZE = 224;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];
const MAX_SIDE = 4096; // imágenes más grandes se pre-reducen para no agotar la memoria
// =========================================================================================================

const NOMBRES = {
  AK: "Queratosis actínica", BCC: "Carcinoma basocelular", BKL: "Queratosis benigna",
  DF: "Dermatofibroma", MEL: "Melanoma", NV: "Nevus melanocítico",
  SCC: "Carcinoma escamoso", VASC: "Lesión vascular",
};
const RIESGO = { MEL: "Maligno", BCC: "Maligno", SCC: "Maligno", AK: "Premaligno",
                 NV: "Benigno", BKL: "Benigno", DF: "Benigno", VASC: "Benigno" };

const $ = (id) => document.getElementById(id);
function showResult(v) { $("result").hidden = !v; $("ph").hidden = v; }
let file = null;
let session = null;

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
ort.env.wasm.numThreads = 1;

$("file").addEventListener("change", (e) => {
  file = e.target.files[0] || null;
  if (!file) return;
  $("dropText").textContent = file.name + " (clic para cambiar)";
  $("preview").src = URL.createObjectURL(file);
  $("preview").hidden = false;
  showResult(false);
  $("status").textContent = "";
});

async function loadModel() {
  if (session) return session;
  const prog = $("prog");
  prog.hidden = false;
  prog.value = 0;
  $("status").textContent = "Descargando el modelo (solo la primera vez)…";

  const resp = await fetch(MODEL_URL);
  if (!resp.ok) throw new Error("No se encontró el modelo (" + resp.status + ")");
  const total = Number(resp.headers.get("Content-Length")) || 0;
  const reader = resp.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    if (total) prog.value = Math.min(100, (got / total) * 100);
    $("status").textContent = "Descargando el modelo… " + (got / 1048576).toFixed(0) + " MB";
  }
  const buf = new Uint8Array(got);
  let off = 0;
  for (const c of chunks) { buf.set(c, off); off += c.length; }

  $("status").textContent = "Preparando el modelo…";
  session = await ort.InferenceSession.create(buf, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
  prog.hidden = true;
  return session;
}

function loadImage(f) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("El archivo no es una imagen válida."));
    img.src = URL.createObjectURL(f);
  });
}

// ---- Redimensionado bilineal con antialias, equivalente al de PIL (reduce con filtro triangular) ----
function coeffs(inSize, outSize) {
  const scale = inSize / outSize;
  const fscale = Math.max(scale, 1);
  const support = fscale; // bilinear: soporte 1.0 * fscale
  const out = [];
  for (let xx = 0; xx < outSize; xx++) {
    const center = (xx + 0.5) * scale;
    let xmin = Math.max(Math.floor(center - support + 0.5), 0);
    const xmax = Math.min(Math.floor(center + support + 0.5), inSize);
    const w = [];
    let sum = 0;
    for (let x = xmin; x < xmax; x++) {
      const t = Math.abs((x + 0.5 - center) / fscale);
      const v = t < 1 ? 1 - t : 0;
      w.push(v);
      sum += v;
    }
    out.push({ xmin, w: w.map((v) => (sum ? v / sum : 0)) });
  }
  return out;
}

function resizeRGB(src, w, h, ow, oh) {
  // src: Uint8ClampedArray RGBA (w*h*4). Devuelve Uint8ClampedArray RGB (ow*oh*3).
  const cx = coeffs(w, ow);
  const cy = coeffs(h, oh);
  const tmp = new Uint8ClampedArray(ow * h * 3); // pasada horizontal (se redondea a 8 bits, como PIL)
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < ow; x++) {
      const { xmin, w: ws } = cx[x];
      let r = 0, g = 0, b = 0;
      for (let k = 0; k < ws.length; k++) {
        const p = (y * w + xmin + k) * 4;
        r += src[p] * ws[k]; g += src[p + 1] * ws[k]; b += src[p + 2] * ws[k];
      }
      const o = (y * ow + x) * 3;
      tmp[o] = r; tmp[o + 1] = g; tmp[o + 2] = b;
    }
  }
  const dst = new Uint8ClampedArray(ow * oh * 3); // pasada vertical
  for (let y = 0; y < oh; y++) {
    const { xmin: ymin, w: ws } = cy[y];
    for (let x = 0; x < ow; x++) {
      let r = 0, g = 0, b = 0;
      for (let k = 0; k < ws.length; k++) {
        const p = ((ymin + k) * ow + x) * 3;
        r += tmp[p] * ws[k]; g += tmp[p + 1] * ws[k]; b += tmp[p + 2] * ws[k];
      }
      const o = (y * ow + x) * 3;
      dst[o] = r; dst[o + 1] = g; dst[o + 2] = b;
    }
  }
  return dst;
}

function toTensor(img) {
  let w = img.naturalWidth, h = img.naturalHeight;
  const k = Math.min(1, MAX_SIDE / Math.max(w, h));
  w = Math.max(1, Math.round(w * k));
  h = Math.max(1, Math.round(h * k));
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(img, 0, 0, w, h);
  const rgb = resizeRGB(ctx.getImageData(0, 0, w, h).data, w, h, SIZE, SIZE);

  const n = SIZE * SIZE;
  const data = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < 3; c++) {
      data[c * n + i] = (rgb[i * 3 + c] / 255 - MEAN[c]) / STD[c];
    }
  }
  return new ort.Tensor("float32", data, [1, 3, SIZE, SIZE]);
}

function softmax(a) {
  const m = Math.max(...a);
  const e = a.map((v) => Math.exp(v - m));
  const s = e.reduce((x, y) => x + y, 0);
  return e.map((v) => v / s);
}

function mostrar(probs) {
  const idx = probs.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]);
  const top = CLASSES[idx[0][1]];
  $("rTitle").textContent = NOMBRES[top] + " (" + top + ")";
  $("rMeta").textContent = "Categoría: " + RIESGO[top] + " · Confianza: " + (idx[0][0] * 100).toFixed(1) + "%";
  $("bars").innerHTML = "";
  idx.forEach(([p, i]) => {
    const bar = document.createElement("div");
    bar.className = "bar";
    const fill = document.createElement("div");
    fill.style.width = Math.max(p * 100, 4) + "%";
    fill.textContent = CLASSES[i] + " - " + NOMBRES[CLASSES[i]] + " " + (p * 100).toFixed(1) + "%";
    bar.appendChild(fill);
    $("bars").appendChild(bar);
  });
  showResult(true);
}

$("btn").addEventListener("click", async () => {
  if (!file) { $("status").textContent = "Selecciona una imagen primero."; return; }
  $("btn").disabled = true;
  showResult(false);
  try {
    const s = await loadModel();
    $("status").textContent = "Analizando…";
    const img = await loadImage(file);
    const out = await s.run({ input: toTensor(img) });
    mostrar(softmax(Array.from(out.logits.data)));
    $("status").textContent = "";
  } catch (err) {
    console.error(err);
    $("status").textContent = "Error: " + err.message;
    $("prog").hidden = true;
  } finally {
    $("btn").disabled = false;
  }
});
