// Headless render driver for the demo storyboard (run with `bun`).
//
// Plain `chromium --headless=new URL` does not actually navigate/run the page,
// and `--screenshot` exits after a single frame — neither can drive a multi-frame
// capture loop. So we launch chromium with the DevTools endpoint and use CDP
// (Target.createTarget) to open the capture URL as a real, running tab. The page
// itself renders each frame and POSTs the PNGs to the Go server; this driver just
// opens it and waits for the frames / encoded MP4 to appear on disk.
//
//   PORT=4001 RES=1280x720 bun scripts/demo/render.js              # full clip -> MP4
//   RES=960x540 STEP=45 ENCODE=0 bun scripts/demo/render.js        # sparse stills (no encode)
//
// Env: PORT (4001), RES (1280x720), SESSION, FRAMES (cap), STEP (sparse), ENCODE (1),
//      CHROME (chromium), WORKDIR (/tmp/freesplatter-demo), DEVPORT (9222).
import { readdirSync, existsSync, statSync } from "node:fs";

const PORT = process.env.PORT || "4001";
const RES = process.env.RES || "1280x720";
const [W, H] = RES.split("x");
const SESSION = process.env.SESSION || `demo_${RES}`;
const FRAMES = process.env.FRAMES || "";
const STEP = process.env.STEP || "";
const ENCODE = process.env.ENCODE ?? "1";
const CHROME = process.env.CHROME || "chromium";
const WORK = process.env.WORKDIR || "/tmp/freesplatter-demo";
const DEVPORT = process.env.DEVPORT || "9222";
const SERVER = `http://127.0.0.1:${PORT}`;

let url = process.env.URL || `${SERVER}/demo.html?capture=1&session=${SESSION}&w=${W}&h=${H}`;
if (!process.env.URL) {
  if (FRAMES) url += `&frames=${FRAMES}`;
  if (STEP) url += `&step=${STEP}`;
  if (ENCODE === "0") url += `&encode=0`;
  if (process.env.ZOOM) url += `&zoom=${process.env.ZOOM}`;       // camera zoom toward the splats
  if (process.env.IMGBIG) url += `&imgbig=${process.env.IMGBIG}`; // source-photo size in the big phase
  if (process.env.SPACING) url += `&spacing=${process.env.SPACING}`; // world gap between scenes
  if (process.env.HOLD) url += `&hold=${process.env.HOLD}`;           // seconds the photos sit big
  if (process.env.MAXSPLATS) url += `&maxsplats=${process.env.MAXSPLATS}`; // per-scene render ceiling
  if (process.env.BUILDCURVE) url += `&buildcurve=${process.env.BUILDCURVE}`; // reveal ease-in exponent
  // timeline phases (seconds)
  for (const [env, q] of [["TRAVEL","travel"],["IMAGES_IN","images_in"],["IMAGES_SHRINK","images_shrink"],["SPLAT_LAG","splat_lag"],["BUILD","build"],["SETTLE","settle"]])
    if (process.env[env]) url += `&${q}=${process.env[env]}`;
  if (process.env.SAT) url += `&sat=${process.env.SAT}`;          // vibrance
}

const framesDir = `${WORK}/frames/${SESSION}`;
const videoPath = `${WORK}/video/${SESSION}.mp4`;
const nFrames = () => { try { return readdirSync(framesDir).filter(f => f.endsWith(".png")).length; } catch { return 0; } };

// GL backend: software (SwiftShader, default — runs anywhere but slow and chokes
// on big integer textures) vs real GPU via ANGLE/Vulkan (GPU=1, far faster).
const glFlags = process.env.GPU === "1"
  ? ["--use-gl=angle", "--use-angle=vulkan", "--ignore-gpu-blocklist", "--enable-gpu"]
  : ["--enable-unsafe-swiftshader", "--use-gl=angle", "--use-angle=swiftshader"];
const proc = Bun.spawn([
  CHROME, "--headless=new", "--no-sandbox", ...glFlags,
  "--hide-scrollbars", "--no-first-run",
  // new-headless throttles/backgrounds non-foreground tabs to a crawl; these keep
  // the capture tab running full-speed.
  "--disable-background-timer-throttling", "--disable-renderer-backgrounding",
  "--disable-backgrounding-occluded-windows", "--disable-features=CalculateNativeWinOcclusion",
  `--user-data-dir=/tmp/fsdemo-cr-${SESSION}`, `--window-size=${W},${H}`,
  `--remote-debugging-port=${DEVPORT}`, "about:blank",
], { stdout: "ignore", stderr: "ignore" });

// Grab the *foreground* page target (the initial about:blank tab) and navigate it
// in place — a created/background target gets throttled and never runs the loop.
async function pageTarget() {
  for (let i = 0; i < 80; i++) {
    try {
      const list = await (await fetch(`http://127.0.0.1:${DEVPORT}/json`)).json();
      const pg = list.find(t => t.type === "page" && t.webSocketDebuggerUrl);
      if (pg) return pg.webSocketDebuggerUrl;
    } catch {}
    await Bun.sleep(500);
  }
  return null;
}
const wsUrl = await pageTarget();
if (!wsUrl) { console.error("error: no page target"); proc.kill(); process.exit(1); }

const ws = new WebSocket(wsUrl);
await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });
ws.send(JSON.stringify({ id: 1, method: "Page.navigate", params: { url } }));
console.error(`driving ${url}`);

// Wait for work: for a full encode, until the MP4 appears; for sparse/no-encode,
// until the frame count stops growing. Report progress; bail on a long stall.
let last = -1, stable = 0, started = Date.now();
for (let i = 0; i < 5400; i++) {           // up to ~3h ceiling
  await Bun.sleep(2000);
  if (ENCODE !== "0" && existsSync(videoPath)) {
    const sz = statSync(videoPath).size;
    console.log(`VIDEO_READY ${videoPath} ${(sz / 1e6).toFixed(2)}MB`);
    proc.kill(); process.exit(0);
  }
  const n = nFrames();
  if (n !== last) { console.error(`  ${n} frames (${((Date.now() - started) / 1000) | 0}s)`); last = n; stable = 0; }
  else if (n > 0) { if (++stable >= (ENCODE === "0" ? 4 : 30)) {
    console.log(ENCODE === "0" ? `FRAMES_DONE ${n} frames in ${framesDir}` : `STALLED at ${n} frames (encode may still run)`);
    if (ENCODE === "0") { proc.kill(); process.exit(0); }
  } }
}
console.error("timed out"); proc.kill(); process.exit(1);
