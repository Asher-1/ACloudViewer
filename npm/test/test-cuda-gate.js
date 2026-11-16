// Pure-function tests for the explicit-CUDA + driver-gate contract.
// Run: npm test  (from npm/). Zero dependencies, Node >= 18.
const assert = require("assert");
const path = require("path");
const { spawnSync } = require("child_process");
const m = require(path.join(__dirname, "..", "bin", "acloudviewer.js"));

let pass = 0, skip = 0;
function t(label, fn) { fn(); pass++; console.log(`PASS ${label}`); }

// pickFlavor: CUDA only on explicit --cuda; GPU presence is irrelevant.
t("default flavor is cpu", () => assert.strictEqual(m.pickFlavor(new Map()), "cpu"));
t("--cpu stays cpu", () => assert.strictEqual(m.pickFlavor(new Map([["cpu", true]])), "cpu"));
t("--cuda opts in", () => assert.strictEqual(m.pickFlavor(new Map([["cuda", true]])), "cuda"));

// cudaSupported: compatibility is per CUDA MAJOR (12.x drivers all run a
// 12.x build); only a lower major is incompatible.
t("requirement is 12.6", () => assert.strictEqual(m.REQUIRED_CUDA, "12.6"));
t("cuda 13.0 supported", () => assert.strictEqual(m.cudaSupported({ driver: "580.95.05", cuda: "13.0" }), true));
t("cuda 12.6 supported", () => assert.strictEqual(m.cudaSupported({ driver: "560.35.03", cuda: "12.6" }), true));
t("cuda 12.4 same major supported", () => assert.strictEqual(m.cudaSupported({ driver: "550.54.14", cuda: "12.4" }), true));
t("cuda 12.0 same major supported", () => assert.strictEqual(m.cudaSupported({ driver: "525.60.13", cuda: "12.0" }), true));
t("cuda 11.4 major mismatch refused", () => assert.strictEqual(m.cudaSupported({ driver: "470.256.02", cuda: "11.4" }), false));
t("no driver refused", () => assert.strictEqual(m.cudaSupported(null), false));
t("future major supported", () => assert.strictEqual(m.cudaSupported({ driver: "x", cuda: "14.0" }), true));

// Contradictory flavors must fail fast, before any network access
// (mirrors installer.py's mutual-exclusion refusal).
t("--cuda + --cpu-only refused", () => {
  const r = spawnSync(process.execPath,
    [path.join(__dirname, "..", "bin", "acloudviewer.js"), "install", "--cuda", "--cpu-only"],
    { encoding: "utf8", timeout: 15000 });
  assert.strictEqual(r.status, 1);
  assert.match(r.stderr, /mutually exclusive/);
});

// Live probe on this machine: opportunistic — CI runners without an NVIDIA
// driver SKIP instead of failing (the predicate matrix above is the gate).
const report = m.nvidiaDriverReport();
if (report) {
  t("live nvidia-smi report parses", () => {
    assert.match(report.driver, /^\d+\.\d+/);
    assert.match(report.cuda, /^\d+\.\d+$/);
  });
  t("live driver passes the gate predicate", () => assert.strictEqual(m.cudaSupported(report), true));
} else {
  skip += 2;
  console.log("SKIP live nvidia-smi probes (no working NVIDIA driver on this host)");
}

console.log(`\n${pass} passed, ${skip} skipped`);