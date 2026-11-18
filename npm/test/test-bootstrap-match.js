// Verify closest-compatible Linux asset selection in npm/bin/acloudviewer.js
// Run: npm test  (from npm/). Zero dependencies, Node >= 18.
"use strict";
const path = require("path");
const { findInstallerApp, linuxTagCandidates } =
  require(path.join(__dirname, "..", "bin", "acloudviewer.js"));

const A = (name) => ({ name });
// Real main-devel asset names (verified via API 2026-11-16)
const FULL = [
  "ACloudViewer-3.9.5+1cf1fa06-ubuntu20.04-cpu-amd64.run",
  "ACloudViewer-3.9.5+1cf1fa06-ubuntu20.04-cuda-amd64.run",
  "ACloudViewer-3.9.5+1cf1fa06-ubuntu22.04-cpu-amd64.run",
  "ACloudViewer-3.9.5+1cf1fa06-ubuntu22.04-cuda-amd64.run",
  "ACloudViewer-3.9.5+1cf1fa06-ubuntu24.04-cpu-amd64.run",
  "ACloudViewer-3.9.5+1cf1fa06-ubuntu24.04-cuda-amd64.run",
  "ACloudViewer-3.9.5+1cf1fa06-win-cpu-amd64.exe",
  "ACloudViewer-3.9.5+1cf1fa06-mac-cpu-ARM64.dmg",
].map(A);
const plat = (id, version, glibc) => ({ distro: { id, version }, glibc, macArch: "amd64" });

let pass = 0, fail = 0;
function check(label, assets, p, flavor, expect) {
  const got = findInstallerApp(assets, p, flavor);
  const name = got ? got.name : null;
  const ok = expect === null ? name === null : (name !== null && name.includes(expect));
  console.log(`${ok ? "PASS" : "FAIL"}  ${label}  ->  ${name}`);
  ok ? pass++ : fail++;
}

// 1. This machine: Ubuntu 22.04 exact match, never 24.04
check("22.04 exact cpu", FULL, plat("ubuntu", "22.04", "2.35"), "cpu", "ubuntu22.04-cpu");
check("22.04 exact cuda", FULL, plat("ubuntu", "22.04", "2.35"), "cuda", "ubuntu22.04-cuda");
// 2. Missing exact tag -> newest older build, never newer
const no2204 = FULL.filter((a) => !a.name.includes("ubuntu22.04"));
check("22.04 host, no 22.04 assets -> 20.04", no2204, plat("ubuntu", "22.04", "2.35"), "cpu", "ubuntu20.04");
// 3. Never upward: 20.04 host with only 22.04/24.04 -> null
const onlyNewer = FULL.filter((a) => a.name.includes("ubuntu22.04") || a.name.includes("ubuntu24.04"));
check("20.04 host, only newer builds -> null", onlyNewer, plat("ubuntu", "20.04", "2.31"), "cpu", null);
// 4. Non-LTS Ubuntu 24.10 (glibc 2.40) -> ubuntu24.04
check("24.10 non-LTS -> 24.04", FULL, plat("ubuntu", "24.10", "2.40"), "cpu", "ubuntu24.04");
// 5. Future Ubuntu 26.04 (glibc 2.43) -> newest published 24.04
check("26.04 future -> 24.04", FULL, plat("ubuntu", "26.04", "2.43"), "cpu", "ubuntu24.04");
// 6. Non-Ubuntu distro via glibc (Debian 12, 2.36) -> 22.04
check("debian glibc2.36 -> 22.04", FULL, plat("debian", "12", "2.36"), "cpu", "ubuntu22.04");
// 7. Old host glibc 2.27 -> 18.04 tag absent -> null (no compatible asset)
check("glibc2.27 host, no 18.04 assets -> null", FULL, plat("centos", "7", "2.27"), "cpu", null);
// 8. cuda->cpu flavor fallback within the chosen tag
const cpuOnly2204 = FULL.filter((a) => !a.name.includes("ubuntu22.04-cuda"));
check("22.04 cuda missing -> 22.04 cpu", cpuOnly2204, plat("ubuntu", "22.04", "2.35"), "cuda", "ubuntu22.04-cpu");
// 9. Unknown glibc -> conservative (<=2.31 floors only)
check("unknown glibc ubuntu: exact tag still first", FULL, plat("ubuntu", "22.04", ""), "cpu", "ubuntu22.04-cpu");
check("unknown glibc non-ubuntu -> 20.04", FULL, plat("arch", "", ""), "cpu", "ubuntu20.04");
// 10. Candidate ordering sanity
const cand = linuxTagCandidates(plat("ubuntu", "22.04", "2.35"));
const orderOk = JSON.stringify(cand) === JSON.stringify(["ubuntu22.04", "ubuntu20.04", "ubuntu18.04"]);
console.log(`${orderOk ? "PASS" : "FAIL"}  candidate order 22.04 -> ${JSON.stringify(cand)}`);
orderOk ? pass++ : fail++;

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);