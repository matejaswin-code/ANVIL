#!/usr/bin/env node
/* Copies the three.js files the frontend needs out of ./node_modules and into
 * ./frontend/vendor, so the browser loads them from our own server rather than
 * a CDN. Keeps the app fully functional offline, which is the whole point of a
 * local-first tool.
 *
 * Safe to run repeatedly. If node_modules isn't present it says so and exits 0
 * — the 3D preview is an enhancement, not a hard requirement, and the Python
 * side of the app works without it.
 */

const fs = require("fs");
const path = require("path");

const ROOT = path.resolve(__dirname, "..");
const NODE_MODULES = path.join(ROOT, "node_modules");
const VENDOR = path.join(ROOT, "frontend", "vendor");

const FILES = [
  ["three/build/three.module.js", "three.module.js"],
  ["three/examples/jsm/loaders/GLTFLoader.js", "GLTFLoader.js"],
  ["three/examples/jsm/controls/OrbitControls.js", "OrbitControls.js"],
  ["three/examples/jsm/utils/BufferGeometryUtils.js", "BufferGeometryUtils.js"],
];

if (!fs.existsSync(NODE_MODULES)) {
  console.log("[vendor] node_modules not found — skipping.");
  console.log("[vendor] Run `npm install` in the project root to enable the 3D preview.");
  process.exit(0);
}

fs.mkdirSync(VENDOR, { recursive: true });

let copied = 0;
for (const [src, dest] of FILES) {
  const from = path.join(NODE_MODULES, src);
  const to = path.join(VENDOR, dest);
  if (!fs.existsSync(from)) {
    console.warn(`[vendor] missing: ${src}`);
    continue;
  }
  let code = fs.readFileSync(from, "utf8");

  // The examples/jsm modules import bare "three"; rewrite to a relative path so
  // the browser can resolve them without an import map or bundler.
  code = code.replace(/from\s+["']three["']/g, 'from "./three.module.js"');
  code = code.replace(
    /from\s+["']three\/examples\/jsm\/([^"']+)["']/g,
    (_m, rest) => `from "./${path.basename(rest)}"`
  );
  code = code.replace(
    /from\s+["']\.\.\/(?:utils|libs)\/([^"']+)["']/g,
    (_m, rest) => `from "./${path.basename(rest)}"`
  );

  fs.writeFileSync(to, code);
  copied++;
  console.log(`[vendor] ${dest}`);
}

console.log(`[vendor] Done — ${copied}/${FILES.length} files in frontend/vendor/`);
