// The video export's container bytes. The encoder itself needs a browser
// (WebCodecs); the IVF wrapper a VP8/VP9 stream travels to the kernel in is
// pure, so its layout is checked here against the IVF spec ffmpeg reads.
import assert from "node:assert/strict";
import test from "node:test";

import { transform } from "esbuild";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const compiled = await transform(readFileSync(join(here, "video.ts"), "utf8"), {
  loader: "ts",
  format: "esm",
});
const video = await import(
  "data:text/javascript;base64," + Buffer.from(compiled.code).toString("base64")
);

test("ivf: a 32-byte file header, then a 12-byte header per frame", () => {
  const frames = [new Uint8Array([1, 2, 3]), new Uint8Array([4, 5])];
  const bytes = video.ivf(frames, "vp9", 640, 360, 20);
  const view = new DataView(bytes.buffer);
  assert.equal(String.fromCharCode(...bytes.slice(0, 4)), "DKIF");
  assert.equal(view.getUint16(4, true), 0, "version");
  assert.equal(view.getUint16(6, true), 32, "header size");
  assert.equal(String.fromCharCode(...bytes.slice(8, 12)), "VP90");
  assert.equal(view.getUint16(12, true), 640);
  assert.equal(view.getUint16(14, true), 360);
  assert.equal(view.getUint32(16, true), 20, "timebase denominator = fps");
  assert.equal(view.getUint32(20, true), 1, "timebase numerator");
  assert.equal(view.getUint32(24, true), 2, "frame count");
  // frame 0: size 3, timestamp 0, then its bytes
  assert.equal(view.getUint32(32, true), 3);
  assert.equal(view.getBigUint64(36, true), 0n);
  assert.deepEqual([...bytes.slice(44, 47)], [1, 2, 3]);
  // frame 1: size 2, timestamp 1
  assert.equal(view.getUint32(47, true), 2);
  assert.equal(view.getBigUint64(51, true), 1n);
  assert.deepEqual([...bytes.slice(59, 61)], [4, 5]);
  assert.equal(bytes.byteLength, 61);
});

test("ivf: VP8 streams carry the VP80 fourcc", () => {
  const bytes = video.ivf([new Uint8Array([9])], "vp8", 2, 2, 30);
  assert.equal(String.fromCharCode(...bytes.slice(8, 12)), "VP80");
});

test("no WebCodecs in node: the export reports itself unsupported", () => {
  assert.equal(video.videoSupported(), typeof VideoEncoder !== "undefined");
});
