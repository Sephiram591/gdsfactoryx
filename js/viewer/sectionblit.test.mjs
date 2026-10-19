// The 2D field blit's raster orientation (spec R15).
//
// The field raster is row-major in (v, u), so raster row 0 is the plane's
// low-v edge. `drawImage` does NOT flip on a negative height — Chromium
// normalises the rectangle — so the blit flips with a transform instead. This
// pins that mapping: applying the transform to a raster row must land the low-v
// row at the low-v screen edge (the larger y), not the other way round.
import assert from "node:assert/strict";
import test from "node:test";

import { build } from "esbuild";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
// bundle the section module so its local imports (fieldramp, ticks) resolve,
// then import the bundle to reach the exported pure helper
const outfile = join(mkdtempSync(join(tmpdir(), "sectionblit-")), "section.mjs");
writeFileSync(
  outfile,
  (
    await build({
      entryPoints: [join(here, "section.ts")],
      bundle: true,
      format: "esm",
      write: false,
      external: ["three"],
    })
  ).outputFiles[0].text,
);
const { fieldBlit } = await import(outfile);

test("the field raster's row 0 lands at the low-v screen edge", () => {
  // a rect on screen: x0=10, x1=90 (u across), y0=20 (high v), y1=80 (low v)
  const blit = fieldBlit(10, 80, 90, 20);

  // raster row 0 sits at image y = 0; the transform maps it to
  // translate.y + 0 * scale.y = y1, the low-v edge
  const rasterRow0Y = blit.translate[1] + 0 * blit.scale[1];
  const rasterRowLastY = blit.translate[1] + blit.dest[3] * blit.scale[1];

  assert.equal(rasterRow0Y, 80, "raster row 0 must land at the low-v edge (y1)");
  assert.equal(rasterRowLastY, 20, "the last raster row must land at the high-v edge (y0)");
  assert.ok(rasterRow0Y > rasterRowLastY, "the raster must run upward as v increases");
});

test("the flip is a transform, not a negative height", () => {
  const blit = fieldBlit(10, 80, 90, 20);
  assert.ok(blit.dest[3] > 0, "the destination height must be positive");
  assert.equal(blit.scale[1], -1, "the y flip is carried by the transform");
  assert.deepEqual(blit.translate, [10, 80], "the transform starts at the low-v corner");
  assert.deepEqual(blit.dest, [0, 0, 80, 60], "the raster fills the rect");
});
