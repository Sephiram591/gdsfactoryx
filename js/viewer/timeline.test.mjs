// The shared timeline the Result tab's play / record all step (timeline.ts):
// the detectors' own frames when they share a time grid, else every step of
// the solve with each detector holding its last frame (-1 before its first).
import assert from "node:assert/strict";
import test from "node:test";

import { transform } from "esbuild";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const source = readFileSync(join(here, "timeline.ts"), "utf8").replace(/^import type .*?;$/ms, "");
const compiled = await transform(source, { loader: "ts", format: "esm" });
const timeline = await import(
  "data:text/javascript;base64," + Buffer.from(compiled.code).toString("base64")
);

const DT = 1e-17;

// a time-domain payload recorded at `steps` of a `total`-step solve
function recording(steps, total = 10) {
  return {
    frame_kind: "time", steps, time_step: DT, steps_total: total,
    times: steps.map((s) => s * DT),
  };
}

test("detectors on one grid play their own frames together", () => {
  const line = timeline.buildTimeline({ a: recording([0, 4, 8]), b: recording([0, 4, 8]) });
  assert.equal(line.kind, "shared");
  assert.deepEqual(line.ids, ["a", "b"]);
  assert.deepEqual(line.times, [0, 4 * DT, 8 * DT]);
  assert.deepEqual([...line.frames.get("a")], [0, 1, 2]);
  assert.deepEqual([...line.frames.get("b")], [0, 1, 2]);
});

test("different grids play every step of the solve, each holding its last frame", () => {
  // a records every 4th step from 0; b every 3rd from step 2 (a later start)
  const line = timeline.buildTimeline({ a: recording([0, 4, 8]), b: recording([2, 5, 8]) });
  assert.equal(line.kind, "solve");
  assert.equal(line.times.length, 10, "one entry per solver step");
  assert.equal(line.times[3], 3 * DT);
  assert.deepEqual([...line.frames.get("a")], [0, 0, 0, 0, 1, 1, 1, 1, 2, 2]);
  // b has not recorded before step 2: transparent (-1)
  assert.deepEqual([...line.frames.get("b")], [-1, -1, 0, 0, 0, 1, 1, 1, 2, 2]);
});

test("only time-domain detectors are on the timeline", () => {
  const phasor = { frame_kind: "freq", freqs: [1, 2] };
  assert.equal(timeline.buildTimeline({ p: phasor }), null);
  const line = timeline.buildTimeline({ p: phasor, a: recording([0, 1]) });
  assert.deepEqual(line.ids, ["a"]);
});

test("an untimed recording falls back to frame indices, holding the last frame", () => {
  const line = timeline.buildTimeline({
    a: { frame_kind: "time", times: [0, 1, 2] },
    b: { frame_kind: "time", times: [0] },
  });
  assert.equal(line.kind, "index");
  assert.deepEqual([...line.frames.get("a")], [0, 1, 2]);
  assert.deepEqual([...line.frames.get("b")], [0, 0, 0]);
});

test("holdFrames: the last frame at or before each step", () => {
  assert.deepEqual([...timeline.holdFrames([1, 3], 5)], [-1, 0, 0, 1, 1]);
  assert.deepEqual([...timeline.holdFrames([], 3)], [-1, -1, -1]);
  // every 2nd solver step: 0, 2, 4
  assert.deepEqual([...timeline.holdFrames([1, 3], 5, 2)], [-1, 0, 1]);
});

test("a global downsampling steps every n-th solver step, each at its last frame", () => {
  // a records steps 0, 4, 8; b 2, 5, 8; every 3rd of 10 solver steps: 0, 3, 6, 9
  const line = timeline.buildTimeline({ a: recording([0, 4, 8]), b: recording([2, 5, 8]) }, 3);
  assert.equal(line.kind, "solve");
  assert.equal(line.stride, 3);
  assert.deepEqual(line.times, [0, 3 * DT, 6 * DT, 9 * DT]);
  assert.deepEqual([...line.frames.get("a")], [0, 0, 1, 2]);
  // b is transparent at step 0, then its closest frame at or before each step
  assert.deepEqual([...line.frames.get("b")], [-1, 0, 1, 2]);
});

test("the global downsampling only applies when the recorded steps differ", () => {
  const line = timeline.buildTimeline({ a: recording([0, 4, 8]), b: recording([0, 4, 8]) }, 3);
  assert.equal(line.kind, "shared");
  assert.equal(line.stride, 1);
  assert.equal(line.times.length, 3);
});

test("the global downsampling is clamped to 1..steps", () => {
  const fields = { a: recording([0, 4, 8]), b: recording([2, 5, 8]) };
  assert.equal(timeline.buildTimeline(fields, 0).stride, 1);
  const coarse = timeline.buildTimeline(fields, 50);
  assert.equal(coarse.stride, 10);
  assert.deepEqual([...coarse.frames.get("a")], [0]);
});

test("losslessStride: the largest stride that lands on every recorded step", () => {
  const every20 = recording([0, 20, 40, 60], 100);
  const every50 = recording([0, 50], 100);
  assert.equal(timeline.losslessStride({ a: every20, b: every50 }), 10);
  // an offset start counts too
  assert.equal(timeline.losslessStride({ a: every20, b: recording([5, 55], 100) }), 5);
  assert.equal(timeline.losslessStride({}), 1);
  assert.equal(timeline.losslessStride({ a: recording([0], 10) }), 1);
});
