// One timeline for every time-domain detector: the Result tab's "play all" and
// "record all" step them together.
//
// When every detector recorded the same solver steps, the timeline is those
// frames. Otherwise it is the solve's own time grid, downsampled by a global
// stride n: solver steps 0, n, 2n, … Each detector shows the last frame it
// recorded at or before the step (sample-and-hold), and -1 — fully transparent
// — before its first sample. n = 1 is every step of the solve; the largest n
// that still lands on every recorded frame of every detector is
// `losslessStride`. A recording without the solve's clock (a foreign result)
// falls back to frame indices, each detector holding its last frame.
import type { FieldPayload } from "./wire";

export interface Timeline {
  //: the time-domain detectors on the timeline, in payload order
  ids: string[];
  //: each entry's time in seconds (or its index, for an untimed recording)
  times: number[];
  //: per detector, the frame it shows at each entry; -1 = not recorded yet
  frames: Map<string, Int32Array>;
  //: "shared" — the detectors' own common frames; "solve" — the solve's time
  //: grid, every `stride`-th step; "index" — frame indices of an untimed recording
  kind: "shared" | "solve" | "index";
  //: the solver steps between entries of a "solve" timeline (1 otherwise)
  stride: number;
}

/**
 * The common timeline of a result's time-domain detectors, or null if none.
 *
 * `stride` downsamples the solve's time grid when the detectors' recorded
 * steps differ (a "solve" timeline); it is ignored when they share a grid.
 */
export function buildTimeline(fields: Record<string, FieldPayload>, stride = 1): Timeline | null {
  const entries = timeEntries(fields);
  if (entries.length === 0) return null;
  const ids = entries.map(([id]) => id);
  const frames = new Map<string, Int32Array>();

  const grids = entries.map(([, field]) => field.steps ?? null);
  const first = grids[0];
  if (first && grids.every((grid) => grid !== null && sameNumbers(grid, first))) {
    // one shared grid: frame i of every detector is entry i
    for (const id of ids) frames.set(id, Int32Array.from(first, (_, i) => i));
    const times = entries[0][1].times ?? first.map((_, i) => i);
    return { ids, times: times.slice(0, first.length), frames, kind: "shared", stride: 1 };
  }

  const timed = entries.every(([, field]) =>
    field.steps && field.time_step !== undefined && field.steps_total !== undefined);
  if (timed) {
    // the solve's time grid, every n-th step; each detector holds its last
    // frame at or before each step
    const total = Math.max(...entries.map(([, field]) => field.steps_total ?? 0));
    const n = Math.min(Math.max(1, Math.round(stride) || 1), Math.max(1, total));
    const step = entries[0][1].time_step ?? 0;
    for (const [id, field] of entries) frames.set(id, holdFrames(field.steps ?? [], total, n));
    const count = Math.ceil(total / n);
    return {
      ids, times: Array.from({ length: count }, (_, i) => i * n * step), frames, kind: "solve", stride: n,
    };
  }

  // untimed: frame indices, each detector holding its last frame
  const counts = entries.map(([, field]) => (field.times ?? []).length);
  const length = Math.max(1, ...counts);
  entries.forEach(([id], d) => {
    const last = Math.max(0, counts[d] - 1);
    frames.set(id, Int32Array.from({ length }, (_, i) => Math.min(i, last)));
  });
  return { ids, times: Array.from({ length }, (_, i) => i), frames, kind: "index", stride: 1 };
}

/**
 * The frames a detector shows at solver steps 0, n, 2n, … below `total`: at
 * each, the last frame recorded at or before it (-1 before the first).
 */
export function holdFrames(steps: number[], total: number, stride = 1): Int32Array {
  const count = Math.ceil(total / stride);
  const out = new Int32Array(count).fill(-1);
  let next = 0; // the next recorded frame not yet reached
  for (let i = 0; i < count; i += 1) {
    const k = i * stride;
    while (next < steps.length && steps[next] <= k) next += 1;
    out[i] = next - 1;
  }
  return out;
}

/**
 * The largest global stride that still lands on every recorded step of every
 * time-domain detector (the greatest common divisor of their steps), so no
 * frame is skipped; 1 when there is none to divide.
 */
export function losslessStride(fields: Record<string, FieldPayload>): number {
  let divisor = 0;
  for (const [, field] of timeEntries(fields)) {
    for (const step of field.steps ?? []) divisor = gcd(divisor, step);
  }
  return Math.max(1, divisor);
}

function timeEntries(fields: Record<string, FieldPayload>): [string, FieldPayload][] {
  return Object.entries(fields).filter(([, field]) => field.frame_kind === "time");
}

function gcd(a: number, b: number): number {
  let x = Math.abs(Math.round(a));
  let y = Math.abs(Math.round(b));
  while (y !== 0) [x, y] = [y, x % y];
  return x;
}

function sameNumbers(a: number[], b: number[]): boolean {
  return a.length === b.length && a.every((value, i) => value === b[i]);
}
