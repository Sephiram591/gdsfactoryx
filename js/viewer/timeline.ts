// One timeline for every time-domain detector: the Result tab's "play all" and
// "record all" step them together.
//
// When every detector recorded the same solver steps, the timeline is those
// frames. Otherwise it is **every step of the solve**: each detector shows the
// last frame it recorded at or before the step (sample-and-hold), and -1 —
// fully transparent — before its first sample. A recording without the solve's
// clock (a foreign result) falls back to frame indices, each detector holding
// its last frame.
import type { FieldPayload } from "./wire";

export interface Timeline {
  //: the time-domain detectors on the timeline, in payload order
  ids: string[];
  //: each entry's time in seconds (or its index, for an untimed recording)
  times: number[];
  //: per detector, the frame it shows at each entry; -1 = not recorded yet
  frames: Map<string, Int32Array>;
  //: "shared" — the detectors' own common frames; "solve" — every solver step;
  //: "index" — frame indices of an untimed recording
  kind: "shared" | "solve" | "index";
}

/** The common timeline of a result's time-domain detectors, or null if none. */
export function buildTimeline(fields: Record<string, FieldPayload>): Timeline | null {
  const entries = Object.entries(fields).filter(([, field]) => field.frame_kind === "time");
  if (entries.length === 0) return null;
  const ids = entries.map(([id]) => id);
  const frames = new Map<string, Int32Array>();

  const grids = entries.map(([, field]) => field.steps ?? null);
  const first = grids[0];
  if (first && grids.every((grid) => grid !== null && sameNumbers(grid, first))) {
    // one shared grid: frame i of every detector is entry i
    for (const id of ids) frames.set(id, Int32Array.from(first, (_, i) => i));
    const times = entries[0][1].times ?? first.map((_, i) => i);
    return { ids, times: times.slice(0, first.length), frames, kind: "shared" };
  }

  const timed = entries.every(([, field]) =>
    field.steps && field.time_step !== undefined && field.steps_total !== undefined);
  if (timed) {
    // every step of the solve; each detector holds its last frame at or before it
    const total = Math.max(...entries.map(([, field]) => field.steps_total ?? 0));
    const step = entries[0][1].time_step ?? 0;
    for (const [id, field] of entries) frames.set(id, holdFrames(field.steps ?? [], total));
    return { ids, times: Array.from({ length: total }, (_, k) => k * step), frames, kind: "solve" };
  }

  // untimed: frame indices, each detector holding its last frame
  const counts = entries.map(([, field]) => (field.times ?? []).length);
  const length = Math.max(1, ...counts);
  entries.forEach(([id], d) => {
    const last = Math.max(0, counts[d] - 1);
    frames.set(id, Int32Array.from({ length }, (_, i) => Math.min(i, last)));
  });
  return { ids, times: Array.from({ length }, (_, i) => i), frames, kind: "index" };
}

/** The frames a detector shows at each of `total` solver steps (-1 before its first). */
export function holdFrames(steps: number[], total: number): Int32Array {
  const out = new Int32Array(total).fill(-1);
  let next = 0; // the next recorded frame not yet reached
  for (let k = 0; k < total; k += 1) {
    while (next < steps.length && steps[next] <= k) next += 1;
    out[k] = next - 1;
  }
  return out;
}

function sameNumbers(a: number[], b: number[]): boolean {
  return a.length === b.length && a.every((value, i) => value === b[i]);
}
