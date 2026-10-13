// Axis tick helpers shared by the 3D and 2D viewports (spec R3, R4).
//
// The simulation region (the device box, `StackDevice.get_simulation_volume`)
// is outlined in both viewports and labelled with ticks on each axis. Steps are
// rounded to 1/2/5 x 10^n so the labels read cleanly.

//: Outline/tick color: light, to read against the dark viewport backgrounds.
export const REGION_COLOR = "#e2e2ec";

/** A tick step (1/2/5 x 10^n) giving roughly 8 ticks across `span`. */
export function niceStep(span: number): number {
  if (!(span > 0)) return 1;
  const raw = span / 8;
  const power = Math.pow(10, Math.floor(Math.log10(raw)));
  const normalized = raw / power;
  const step = normalized < 1.5 ? 1 : normalized < 3.5 ? 2 : normalized < 7.5 ? 5 : 10;
  return step * power;
}

/** The multiples of `step` in `[lo, hi]`, with -0 collapsed to 0. */
export function tickValues(lo: number, hi: number, step: number): number[] {
  const values: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) {
    values.push(Math.abs(v) < 1e-9 ? 0 : v);
  }
  return values;
}

/** A tick label with trailing zeros trimmed (e.g. 1.5, 0, -2). */
export function formatTick(value: number): string {
  return Number(value.toFixed(3)).toString();
}
