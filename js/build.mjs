// Bundles the viewer frontend into the committed ES module anywidget loads
// (spec D6). The output goes straight into the Python package so flit ships it;
// `dist/` is gitignored, so there is no dist step.
import { build } from "esbuild";

await build({
  entryPoints: ["viewer/index.ts"],
  bundle: true,
  format: "esm",
  target: "es2020",
  platform: "browser",
  outfile: "../gdsfactory/viewer/static/viewer.js",
  sourcemap: false,
  logLevel: "info",
});

await build({
  entryPoints: ["viewer/style.css"],
  bundle: true,
  outfile: "../gdsfactory/viewer/static/viewer.css",
  logLevel: "info",
});
