// Frame-exact video encoding for a multi-frame detector's "save video".
//
// Each frame is rendered on demand, copied from the viewport's canvas and fed
// to the browser's WebCodecs `VideoEncoder`, so encoding runs faster than real
// time and never drops a frame (a `MediaRecorder` would record in real time and
// skip frames the renderer missed). H.264 is preferred — it is usually
// hardware-encoded, and the kernel only has to wrap it into an MP4 with ffmpeg,
// no re-encode. A browser without an H.264 encoder falls back to VP9, then VP8,
// which the kernel transcodes to H.264, so the saved file is an MP4 either way.

export type VideoCodec = "h264" | "vp9" | "vp8";

export interface EncodedVideo {
  //: the stream's codec: an Annex B H.264 stream, or VP8/VP9 frames in IVF
  codec: VideoCodec;
  bytes: Uint8Array;
  width: number;
  height: number;
  fps: number;
  frames: number;
}

interface Candidate {
  codec: string;
  kind: VideoCodec;
}

//: tried in order; H.264 High then Main (Annex B, so parameter sets ride in
//: the keyframes), then VP9 and VP8
const CANDIDATES: Candidate[] = [
  { codec: "avc1.640033", kind: "h264" },
  { codec: "avc1.4d0028", kind: "h264" },
  { codec: "vp09.00.40.08", kind: "vp9" },
  { codec: "vp8", kind: "vp8" },
];

/** Whether this browser can encode video at all (WebCodecs). */
export function videoSupported(): boolean {
  return typeof VideoEncoder !== "undefined" && typeof VideoFrame !== "undefined";
}

/**
 * Renders and encodes `count` frames into one video stream.
 *
 * `draw(i)` must leave frame `i` on the returned canvas (it is called once per
 * frame, in order); the canvas is cropped to even dimensions, as H.264 needs.
 * `onProgress(i, count)` is called after each frame is queued.
 *
 * Raises if no codec is supported or the encoder reports an error.
 */
export async function encodeFrames(
  count: number,
  fps: number,
  draw: (index: number) => HTMLCanvasElement,
  onProgress?: (index: number, count: number) => void,
): Promise<EncodedVideo> {
  const first = draw(0);
  const width = Math.max(2, first.width - (first.width % 2));
  const height = Math.max(2, first.height - (first.height % 2));
  // bits per second: ~0.2 bit per pixel per frame, clamped to a sane span
  const bitrate = Math.round(Math.min(40e6, Math.max(2e6, width * height * fps * 0.2)));
  const config = await chooseConfig(width, height, fps, bitrate);
  if (!config) throw new Error("this browser has no H.264, VP9 or VP8 video encoder");

  const chunks: Uint8Array[] = [];
  let failure: unknown = null;
  const encoder = new VideoEncoder({
    output: (chunk) => {
      const data = new Uint8Array(chunk.byteLength);
      chunk.copyTo(data);
      chunks.push(data);
    },
    error: (error) => {
      failure = error;
    },
  });
  encoder.configure(config.config);

  // a staging canvas of exactly the encoded size, so an odd-sized viewport is
  // cropped rather than scaled, and the frame is opaque
  const staging = document.createElement("canvas");
  staging.width = width;
  staging.height = height;
  const context = staging.getContext("2d");
  if (!context) throw new Error("no 2D canvas context for the video frames");
  const keyEvery = Math.max(1, Math.round(2 * fps));
  for (let i = 0; i < count; i += 1) {
    if (failure) break;
    const source = i === 0 ? first : draw(i);
    context.fillStyle = "#000000";
    context.fillRect(0, 0, width, height);
    context.drawImage(source, 0, 0, width, height, 0, 0, width, height);
    const frame = new VideoFrame(staging, {
      timestamp: Math.round((i * 1e6) / fps),
      duration: Math.round(1e6 / fps),
    });
    encoder.encode(frame, { keyFrame: i % keyEvery === 0 });
    frame.close();
    // backpressure: let the encoder drain before queuing more frames
    while (encoder.encodeQueueSize > 4 && !failure) {
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
    onProgress?.(i, count);
  }
  if (!failure) await encoder.flush();
  encoder.close();
  if (failure) throw failure;

  const bytes = config.kind === "h264"
    ? concat(chunks)
    : ivf(chunks, config.kind, width, height, fps);
  return { codec: config.kind, bytes, width, height, fps, frames: chunks.length };
}

/** The first supported encoder configuration, or null. */
async function chooseConfig(
  width: number,
  height: number,
  fps: number,
  bitrate: number,
): Promise<{ kind: VideoCodec; config: VideoEncoderConfig } | null> {
  for (const candidate of CANDIDATES) {
    const config: VideoEncoderConfig = {
      codec: candidate.codec,
      width,
      height,
      bitrate,
      framerate: fps,
    };
    if (candidate.kind === "h264") config.avc = { format: "annexb" };
    try {
      const support = await VideoEncoder.isConfigSupported(config);
      if (support.supported) return { kind: candidate.kind, config };
    } catch {
      // an unknown codec string throws in some browsers: try the next one
    }
  }
  return null;
}

/** Concatenated chunks: an Annex B H.264 elementary stream. */
function concat(chunks: Uint8Array[]): Uint8Array {
  const total = chunks.reduce((sum, chunk) => sum + chunk.byteLength, 0);
  const out = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return out;
}

/**
 * VP8/VP9 frames in an IVF container (a 32-byte file header, then a 12-byte
 * header per frame), the simplest file ffmpeg reads them from. The timebase is
 * 1/fps, so frame `i` has timestamp `i`.
 */
export function ivf(
  chunks: Uint8Array[],
  kind: "vp8" | "vp9",
  width: number,
  height: number,
  fps: number,
): Uint8Array {
  const total = 32 + chunks.reduce((sum, chunk) => sum + 12 + chunk.byteLength, 0);
  const out = new Uint8Array(total);
  const view = new DataView(out.buffer);
  out.set([0x44, 0x4b, 0x49, 0x46], 0); // "DKIF"
  view.setUint16(4, 0, true); // version
  view.setUint16(6, 32, true); // header size
  out.set(kind === "vp9" ? [0x56, 0x50, 0x39, 0x30] : [0x56, 0x50, 0x38, 0x30], 8); // "VP90"/"VP80"
  view.setUint16(12, width, true);
  view.setUint16(14, height, true);
  view.setUint32(16, Math.round(fps), true); // timebase denominator
  view.setUint32(20, 1, true); // timebase numerator
  view.setUint32(24, chunks.length, true);
  let offset = 32;
  chunks.forEach((chunk, index) => {
    view.setUint32(offset, chunk.byteLength, true);
    view.setBigUint64(offset + 4, BigInt(index), true);
    out.set(chunk, offset + 12);
    offset += 12 + chunk.byteLength;
  });
  return out;
}
