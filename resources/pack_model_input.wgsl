// Packs the previous and current frame into the RaCo-ALIKED-LightGlue input
// tensor: float32 [2, 3, OUT_HEIGHT, OUT_WIDTH], NCHW, RGB in [0, 1].
// Batch 0 is the left (previous) image and batch 1 the right (current) one.
// The frames are grayscale, so the gray value goes to all three channels.
// The output is a centre crop of the source level starting at
// (CROP_X, CROP_Y)

@group(0) @binding(0) var left_src: texture_2d<f32>;
@group(0) @binding(1) var right_src: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> images: array<f32>;

override OUT_WIDTH: u32;
override OUT_HEIGHT: u32;
override CROP_X: u32;
override CROP_Y: u32;

const CHANNELS: u32 = 3u;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= OUT_WIDTH || id.y >= OUT_HEIGHT) { return; }

    let src = vec2u(id.x + CROP_X, id.y + CROP_Y);
    let left = textureLoad(left_src, src, 0).r;
    let right = textureLoad(right_src, src, 0).r;

    let plane = OUT_WIDTH * OUT_HEIGHT;
    let pixel = id.y * OUT_WIDTH + id.x;
    let right_base = CHANNELS * plane;

    for (var channel = 0u; channel < CHANNELS; channel++) {
        images[channel * plane + pixel] = left;
        images[right_base + channel * plane + pixel] = right;
    }
}
