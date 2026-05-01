const WORKFLOW_HORIZONTAL = 1;
const WORKFLOW_VERTICAL = 2;
const WORKFLOW_SCALE_SPACE = 3;

@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dest: array<u32>;

override SRC_IMAGE_W: u32;
override SRC_IMAGE_H: u32;
override SCALE_FACTOR: f32;

override REGION_SIZE = 3u;

@compute @workgroup_size(4, 4, 4) 
fn main_horizontal(@builtin(global_invocation_id) gid: vec3<u32>) {
    let lod = gid.z;

    if lod > get_max_lod() { return; }

    if gid.x == 0 && gid.y == 0 {
        writeLodDimensions(lod);
    }

    let lodStart = getLodStart(lod, SRC_IMAGE_W, SRC_IMAGE_H);
    let w = src[lodStart];

    let xStart = lodStart + 2 + gid.x * REGION_SIZE;
    let xEnd = min(w, xStart + REGION_SIZE);

    let step = step_horizontal();

    var maxIdx = xStart;
    var hasMaxIdx = false;

    for (var x = xStart; x < xEnd; x += step) {
        if src[x] > src[maxIdx] {
            maxIdx = x;
            hasMaxIdx = true;
        }
        dest[x] = 0;
    }

    dest[maxIdx] = select(0, src[maxIdx], hasMaxIdx);
}

fn step_horizontal() -> u32 {
    return 1;
}

@compute @workgroup_size(4, 4, 4)
fn main_vertical(@builtin(global_invocation_id) gid: vec3<u32>) {
    let lod = gid.z;
    if lod > get_max_lod() { return; }
    if gid.x == 0 && gid.y == 0 { writeLodDimensions(lod); }

    let lodStart = getLodStart(lod, SRC_IMAGE_W, SRC_IMAGE_H);

    let dims = getLodDimanesions(lod);

    let totalSteps = REGION_SIZE;
    let step = step_vertical(lod);

    let yStart = lodStart + 2 + gid.y * dims.x;
    let yEnd = min(dims.y, yStart + step * totalSteps);

    var maxIdx = yStart;
    var hasMaxIdx = false;
    for (var i = yStart; i < yEnd; i += step) {
        if src[i] > src[maxIdx] {
            maxIdx = i;
            hasMaxIdx = true;
        }

        dest[i] = src[i];
    }

    dest[maxIdx] = select(0, src[maxIdx], hasMaxIdx);
}

fn step_vertical(lod: u32) -> u32 {
    let start = getLodStart(lod, SRC_IMAGE_W, SRC_IMAGE_H);
    let w = src[start];
    return w;
}

// GENERAL UTIL FUNCTIONS

fn get_max_lod() -> u32 {
    let lod_size: u32 = SRC_IMAGE_H * SRC_IMAGE_W;
    let total: u32 = arrayLength(&src);
    return u32(total / lod_size);
}

fn getLodStart(lod: u32, w: u32, h: u32) -> u32 {
    return lod * w * h;
}

fn getLodDimanesions(lod: u32) -> vec2<u32> {
    let start = getLodStart(lod, SRC_IMAGE_W, SRC_IMAGE_H);

    return vec2(src[start], src[start + 1]);
}

fn writeLodDimensions(lod: u32) {
    let start = SRC_IMAGE_W * SRC_IMAGE_H * lod;
    dest[start] = src[start];
    dest[start + 1] = src[start + 1];
}
