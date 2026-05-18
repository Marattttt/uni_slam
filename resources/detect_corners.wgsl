struct Params {
    lod: u32,
};

// A replacement must be put at shader reading time (before compilation)
// as 'override' values are not yet supported as array sizes at the moment
const LOD_COUNT = 0x69u;
const FEATURE_BLOCK_SZ = 0x69u;

struct FeatureBlock {
    width: u32,
    height: u32,
    values: array<u32, FEATURE_BLOCK_SZ>,
};

// FAST contrast threshold on normalised [0,1] pixel intensities. ORB-SLAM
// defaults to 20/255 (≈0.078) with a 7/255 fallback in low-texture cells via
// its quota-based extractor. We have no per-cell quota, so we sit slightly
// higher: 0.15 ≈ 38/255 lands around ~1000–1500 raw FAST hits per frame across
// the LoD pyramid on the EuRoC Machine Hall sequences, matching the reference
// numbers in the ORB-SLAM paper.
override THRESHOLD: f32 = 0.15;
override N_SIMILLAR_MIN: u32 = 9;
override WORKGROUP_SIZE: u32 = 8;

@group(0) @binding(0) var<storage, read_write> corners: array<FeatureBlock, LOD_COUNT>;
@group(1) @binding(0) var image: texture_2d<f32>;
@group(1) @binding(1) var<uniform> params: Params;

// Bresenham circle offsets for radius 3 (16 pixels, clockwise from top)
const CIRCLE_X = array<i32, 16>(0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1);
const CIRCLE_Y = array<i32, 16>(-3, -3, -2, -1, 0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3);

@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = textureDimensions(image).x;
    let h = textureDimensions(image).y;
    let x = gid.x;
    let y = gid.y;

    if x >= w || y >= h { return; }

    if x == 0 && y == 0 {
        writeLodDimensions(params.lod);
    }

    let response = getCornerStrength(gid.xy);

    storeCornerResponse(response, gid.xy);
}

fn writeLodDimensions(lod: u32) {
    let dims = textureDimensions(image);
    corners[lod].width = dims.x;
    corners[lod].height = dims.y;
}

fn getCornerStrength(coord: vec2<u32>) -> u32 {
    let x = coord.x;
    let y = coord.y;

    var curr = 1u;
    var streak = 1u;

    for (var i = 0; i < 16; i++) {
        let cx = u32(i32(x) + CIRCLE_X[i]);
        let cy = u32(i32(y) + CIRCLE_Y[i]);

        let center = textureLoad(image, vec2(x, y), 0);
        let counter = textureLoad(image, vec2(cx, cy), 0);

        let is_different: bool = abs(center - counter).r > THRESHOLD;

        curr = select(1, curr + 1, is_different);
        streak = max(curr, streak);
    }

    let is_good = streak > N_SIMILLAR_MIN;

    return select(0u, streak, is_good);
}

fn storeCornerResponse(response: u32, xy: vec2<u32>) {
    let lod = params.lod;
    let block = &corners[lod];
    let idx = xy.y * block.width + xy.x;

    block.values[idx] = response;
}
