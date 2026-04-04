struct Params {
    src_image_w: u32,
    src_image_h: u32,
    lod: u32,
};

struct CornersHeader {
    offset_per_lod: u32,
    corners: array<u32>,
};

@group(0) @binding(0) var image: texture_2d<f32>;
@group(0) @binding(1) var<uniform> params: Params;
@group(1) @binding(0) var<storage, read_write> corners: array<u32>;

override SCALE_FACTOR: f32;
override THRESHOLD: f32 = 0.3;
override N_SIMILLAR_MIN: u32 = 9;
override WORKGROUP_SIZE: u32 = 16;

// Bresenham circle offsets for radius 3 (16 pixels, clockwise from top)
const CIRCLE_X = array<i32, 16>(0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1);
const CIRCLE_Y = array<i32, 16>(-3, -3, -2, -1, 0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3);

@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total_threads = 256u * /* num_workgroups — match dispatch */ 256u;

    let w = textureDimensions(image).x;
    let h = textureDimensions(image).y;
    let x = gid.x;
    let y = gid.y;

    if x >= w || y >= h { return; }

    let response = getCornerStrength(gid.xy);

    let lod = params.lod;

    storeCornerResponse(response, gid.xy, lod);
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

    return max(9u, streak);
}

fn storeCornerResponse(response: u32, xy: vec2<u32>, lod: u32) {
    let w = textureDimensions(image).x;

    let lod_offset = lod * (params.src_image_w * params.src_image_h);
    let pixel_offset = xy.x + xy.y * w;

    let corner_idx: u32 = lod_offset + pixel_offset;

    corners[corner_idx] = response;
}
