struct Constants {
    frame_width:  u32,
    frame_height: u32,
};

struct Pyramid {
    layer_count: u32,
    bwdata:      array<u32>,
};

@group(0) @binding(0) var<storage, read>       constants:     Constants;
@group(0) @binding(1) var<storage, read>       image_pyramid: Pyramid;
@group(1) @binding(0) var<storage, read_write> corners:       array<u32>;

const THRESHOLD: u32 = 80u;

// Bresenham circle offsets for radius 3 (16 pixels, clockwise from top)
const CIRCLE_X = array<i32, 16>( 0,  1,  2,  3,  3,  3,  2,  1,  0, -1, -2, -3, -3, -3, -2, -1);
const CIRCLE_Y = array<i32, 16>(-3, -3, -2, -1,  0,  1,  2,  3,  3,  3,  2,  1,  0, -1, -2, -3);

// Extract an 8-bit grayscale pixel from the packed u32 array.
// 4 pixels per u32, packed as [p0 | p1 | p2 | p3] in the low→high bytes.
fn sample(x: u32, y: u32) -> u32 {
    let idx        = y * constants.frame_width + x;
    let word_idx   = idx >> 2u;                      // which u32
    let byte_shift = (idx & 3u) << 3u;               // which byte within the u32 (0,8,16,24)
    return (image_pyramid.bwdata[word_idx] >> byte_shift) & 0xFFu;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total_threads = 256u * /* num_workgroups — match dispatch */ 256u;
    var pixel_idx = gid.x;

    let w = constants.frame_width;
    let h = constants.frame_height;

    while (pixel_idx < w * h) {
        let x = pixel_idx % w;
        let y = pixel_idx / w;

        // Requirement 1: bail if within 3px of any edge
        if (x <= 3u || y <= 3u || x + 3u >= w || y + 3u >= h) {
            corners[pixel_idx] = 0u;
            pixel_idx += total_threads;
            continue;
        }

        // Requirement 2: sum absolute differences around the Bresenham circle
        let center = sample(x, y);
        var sad: u32 = 0u;

        for (var i = 0; i < 16; i++) {
            let cx = u32(i32(x) + CIRCLE_X[i]);
            let cy = u32(i32(y) + CIRCLE_Y[i]);
            let s  = sample(cx, cy);
            sad   += select(center - s, s - center, s >= center); // abs diff
        }

        // Requirement 3: write 1 if SAD exceeds threshold, else 0
        corners[pixel_idx] = select(0u, 1u, sad > THRESHOLD);

        pixel_idx += total_threads;
    }
}
