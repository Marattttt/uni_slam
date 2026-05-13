// A replacement must be put at shader reading time (before compilation)
// as 'override' values are not yet supported as array sizes at the moment
const LOD_COUNT = 0x69u;
const FEATURE_BLOCK_SZ = 0x69u;
override ORIENTATION_R: i32 = 3;

struct CornerBlock {
    width: u32,
    height: u32,
    values: array<u32, FEATURE_BLOCK_SZ>,
};

struct PointPair {
    ax: i32,
    ay: i32,
    bx: i32, 
    by: i32,
};

alias Descriptor = array<u32, 4>; // 256 bit string

struct Feature {
    coords: vec2<u32>,
    strength: u32,
    orientation: f32,
    descriptor: Descriptor,
};

struct FeatureArray {
    count: atomic<u32>,
    values: array<Feature>,
};

@group(0) @binding(0) var<storage, read> corners: array<CornerBlock, LOD_COUNT>;
@group(0) @binding(1) var<uniform> brief_tests: array<PointPair, 256>;
@group(0) @binding(2) var<storage, read_write> features: FeatureArray;

@group(1) @binding(0) var image: texture_2d<f32>;
@group(1) @binding(1) var samp: sampler; // linear filtering, clamp-to-edge

override WG_SIZE_X: u32;
override WG_SIZE_Y: u32;

@compute @workgroup_size(WG_SIZE_X, WG_SIZE_Y, 1) 
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    let lod = gid.z;

    if lod > LOD_COUNT { return; }

    let block = &corners[lod];

    if x >= block.width || y >= block.height { return; }

    let strength = block.values[block.width * y + x];
    
    if strength == 0 { return; }

    let orientation = getOrientation(gid.xy);

    let descriptor = computeDescriptor(vec2(i32(gid.x), i32(gid.y)), orientation);

    let feature = Feature(
        gid.xy,
        strength,
        orientation,
        descriptor,
    );

    let idx = atomicAdd(&features.count, 1u);
    features.values[idx] = feature;
}

fn getOrientation(coord: vec2<u32>) -> f32 {
    let r = ORIENTATION_R;
    var m10 = 0.0; // x-moment
    var m01 = 0.0; // y-moment
    let fx = i32(coord.x);
    let fy = i32(coord.y);

    for (var y = -r; y <= r; y = y + 1) {
        for (var x = -r; x <= r; x = x + 1) {
            // Circular mask
            if (x*x + y*y > r*r) {
                continue;
            }
            let pixel = textureLoad(image, vec2(fx + x, fy + y), 0).r;
            m10 += f32(x) * pixel;
            m01 += f32(y) * pixel;
        }
    }
    return atan2(m01, m10);
}

fn computeDescriptor(feature: vec2<i32>, orientation: f32) -> Descriptor {
    let c = cos(orientation);
    let s = sin(orientation);

    var desc = Descriptor();

    for (var i = 0u; i < 256u; i++) {
        let pair = brief_tests[i];

        // Rotating the points
        let ax = i32(round(f32(pair.ax) * c - f32(pair.ay) * s));
        let ay = i32(round(f32(pair.ax) * s + f32(pair.ay) * c));
        let bx = i32(round(f32(pair.bx) * c - f32(pair.by) * s));
        let by = i32(round(f32(pair.bx) * s + f32(pair.by) * c));

        let coords_a = vec2(
            u32(feature.x + ax),
            u32(feature.y + ay)
        );
        let coords_b = vec2(
            u32(feature.x + bx),
            u32(feature.y + by)
        ); 

        // let sample_a = textureLoad(image, coords_a, 0).r;
        // let sample_b = textureLoad(image, coords_b, 0).r;

        let sample_a = gaussianTap9(coords_a);
        let sample_b = gaussianTap9(coords_b);

        let test = select(0u, 1u, sample_a > sample_b);

        descriptorSetBit(&desc, test, i);
    }

    return desc;
}

fn descriptorSetBit(desc: ptr<function, Descriptor>, value: u32, bit: u32) {
    let segment = bit / 4;
    let offset = bit % 32;

    desc[segment] = desc[segment] | (value << offset);
}

// Returns pixel value as if it was smoothed with an r=9 gaussian blur
// delta ~= 1.5
// Should later need rewrite and use a separate smoothing pass
fn gaussianTap9(centerPx: vec2<u32>) -> f32 {
    let dims = vec2<f32>(textureDimensions(image, 0));
    let texel = 1.0 / dims;

    let centerUV = vec2(
        f32(centerPx.x) / dims.x,
        f32(centerPx.y) / dims.y
    );
    let uv = (centerUV + vec2<f32>(0.5)) * texel; // pixel-center to UV

    let offsets = array<f32, 3>(-1.2, 0.0, 1.2);
    let weights = array<f32, 3>(0.3125, 0.375, 0.3125);

    var sum: f32 = 0.0;
    for (var j = 0; j < 3; j = j + 1) {
        for (var i = 0; i < 3; i = i + 1) {
            let o = vec2<f32>(offsets[i], offsets[j]) * texel;
            let s = textureSampleLevel(image, samp, uv + o, 0.0).r;
            sum = sum + s * weights[i] * weights[j];
        }
    }
    return sum;
}

