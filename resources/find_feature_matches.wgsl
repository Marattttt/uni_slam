// Value to be replaced
const LOD_COUNT = 0x69u;
const FEATURE_BLOCK_SZ = 0x69u;

const DESCRIPTOR_SIZE = 8;
alias Descriptor = array<u32, 8>:

struct Feature {
	coords: vec2<f32>,
	strength: u32,
	orientation: f32,
	descriptor: Descriptor,
};

struct FeatureBlock {
    count: atomic<u32>,
    values: array<Feature, FEATURE_BLOCK_SZ>,
};

alias FeatureArray = array<FeatureBlock, LOD_COUNT>;

struct FeatureMatch {
	match: Feature,
	against: Feature,
};

struct FeatureMatchArray {
	count: atomic<u32>,
	values: array<FeatureMatch>,
};

override  WG_SIZE_X: u32;
override  WG_SIZE_Y: u32;

@group(0) @binding(0) var<storage, read> features: FeatureArray;
@group(0) @binding(1) var<storage, read_write> matches: FeatureMatchArray;

@compute @workgroup_size(WG_SIZE_X, WG_SIZE_Y) 
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
}

fn hammingDistance(a: Descriptor, b: Descriptor) {
	var sum = 0u;
	for (var i = 0u; i < DESCRIPTOR_SIZE; i++) {
		sum += countOneBits(a.values[i] ^ b.values[i]);
	};
	return sum;
}
