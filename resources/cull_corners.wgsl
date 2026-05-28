// A replacement must be put at shader reading time (before compilation)
// as 'override' values are not yet supported as array sizes at the moment
const LOD_COUNT = 0x69u;
const CORNER_BLOCK_SZ = 0x69u;

struct FeatureBlock {
    width: u32,
    height: u32,
    values: array<u32, CORNER_BLOCK_SZ>,
};

@group(0) @binding(0) var<storage, read> src: array<FeatureBlock, LOD_COUNT>;
@group(0) @binding(1) var<storage, read_write> dst: array<FeatureBlock, LOD_COUNT>;

override SRC_IMAGE_W: u32;
override SRC_IMAGE_H: u32;
override WG_SIZE_X: u32;
override WG_SIZE_Y: u32;
override REGION_SIZE = 3u;

@compute @workgroup_size(WG_SIZE_X, WG_SIZE_Y)
fn main_horizontal(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x * REGION_SIZE;
    let y = gid.y;
    let lod = gid.z;

    if lod >= LOD_COUNT { return; }

    let src_block = &src[lod];
    let dst_block = &dst[lod];

    if gid.x == 0u && gid.y == 0u {
        writeBlockDimensions(lod);
    }

    let start = x * REGION_SIZE + y * src_block.width;
    let end = min(start + REGION_SIZE, src_block.width);

    if start >= src_block.width * src_block.height {
        return;
    }

    var maxIdx = start;
    var didFindMax = src_block.values[start] > 0;

    for (var x = start; x < end; x++) {
        if src_block.values[x] > src_block.values[maxIdx] {
            maxIdx = x;
            didFindMax = true;
        }
        dst_block.values[x] = 0u;
    }

    if didFindMax {
        dst_block.values[maxIdx] = src_block.values[maxIdx];
    }
}

// Likely dispatch size is SRC_W / 64, SRC_H / REGION_SIZE
@compute @workgroup_size(WG_SIZE_X, WG_SIZE_Y)
fn main_vertical(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y * REGION_SIZE;
    let lod = gid.z;

    if lod >= LOD_COUNT { return; }

    let src_block = &src[lod];
    let dst_block = &dst[lod];

    if gid.x == 0u && gid.y == 0u {
        writeBlockDimensions(lod);
    }

    let start = x * REGION_SIZE + y * src_block.width;
    if start >= src_block.width * src_block.height {
        return;
    }

    let step = src_block.width;
    let end = getVerticalEnd(start, src_block);

    var maxIdx = start;
    var didFindMax = src_block.values[start] > 0;

    for (var i = start; i < end; i += step) {
        if src_block.values[i] > src_block.values[maxIdx] {
            maxIdx = i;
            didFindMax = true;
        }
        // dst_block.values[i] = 0u;
        dst_block.values[i] = src_block.values[i];
    }

    if didFindMax {
        dst_block.values[maxIdx] = src_block.values[maxIdx];
    }
}

// Start must be less than full block size
fn getVerticalEnd(start: u32, block: ptr<storage, FeatureBlock>) -> u32 {
    let len = CORNER_BLOCK_SZ;
    let left = len - start;

    let step = block.width;
    let step_count = min(left / step, REGION_SIZE);

    return start + step * step_count;
}

fn writeBlockDimensions(lod: u32) {
    dst[lod].width = src[lod].width;
    dst[lod].height = src[lod].height;
}

fn getIdx(coord: vec3<u32>) -> u32 {
    var width = src[coord.z].width;
    var height = src[coord.z].height;
    return 0u;
}
