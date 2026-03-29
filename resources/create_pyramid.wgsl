@group(0) @binding(0) var<storage, read> constants: array<u32>;
@group(0) @binding(1) var src: texture_2d<f32>;

@group(1) @binding(1) var src_sampler: sampler;
@group(1) @binding(2) var dst: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let dst_size = vec2f(textureDimensions(dst));
    if any(id.xy >= vec2u(dst_size)) { return; }

    // Sample center of each destination texel from the source mip
    let uv = (vec2f(id.xy) + 0.5) / dst_size;
    let color = textureSampleLevel(src, src_sampler, uv, 0.0);
    textureStore(dst, id.xy, color);
}
