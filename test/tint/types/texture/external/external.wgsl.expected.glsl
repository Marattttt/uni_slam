#version 310 es


struct tint_GammaTransferParams {
  float G;
  float A;
  float B;
  float C;
  float D;
  float E;
  float F;
  uint padding;
};

struct tint_ExternalTextureParams {
  uint numPlanes;
  uint doYuvToRgbConversionOnly;
  mat3x4 yuvToRgbConversionMatrix;
  tint_GammaTransferParams gammaDecodeParams;
  tint_GammaTransferParams gammaEncodeParams;
  mat3 gamutConversionMatrix;
  mat3x2 sampleTransform;
  mat3x2 loadTransform;
  vec2 samplePlane0RectMin;
  vec2 samplePlane0RectMax;
  vec2 samplePlane1RectMin;
  vec2 samplePlane1RectMax;
  uvec2 apparentSize;
  vec2 plane1CoordFactor;
};

layout(binding = 3, std140)
uniform t_f_params_block_1_ubo {
  uvec4 inner[17];
} v_1;
layout(binding = 0, std430)
buffer out_block_1_ssbo {
  vec4 inner;
} v_2;
uniform highp sampler2D t_f_plane0;
uniform highp sampler2D t_f_plane1;
vec3 tint_GammaCorrection(vec3 v, tint_GammaTransferParams params) {
  vec3 v_3 = vec3(params.G);
  return mix((sign(v) * (pow(((params.A * abs(v)) + params.B), v_3) + params.E)), (sign(v) * ((params.C * abs(v)) + params.F)), lessThan(abs(v), vec3(params.D)));
}
vec4 tint_TextureLoadMultiplanarExternal(tint_ExternalTextureParams params, uvec2 coords) {
  vec2 v_4 = round((params.loadTransform * vec3(vec2(min(coords, params.apparentSize)), 1.0f)));
  uvec2 v_5 = uvec2(v_4);
  vec3 v_6 = vec3(0.0f);
  float v_7 = 0.0f;
  if ((params.numPlanes == 1u)) {
    ivec2 v_8 = ivec2(v_5);
    vec4 v_9 = texelFetch(t_f_plane0, v_8, int(0u));
    v_6 = v_9.xyz;
    v_7 = v_9.w;
  } else {
    ivec2 v_10 = ivec2(v_5);
    float v_11 = texelFetch(t_f_plane0, v_10, int(0u)).x;
    ivec2 v_12 = ivec2(uvec2((v_4 * params.plane1CoordFactor)));
    v_6 = (vec4(v_11, texelFetch(t_f_plane1, v_12, int(0u)).xy, 1.0f) * params.yuvToRgbConversionMatrix);
    v_7 = 1.0f;
  }
  vec3 v_13 = v_6;
  vec3 v_14 = vec3(0.0f);
  if ((params.doYuvToRgbConversionOnly == 0u)) {
    v_14 = tint_GammaCorrection((params.gamutConversionMatrix * tint_GammaCorrection(v_13, params.gammaDecodeParams)), params.gammaEncodeParams);
  } else {
    v_14 = v_13;
  }
  return vec4(v_14, v_7);
}
mat3x2 v_15(uint start_byte_offset) {
  uvec4 v_16 = v_1.inner[(start_byte_offset / 16u)];
  vec2 v_17 = uintBitsToFloat(mix(v_16.xy, v_16.zw, bvec2((((start_byte_offset & 15u) >> 2u) == 2u))));
  uvec4 v_18 = v_1.inner[((8u + start_byte_offset) / 16u)];
  vec2 v_19 = uintBitsToFloat(mix(v_18.xy, v_18.zw, bvec2(((((8u + start_byte_offset) & 15u) >> 2u) == 2u))));
  uvec4 v_20 = v_1.inner[((16u + start_byte_offset) / 16u)];
  return mat3x2(v_17, v_19, uintBitsToFloat(mix(v_20.xy, v_20.zw, bvec2(((((16u + start_byte_offset) & 15u) >> 2u) == 2u)))));
}
mat3 v_21(uint start_byte_offset) {
  return mat3(uintBitsToFloat(v_1.inner[(start_byte_offset / 16u)].xyz), uintBitsToFloat(v_1.inner[((16u + start_byte_offset) / 16u)].xyz), uintBitsToFloat(v_1.inner[((32u + start_byte_offset) / 16u)].xyz));
}
tint_GammaTransferParams v_22(uint start_byte_offset) {
  uvec4 v_23 = v_1.inner[(start_byte_offset / 16u)];
  uvec4 v_24 = v_1.inner[((4u + start_byte_offset) / 16u)];
  uvec4 v_25 = v_1.inner[((8u + start_byte_offset) / 16u)];
  uvec4 v_26 = v_1.inner[((12u + start_byte_offset) / 16u)];
  uvec4 v_27 = v_1.inner[((16u + start_byte_offset) / 16u)];
  uvec4 v_28 = v_1.inner[((20u + start_byte_offset) / 16u)];
  uvec4 v_29 = v_1.inner[((24u + start_byte_offset) / 16u)];
  uvec4 v_30 = v_1.inner[((28u + start_byte_offset) / 16u)];
  return tint_GammaTransferParams(uintBitsToFloat(v_23[((start_byte_offset & 15u) >> 2u)]), uintBitsToFloat(v_24[(((4u + start_byte_offset) & 15u) >> 2u)]), uintBitsToFloat(v_25[(((8u + start_byte_offset) & 15u) >> 2u)]), uintBitsToFloat(v_26[(((12u + start_byte_offset) & 15u) >> 2u)]), uintBitsToFloat(v_27[(((16u + start_byte_offset) & 15u) >> 2u)]), uintBitsToFloat(v_28[(((20u + start_byte_offset) & 15u) >> 2u)]), uintBitsToFloat(v_29[(((24u + start_byte_offset) & 15u) >> 2u)]), v_30[(((28u + start_byte_offset) & 15u) >> 2u)]);
}
mat3x4 v_31(uint start_byte_offset) {
  return mat3x4(uintBitsToFloat(v_1.inner[(start_byte_offset / 16u)]), uintBitsToFloat(v_1.inner[((16u + start_byte_offset) / 16u)]), uintBitsToFloat(v_1.inner[((32u + start_byte_offset) / 16u)]));
}
tint_ExternalTextureParams v_32(uint start_byte_offset) {
  uvec4 v_33 = v_1.inner[(start_byte_offset / 16u)];
  uvec4 v_34 = v_1.inner[((4u + start_byte_offset) / 16u)];
  mat3x4 v_35 = v_31((16u + start_byte_offset));
  tint_GammaTransferParams v_36 = v_22((64u + start_byte_offset));
  tint_GammaTransferParams v_37 = v_22((96u + start_byte_offset));
  mat3 v_38 = v_21((128u + start_byte_offset));
  mat3x2 v_39 = v_15((176u + start_byte_offset));
  mat3x2 v_40 = v_15((200u + start_byte_offset));
  uvec4 v_41 = v_1.inner[((224u + start_byte_offset) / 16u)];
  vec2 v_42 = uintBitsToFloat(mix(v_41.xy, v_41.zw, bvec2(((((224u + start_byte_offset) & 15u) >> 2u) == 2u))));
  uvec4 v_43 = v_1.inner[((232u + start_byte_offset) / 16u)];
  vec2 v_44 = uintBitsToFloat(mix(v_43.xy, v_43.zw, bvec2(((((232u + start_byte_offset) & 15u) >> 2u) == 2u))));
  uvec4 v_45 = v_1.inner[((240u + start_byte_offset) / 16u)];
  vec2 v_46 = uintBitsToFloat(mix(v_45.xy, v_45.zw, bvec2(((((240u + start_byte_offset) & 15u) >> 2u) == 2u))));
  uvec4 v_47 = v_1.inner[((248u + start_byte_offset) / 16u)];
  vec2 v_48 = uintBitsToFloat(mix(v_47.xy, v_47.zw, bvec2(((((248u + start_byte_offset) & 15u) >> 2u) == 2u))));
  uvec4 v_49 = v_1.inner[((256u + start_byte_offset) / 16u)];
  uvec2 v_50 = mix(v_49.xy, v_49.zw, bvec2(((((256u + start_byte_offset) & 15u) >> 2u) == 2u)));
  uvec4 v_51 = v_1.inner[((264u + start_byte_offset) / 16u)];
  return tint_ExternalTextureParams(v_33[((start_byte_offset & 15u) >> 2u)], v_34[(((4u + start_byte_offset) & 15u) >> 2u)], v_35, v_36, v_37, v_38, v_39, v_40, v_42, v_44, v_46, v_48, v_50, uintBitsToFloat(mix(v_51.xy, v_51.zw, bvec2(((((264u + start_byte_offset) & 15u) >> 2u) == 2u)))));
}
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
  tint_ExternalTextureParams v_52 = v_32(0u);
  v_2.inner = tint_TextureLoadMultiplanarExternal(v_52, min(uvec2(ivec2(0)), ((v_52.apparentSize + uvec2(1u)) - uvec2(1u))));
}
