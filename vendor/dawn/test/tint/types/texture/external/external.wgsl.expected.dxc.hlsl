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
  float3x4 yuvToRgbConversionMatrix;
  tint_GammaTransferParams gammaDecodeParams;
  tint_GammaTransferParams gammaEncodeParams;
  float3x3 gamutConversionMatrix;
  float3x2 sampleTransform;
  float3x2 loadTransform;
  float2 samplePlane0RectMin;
  float2 samplePlane0RectMax;
  float2 samplePlane1RectMin;
  float2 samplePlane1RectMax;
  uint2 apparentSize;
  float2 plane1CoordFactor;
};


cbuffer cbuffer_t_f_params : register(b2) {
  uint4 t_f_params[17];
};
Texture2D<float4> t_f_plane0 : register(t0);
Texture2D<float4> t_f_plane1 : register(t1);
RWByteAddressBuffer v_1 : register(u0, space1);
uint2 tint_v2f32_to_v2u32(float2 value) {
  return uint2(clamp(value, (0.0f).xx, (4294967040.0f).xx));
}

float3 tint_GammaCorrection(float3 v, tint_GammaTransferParams params) {
  float3 v_2 = float3((params.G).xxx);
  float3 v_3 = float3((params.D).xxx);
  float3 v_4 = float3(sign(v));
  return (((abs(v) < v_3)) ? ((v_4 * ((params.C * abs(v)) + params.F))) : ((v_4 * (pow(((params.A * abs(v)) + params.B), v_2) + params.E))));
}

float4 tint_TextureLoadMultiplanarExternal(Texture2D<float4> plane_0, Texture2D<float4> plane_1, tint_ExternalTextureParams params, uint2 coords) {
  float2 v_5 = round(mul(float3(float2(min(coords, params.apparentSize)), 1.0f), params.loadTransform));
  uint2 v_6 = tint_v2f32_to_v2u32(v_5);
  float3 v_7 = (0.0f).xxx;
  float v_8 = 0.0f;
  if ((params.numPlanes == 1u)) {
    int2 v_9 = int2(v_6);
    float4 v_10 = plane_0.Load(int3(v_9, int(0u)));
    v_7 = v_10.xyz;
    v_8 = v_10.w;
  } else {
    int2 v_11 = int2(v_6);
    float v_12 = plane_0.Load(int3(v_11, int(0u))).x;
    int2 v_13 = int2(tint_v2f32_to_v2u32((v_5 * params.plane1CoordFactor)));
    v_7 = mul(params.yuvToRgbConversionMatrix, float4(v_12, plane_1.Load(int3(v_13, int(0u))).xy, 1.0f));
    v_8 = 1.0f;
  }
  float3 v_14 = v_7;
  float3 v_15 = (0.0f).xxx;
  if ((params.doYuvToRgbConversionOnly == 0u)) {
    tint_GammaTransferParams v_16 = params.gammaDecodeParams;
    tint_GammaTransferParams v_17 = params.gammaEncodeParams;
    v_15 = tint_GammaCorrection(mul(tint_GammaCorrection(v_14, v_16), params.gamutConversionMatrix), v_17);
  } else {
    v_15 = v_14;
  }
  return float4(v_15, v_8);
}

float3x2 v_18(uint start_byte_offset) {
  uint4 v_19 = t_f_params[(start_byte_offset / 16u)];
  float2 v_20 = asfloat((((((start_byte_offset & 15u) >> 2u) == 2u)) ? (v_19.zw) : (v_19.xy)));
  uint4 v_21 = t_f_params[((8u + start_byte_offset) / 16u)];
  float2 v_22 = asfloat(((((((8u + start_byte_offset) & 15u) >> 2u) == 2u)) ? (v_21.zw) : (v_21.xy)));
  uint4 v_23 = t_f_params[((16u + start_byte_offset) / 16u)];
  return float3x2(v_20, v_22, asfloat(((((((16u + start_byte_offset) & 15u) >> 2u) == 2u)) ? (v_23.zw) : (v_23.xy))));
}

float3x3 v_24(uint start_byte_offset) {
  return float3x3(asfloat(t_f_params[(start_byte_offset / 16u)].xyz), asfloat(t_f_params[((16u + start_byte_offset) / 16u)].xyz), asfloat(t_f_params[((32u + start_byte_offset) / 16u)].xyz));
}

tint_GammaTransferParams v_25(uint start_byte_offset) {
  tint_GammaTransferParams v_26 = {asfloat(t_f_params[(start_byte_offset / 16u)][((start_byte_offset & 15u) >> 2u)]), asfloat(t_f_params[((4u + start_byte_offset) / 16u)][(((4u + start_byte_offset) & 15u) >> 2u)]), asfloat(t_f_params[((8u + start_byte_offset) / 16u)][(((8u + start_byte_offset) & 15u) >> 2u)]), asfloat(t_f_params[((12u + start_byte_offset) / 16u)][(((12u + start_byte_offset) & 15u) >> 2u)]), asfloat(t_f_params[((16u + start_byte_offset) / 16u)][(((16u + start_byte_offset) & 15u) >> 2u)]), asfloat(t_f_params[((20u + start_byte_offset) / 16u)][(((20u + start_byte_offset) & 15u) >> 2u)]), asfloat(t_f_params[((24u + start_byte_offset) / 16u)][(((24u + start_byte_offset) & 15u) >> 2u)]), t_f_params[((28u + start_byte_offset) / 16u)][(((28u + start_byte_offset) & 15u) >> 2u)]};
  return v_26;
}

float3x4 v_27(uint start_byte_offset) {
  return float3x4(asfloat(t_f_params[(start_byte_offset / 16u)]), asfloat(t_f_params[((16u + start_byte_offset) / 16u)]), asfloat(t_f_params[((32u + start_byte_offset) / 16u)]));
}

tint_ExternalTextureParams v_28(uint start_byte_offset) {
  uint v_29 = t_f_params[(start_byte_offset / 16u)][((start_byte_offset & 15u) >> 2u)];
  uint v_30 = t_f_params[((4u + start_byte_offset) / 16u)][(((4u + start_byte_offset) & 15u) >> 2u)];
  float3x4 v_31 = v_27((16u + start_byte_offset));
  tint_GammaTransferParams v_32 = v_25((64u + start_byte_offset));
  tint_GammaTransferParams v_33 = v_25((96u + start_byte_offset));
  float3x3 v_34 = v_24((128u + start_byte_offset));
  float3x2 v_35 = v_18((176u + start_byte_offset));
  float3x2 v_36 = v_18((200u + start_byte_offset));
  uint4 v_37 = t_f_params[((224u + start_byte_offset) / 16u)];
  float2 v_38 = asfloat(((((((224u + start_byte_offset) & 15u) >> 2u) == 2u)) ? (v_37.zw) : (v_37.xy)));
  uint4 v_39 = t_f_params[((232u + start_byte_offset) / 16u)];
  float2 v_40 = asfloat(((((((232u + start_byte_offset) & 15u) >> 2u) == 2u)) ? (v_39.zw) : (v_39.xy)));
  uint4 v_41 = t_f_params[((240u + start_byte_offset) / 16u)];
  float2 v_42 = asfloat(((((((240u + start_byte_offset) & 15u) >> 2u) == 2u)) ? (v_41.zw) : (v_41.xy)));
  uint4 v_43 = t_f_params[((248u + start_byte_offset) / 16u)];
  float2 v_44 = asfloat(((((((248u + start_byte_offset) & 15u) >> 2u) == 2u)) ? (v_43.zw) : (v_43.xy)));
  uint4 v_45 = t_f_params[((256u + start_byte_offset) / 16u)];
  uint2 v_46 = ((((((256u + start_byte_offset) & 15u) >> 2u) == 2u)) ? (v_45.zw) : (v_45.xy));
  uint4 v_47 = t_f_params[((264u + start_byte_offset) / 16u)];
  tint_ExternalTextureParams v_48 = {v_29, v_30, v_31, v_32, v_33, v_34, v_35, v_36, v_38, v_40, v_42, v_44, v_46, asfloat(((((((264u + start_byte_offset) & 15u) >> 2u) == 2u)) ? (v_47.zw) : (v_47.xy)))};
  return v_48;
}

[numthreads(1, 1, 1)]
void main() {
  tint_ExternalTextureParams v_49 = v_28(0u);
  v_1.Store4(0u, asuint(tint_TextureLoadMultiplanarExternal(t_f_plane0, t_f_plane1, v_49, uint2((int(0)).xx))));
}

