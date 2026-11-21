// q8_0/q4_0 block unpack for the Xe decode kernels (K in phase 1, V in phase 2).
// K/V strides arrive in blocks, callers scale them by KV_BLOCK_SIZE to get element offsets.

#define KV_BLOCK_SIZE 32

#if defined(KV_Q8_0)
struct kv_block_t {
    float16_t d;
    int16_t qs[16];
};
#else
struct kv_block_t {
    float16_t d;
    uint16_t qs[8];
};
#endif

layout (binding = 1) readonly buffer KV_QUANT_BLOCKS {kv_block_t kv_blocks[];};

// four consecutive elements starting at elem (a multiple of 4)
f16vec4 dequant_kv_4(const uint elem) {
    const uint ib  = elem / KV_BLOCK_SIZE;
    const uint iqs = elem % KV_BLOCK_SIZE;
    const float d = float(kv_blocks[ib].d);
#if defined(KV_Q8_0)
    const int w0 = int(kv_blocks[ib].qs[iqs / 2    ]);
    const int w1 = int(kv_blocks[ib].qs[iqs / 2 + 1]);
    return f16vec4(d * vec4((w0 << 24) >> 24, w0 >> 8, (w1 << 24) >> 24, w1 >> 8));
#else
    const uint shift = (iqs & 0x10u) >> 2;
    const uint lo = uint(kv_blocks[ib].qs[(iqs & 0xFu) / 2    ]) >> shift;
    const uint hi = uint(kv_blocks[ib].qs[(iqs & 0xFu) / 2 + 1]) >> shift;
    return f16vec4(d * (vec4(lo & 0xFu, (lo >> 8) & 0xFu, hi & 0xFu, (hi >> 8) & 0xFu) - vec4(8.0f)));
#endif
}
