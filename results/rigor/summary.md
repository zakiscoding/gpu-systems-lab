| program | row | metric mean ± 95% CI | CV | ms mean | n | SM clock (MHz) |
|---|---|---:|---:|---:|---:|---|
| fused | add_bias alone | 522.5 ± 0.5 | 0.14% | 1.028 | 10 | 2535–2767 |
| fused | gelu alone | 514.7 ± 0.5 | 0.14% | 1.043 | 10 | 2535–2767 |
| fused | unfused (add_bias + gelu) | 518.2 ± 0.5 | 0.13% | 2.072 | 10 | 2535–2767 |
| fused | fused bias_gelu | 514.3 ± 0.7 | 0.19% | 1.044 | 10 | 2535–2767 |
| matmul | cuBLAS sgemm (reference) | 22,735.6 ± 24.6 | 0.15% | 6.045 | 10 | 2850–2857 |
| matmul | naive (16x16 block) | 2,063.3 ± 4.2 | 0.28% | 66.610 | 10 | 2850–2857 |
| matmul | tiled TILE=16 | 2,859.7 ± 0.6 | 0.03% | 48.061 | 10 | 2850–2857 |
| matmul | tiled TILE=32 | 2,377.3 ± 0.2 | 0.01% | 57.814 | 10 | 2850–2857 |
| reduction | atomic (1 per element) | 3.3 ± 0.0 | 0.0% | 40.385 | 10 | 2880–2880 |
| reduction | smem tree (1 atomic/block) | 500.7 ± 1.0 | 0.27% | 0.268 | 10 | 2880–2880 |
| reduction | warp shuffle + float4 | 624.7 ± 0.8 | 0.19% | 0.215 | 10 | 2880–2880 |
| reduction | cub::DeviceReduce::Sum | 613.5 ± 2.1 | 0.48% | 0.219 | 10 | 2880–2880 |
| vector_add | vector_add (grid-stride) | 563.1 ± 0.2 | 0.05% | 1.430 | 10 | 2535–2625 |
| vector_add | vector_add_f4 (float4) | 582.3 ± 0.3 | 0.06% | 1.383 | 10 | 2535–2625 |
| vector_add | cudaMemcpy D2D (ref) | 571.3 ± 0.7 | 0.16% | 0.940 | 10 | 2535–2625 |
