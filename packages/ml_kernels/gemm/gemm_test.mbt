///|
/// 验证非对齐及大尺寸下 `matmul_2d_simd_f32` 与 `matmul_2d_prepacked_simd_f32` 均与标量内核精确对齐。
test "V128 GEMM (on-the-fly and prepacked) matches scalar GEMM across odd dimensions" {
  let shapes = [(1, 17, 10), (3, 13, 7), (4, 16, 8), (13, 29, 71), (32, 64, 65)]
  for idx = 0; idx < shapes.length(); idx = idx + 1 {
    let (m, k, n) = shapes[idx]
    let a = FixedArray::makei(m * k, fn(i) {
      Float::from_int(i % 19 - 9) * (0.125 : Float)
    })
    let b = FixedArray::makei(k * n, fn(i) {
      Float::from_int(i % 23 - 11) * (0.0625 : Float)
    })
    let out_scalar = FixedArray::make(m * n, (0.0 : Float))
    let out_simd = FixedArray::make(m * n, (0.0 : Float))
    let out_prepacked = FixedArray::make(m * n, (0.0 : Float))

    @gemm.matmul_2d_scalar_f32(a, 0, b, 0, out_scalar, 0, m, k, n)
    @gemm.matmul_2d_simd_f32(a, 0, b, 0, out_simd, 0, m, k, n)

    let packed_b = @gemm.pack_matrix_b_f32(b, 0, k, n)
    assert_eq(packed_b.dims(), (k, n))
    @gemm.matmul_2d_prepacked_simd_f32(a, 0, packed_b, out_prepacked, 0, m)

    for i = 0; i < m * n; i = i + 1 {
      assert_true((out_scalar[i] - out_simd[i]).abs() <= (1.0e-5 : Float))
      assert_true((out_simd[i] - out_prepacked[i]).abs() == (0.0 : Float))
    }
  }
}

///|
/// 验证 `pack_matrix_b_transposed_f32` 与 `matmul_2d_prepacked_nchw_simd_f32`（含偏置融合与非对齐尺寸）
/// 在多组奇偶维度下与标量 `W[N, K] @ A^T[K, M] + bias` 严格对齐。
test "pack_matrix_b_transposed_f32 and matmul_2d_prepacked_nchw_simd_f32 match scalar reference" {
  let shapes = [(1, 9, 3), (5, 17, 7), (8, 25, 8), (13, 19, 11), (16, 33, 19)]
  for idx = 0; idx < shapes.length(); idx = idx + 1 {
    let (m, k, n) = shapes[idx]
    let a = FixedArray::makei(m * k, fn(i) {
      Float::from_int(i % 17 - 8) * (0.125 : Float)
    })
    let w = FixedArray::makei(n * k, fn(i) {
      Float::from_int(i % 23 - 11) * (0.0625 : Float)
    })
    let bias = FixedArray::makei(n, fn(i) {
      Float::from_int(i - n / 2) * (0.2 : Float)
    })

    // 构造 A^T[K, M] 作为标量参考输入，验证 W[N, K] @ A^T[K, M] + bias
    let a_t = FixedArray::makei(k * m, fn(idx_t) {
      let p = idx_t / m
      let row = idx_t % m
      a[row * k + p]
    })
    let ref_out = FixedArray::make(n * m, (0.0 : Float))
    @gemm.matmul_2d_scalar_f32(w, 0, a_t, 0, ref_out, 0, n, k, m)

    let packed_wt = @gemm.pack_matrix_b_transposed_f32(w, 0, n, k)
    assert_eq(packed_wt.dims(), (k, n))

    // 无偏置验证
    let out_no_bias = FixedArray::make(n * m, (-77.0 : Float))
    @gemm.matmul_2d_prepacked_nchw_simd_f32(
      a,
      0,
      packed_wt,
      None,
      0,
      out_no_bias,
      0,
      m,
    )
    for ch = 0; ch < n; ch = ch + 1 {
      for s = 0; s < m; s = s + 1 {
        let pos = ch * m + s
        assert_true((out_no_bias[pos] - ref_out[pos]).abs() <= (1.0e-5 : Float))
      }
    }

    // 融合偏置验证
    let out_with_bias = FixedArray::make(n * m, (-77.0 : Float))
    @gemm.matmul_2d_prepacked_nchw_simd_f32(
      a,
      0,
      packed_wt,
      Some(bias),
      0,
      out_with_bias,
      0,
      m,
    )
    for ch = 0; ch < n; ch = ch + 1 {
      for s = 0; s < m; s = s + 1 {
        let pos = ch * m + s
        assert_true(
          (out_with_bias[pos] - (ref_out[pos] + bias[ch])).abs() <=
          (1.0e-5 : Float),
        )
      }
    }
  }
}

///|
/// 穷举验证 6×8 / 4×8 / 1×8 分块及 K 维 2× 展开奇偶尾部组合（M=1..13, K=1..5, N=1,7,8,9,17）
/// 与 NCHW 全部 Epilogue（Relu / Relu6 / Silu）的数值正确性。
test "6x8 K-unroll-2 GEMM covers all M/K/N tail paths and NCHW epilogues" {
  let ks = [1, 2, 3, 5]
  let ns = [1, 7, 8, 9, 17]
  for m = 1; m <= 13; m = m + 1 {
    for ki = 0; ki < ks.length(); ki = ki + 1 {
      let k = ks[ki]
      for ni = 0; ni < ns.length(); ni = ni + 1 {
        let n = ns[ni]
        let a = FixedArray::makei(m * k, fn(i) {
          Float::from_int(i % 11 - 5) * (0.25 : Float)
        })
        let b = FixedArray::makei(k * n, fn(i) {
          Float::from_int(i % 13 - 6) * (0.125 : Float)
        })
        let ref_out = FixedArray::make(m * n, (0.0 : Float))
        let simd_out = FixedArray::make(m * n, (0.0 : Float))
        @gemm.matmul_2d_scalar_f32(a, 0, b, 0, ref_out, 0, m, k, n)
        let pb = @gemm.pack_matrix_b_f32(b, 0, k, n)
        @gemm.matmul_2d_prepacked_simd_f32(a, 0, pb, simd_out, 0, m)
        for idx = 0; idx < m * n; idx = idx + 1 {
          assert_true((ref_out[idx] - simd_out[idx]).abs() <= (1.0e-5 : Float))
        }
      }
    }
  }
  // NCHW 全部 Epilogue 验证（涵盖 6 行主块 + 4 行尾块 + 1 行尾块：M = 6 + 4 + 1 = 11）
  let m = 11
  let k = 7
  let n = 13
  let a = FixedArray::makei(m * k, fn(i) {
    Float::from_int(i % 15 - 7) * (0.2 : Float)
  })
  let w = FixedArray::makei(n * k, fn(i) {
    Float::from_int(i % 17 - 8) * (0.15 : Float)
  })
  let bias = FixedArray::makei(n, fn(i) {
    Float::from_int(i - 6) * (0.3 : Float)
  })
  let pb_t = @gemm.pack_matrix_b_transposed_f32(w, 0, n, k)
  let eps = [
    @gemm.Relu,
    @gemm.Clip(min=(0.0 : Float), max=(6.0 : Float)),
    @gemm.Silu,
  ]
  for ei = 0; ei < eps.length(); ei = ei + 1 {
    let ep = eps[ei]
    let out = FixedArray::make(n * m, (0.0 : Float))
    @gemm.matmul_2d_prepacked_nchw_simd_f32(
      a,
      0,
      pb_t,
      Some(bias),
      0,
      out,
      0,
      m,
      epilogue=ep,
    )
    for ch = 0; ch < n; ch = ch + 1 {
      for s = 0; s < m; s = s + 1 {
        let mut acc = bias[ch]
        for p = 0; p < k; p = p + 1 {
          acc = acc + a[s * k + p] * w[ch * k + p]
        }
        let expected = @gemm.apply_epilogue_scalar(acc, ep)
        assert_true((out[ch * m + s] - expected).abs() <= (1.0e-4 : Float))
      }
    }
  }
}
