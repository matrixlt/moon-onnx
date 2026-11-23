///|
/// 构造确定性测试矩阵，避免全零数据触发标量内核零值跳过分支导致基准失真。
fn make_matrix(rows : Int32, cols : Int32, seed : Int32) -> FixedArray[Float] {
  FixedArray::makei(rows * cols, fn(i) {
    Float::from_int((i * 17 + seed) % 29 - 14) * (0.05 : Float) + (0.01 : Float)
  })
}

///|
test "bench Add n=65536 [scalar]" (it : @bench.T) {
  let a = @tensor.Tensor::from_f32([65536], make_matrix(256, 256, 3))
  let b = @tensor.Tensor::from_f32([65536], make_matrix(256, 256, 7))
  it.bench(fn() {
    let r = @ops.add_f32(a, b, use_simd=false) catch {
      _ => abort("add failed")
    }
    it.keep(r)
  })
}

///|
test "bench Add n=65536 [simd-v128 FixedArray[Byte] zero-copy]" (it : @bench.T) {
  let a = @vecmath.pack_f32_to_bytes(make_matrix(256, 256, 3))
  let b = @vecmath.pack_f32_to_bytes(make_matrix(256, 256, 7))
  let out = FixedArray::make(65536 * 4, b'\x00')
  it.bench(fn() {
    @vecmath.add_bytes_v128(a, b, out, 65536)
    it.keep(out[0])
  })
}

///|
test "bench Relu n=65536 [scalar]" (it : @bench.T) {
  let x = @tensor.Tensor::from_f32([65536], make_matrix(256, 256, 11))
  it.bench(fn() {
    let r = @ops.relu_f32(x, use_simd=false) catch { _ => abort("relu failed") }
    it.keep(r)
  })
}

///|
test "bench Relu n=65536 [simd-v128 FixedArray[Byte] pmax]" (it : @bench.T) {
  let x = @vecmath.pack_f32_to_bytes(make_matrix(256, 256, 11))
  let out = FixedArray::make(65536 * 4, b'\x00')
  it.bench(fn() {
    @vecmath.relu_bytes_v128(x, out, 65536)
    it.keep(out[0])
  })
}

///|
test "bench GEMM 64x64x64 [scalar i-k-j]" (it : @bench.T) {
  let a = make_matrix(64, 64, 1)
  let b = make_matrix(64, 64, 2)
  let c = FixedArray::make(64 * 64, (0.0 : Float))
  it.bench(fn() {
    @gemm.matmul_2d_scalar_f32(a, 0, b, 0, c, 0, 64, 64, 64)
    it.keep(c[0])
  })
}

///|
test "bench GEMM 64x64x64 [simd-v128 4x8 tiled]" (it : @bench.T) {
  let a = make_matrix(64, 64, 1)
  let b = make_matrix(64, 64, 2)
  let c = FixedArray::make(64 * 64, (0.0 : Float))
  it.bench(fn() {
    @gemm.matmul_2d_simd_f32(a, 0, b, 0, c, 0, 64, 64, 64)
    it.keep(c[0])
  })
}

///|
test "bench GEMM 64x64x64 [simd-v128 prepacked B]" (it : @bench.T) {
  let a = make_matrix(64, 64, 1)
  let packed_b = @gemm.pack_matrix_b_f32(make_matrix(64, 64, 2), 0, 64, 64)
  let c = FixedArray::make(64 * 64, (0.0 : Float))
  it.bench(fn() {
    @gemm.matmul_2d_prepacked_simd_f32(a, 0, packed_b, c, 0, 64)
    it.keep(c[0])
  })
}

///|
test "bench GEMM 256x256x256 [scalar i-k-j]" (it : @bench.T) {
  let a = make_matrix(256, 256, 1)
  let b = make_matrix(256, 256, 2)
  let c = FixedArray::make(256 * 256, (0.0 : Float))
  it.bench(fn() {
    @gemm.matmul_2d_scalar_f32(a, 0, b, 0, c, 0, 256, 256, 256)
    it.keep(c[0])
  })
}

///|
test "bench GEMM 256x256x256 [simd-v128 4x8 tiled]" (it : @bench.T) {
  let a = make_matrix(256, 256, 1)
  let b = make_matrix(256, 256, 2)
  let c = FixedArray::make(256 * 256, (0.0 : Float))
  it.bench(fn() {
    @gemm.matmul_2d_simd_f32(a, 0, b, 0, c, 0, 256, 256, 256)
    it.keep(c[0])
  })
}

///|
test "bench E2E MNIST MLP [scalar]" (it : @bench.T) {
  let sess = @session.Session::from_file(
    "../tests/fixtures/mnist_mlp.onnx",
    backend=@session.Scalar,
  )
  let input = @session.load_f32_file("../tests/fixtures/mnist_mlp_input.f32", [
    1, 1, 28, 28,
  ])
  it.bench(fn() {
    let out = sess.run_single(input) catch { _ => abort("mlp failed") }
    it.keep(out)
  })
}

///|
test "bench E2E MNIST MLP [simd-v128]" (it : @bench.T) {
  let sess = @session.Session::from_file(
    "../tests/fixtures/mnist_mlp.onnx",
    backend=@session.Simd,
  )
  let input = @session.load_f32_file("../tests/fixtures/mnist_mlp_input.f32", [
    1, 1, 28, 28,
  ])
  it.bench(fn() {
    let out = sess.run_single(input) catch { _ => abort("mlp simd failed") }
    it.keep(out)
  })
}

///|
test "bench E2E MNIST CNN [scalar, optimized]" (it : @bench.T) {
  let sess = @session.Session::from_file(
    "../tests/fixtures/mnist_cnn.onnx",
    backend=@session.Scalar,
    optimize=true,
  )
  let input = @session.load_f32_file("../tests/fixtures/mnist_cnn_input.f32", [
    1, 1, 28, 28,
  ])
  it.bench(fn() {
    let out = sess.run_single(input) catch { _ => abort("cnn failed") }
    it.keep(out)
  })
}

///|
test "bench E2E MNIST CNN [simd-v128, optimized]" (it : @bench.T) {
  let sess = @session.Session::from_file(
    "../tests/fixtures/mnist_cnn.onnx",
    backend=@session.Simd,
    optimize=true,
  )
  let input = @session.load_f32_file("../tests/fixtures/mnist_cnn_input.f32", [
    1, 1, 28, 28,
  ])
  it.bench(fn() {
    let out = sess.run_single(input) catch { _ => abort("cnn simd failed") }
    it.keep(out)
  })
}
