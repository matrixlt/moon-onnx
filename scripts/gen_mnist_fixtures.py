#!/usr/bin/env python3
"""
生成用于 moon-onnx 端到端验证的 MNIST 测试模型与 ONNX Runtime 黄金参考输出：
1. tests/fixtures/mnist_mlp.onnx : Flatten -> Gemm -> Relu -> Gemm -> Relu -> Gemm -> Softmax
2. tests/fixtures/mnist_cnn.onnx : 对齐 ONNX Model Zoo MNIST-12 结构的双层卷积 CNN

每个模型同时导出：
- *_input.f32  : 小端序 float32 输入张量（shape=[1, 1, 28, 28]）
- *_golden.f32 : ONNX Runtime 实际推理得到的小端序 float32 输出（shape=[1, 10]）
- *_meta.json  : 输入输出形状、预测类别与前若干数值摘要，便于人工检视与自动化校验
"""

import json
import os
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures")
os.makedirs(FIXTURE_DIR, exist_ok=True)


def save_model_and_golden(model: onnx.ModelProto, name: str, sample_input: np.ndarray) -> None:
    onnx.checker.check_model(model)
    model_path = os.path.join(FIXTURE_DIR, f"{name}.onnx")
    onnx.save(model, model_path)

    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: sample_input})
    golden = np.asarray(outputs[0], dtype=np.float32)

    input_path = os.path.join(FIXTURE_DIR, f"{name}_input.f32")
    golden_path = os.path.join(FIXTURE_DIR, f"{name}_golden.f32")
    meta_path = os.path.join(FIXTURE_DIR, f"{name}_meta.json")

    sample_input.astype("<f4").tofile(input_path)
    golden.astype("<f4").tofile(golden_path)

    pred_class = int(np.argmax(golden))
    meta = {
        "model": f"{name}.onnx",
        "input_name": input_name,
        "input_shape": list(sample_input.shape),
        "output_shape": list(golden.shape),
        "predicted_class": pred_class,
        "golden_values": [float(x) for x in golden.flatten()],
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[OK] {name}: saved {model_path} ({os.path.getsize(model_path)} bytes), "
          f"ORT pred={pred_class}, out_shape={list(golden.shape)}")


def build_mnist_mlp() -> onnx.ModelProto:
    rng = np.random.default_rng(42)
    w1 = (rng.standard_normal((64, 784)) * 0.08).astype(np.float32)
    b1 = (rng.standard_normal((64,)) * 0.05).astype(np.float32)
    w2 = (rng.standard_normal((32, 64)) * 0.12).astype(np.float32)
    b2 = (rng.standard_normal((32,)) * 0.05).astype(np.float32)
    w3 = (rng.standard_normal((10, 32)) * 0.15).astype(np.float32)
    b3 = (rng.standard_normal((10,)) * 0.05).astype(np.float32)

    initializers = [
        numpy_helper.from_array(w1, name="fc1.weight"),
        numpy_helper.from_array(b1, name="fc1.bias"),
        numpy_helper.from_array(w2, name="fc2.weight"),
        numpy_helper.from_array(b2, name="fc2.bias"),
        numpy_helper.from_array(w3, name="fc3.weight"),
        numpy_helper.from_array(b3, name="fc3.bias"),
    ]

    nodes = [
        helper.make_node("Flatten", ["input"], ["flat"], name="flatten_0", axis=1),
        helper.make_node("Gemm", ["flat", "fc1.weight", "fc1.bias"], ["h1"], name="gemm_1", transB=1),
        helper.make_node("Relu", ["h1"], ["a1"], name="relu_1"),
        helper.make_node("Gemm", ["a1", "fc2.weight", "fc2.bias"], ["h2"], name="gemm_2", transB=1),
        helper.make_node("Relu", ["h2"], ["a2"], name="relu_2"),
        helper.make_node("Gemm", ["a2", "fc3.weight", "fc3.bias"], ["logits"], name="gemm_3", transB=1),
        helper.make_node("Softmax", ["logits"], ["probs"], name="softmax_0", axis=1),
    ]

    graph = helper.make_graph(
        nodes=nodes,
        name="mnist_mlp",
        inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 28, 28])],
        outputs=[helper.make_tensor_value_info("probs", TensorProto.FLOAT, [1, 10])],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="moon-onnx-fixture-gen",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 9
    return model


def build_mnist_cnn() -> onnx.ModelProto:
    """
    对齐 ONNX Model Zoo MNIST-12 的经典卷积结构：
    Conv(1->8, k=5, pad=2) -> Add -> Relu -> MaxPool(k=2, s=2) -> [1, 8, 14, 14]
    Conv(8->16, k=5, pad=2) -> Add -> Relu -> MaxPool(k=3, s=3) -> [1, 16, 4, 4]
    Reshape([1, 256]) -> MatMul(256->10) -> Add([10]) -> [1, 10]
    """
    rng = np.random.default_rng(123)
    conv1_w = (rng.standard_normal((8, 1, 5, 5)) * 0.15).astype(np.float32)
    conv1_b = (rng.standard_normal((8, 1, 1)) * 0.05).astype(np.float32)
    conv2_w = (rng.standard_normal((16, 8, 5, 5)) * 0.10).astype(np.float32)
    conv2_b = (rng.standard_normal((16, 1, 1)) * 0.05).astype(np.float32)
    reshape_shape = np.array([1, 256], dtype=np.int64)
    fc_w = (rng.standard_normal((256, 10)) * 0.12).astype(np.float32)
    fc_b = (rng.standard_normal((10,)) * 0.05).astype(np.float32)

    initializers = [
        numpy_helper.from_array(conv1_w, name="conv1.weight"),
        numpy_helper.from_array(conv1_b, name="conv1.bias"),
        numpy_helper.from_array(conv2_w, name="conv2.weight"),
        numpy_helper.from_array(conv2_b, name="conv2.bias"),
        numpy_helper.from_array(reshape_shape, name="reshape.shape"),
        numpy_helper.from_array(fc_w, name="fc.weight"),
        numpy_helper.from_array(fc_b, name="fc.bias"),
    ]

    nodes = [
        helper.make_node(
            "Conv",
            ["input", "conv1.weight"],
            ["c1"],
            name="conv_1",
            kernel_shape=[5, 5],
            pads=[2, 2, 2, 2],
            strides=[1, 1],
        ),
        helper.make_node("Add", ["c1", "conv1.bias"], ["c1_b"], name="add_1"),
        helper.make_node("Relu", ["c1_b"], ["r1"], name="relu_1"),
        helper.make_node(
            "MaxPool",
            ["r1"],
            ["p1"],
            name="pool_1",
            kernel_shape=[2, 2],
            strides=[2, 2],
        ),
        helper.make_node(
            "Conv",
            ["p1", "conv2.weight"],
            ["c2"],
            name="conv_2",
            kernel_shape=[5, 5],
            pads=[2, 2, 2, 2],
            strides=[1, 1],
        ),
        helper.make_node("Add", ["c2", "conv2.bias"], ["c2_b"], name="add_2"),
        helper.make_node("Relu", ["c2_b"], ["r2"], name="relu_2"),
        helper.make_node(
            "MaxPool",
            ["r2"],
            ["p2"],
            name="pool_2",
            kernel_shape=[3, 3],
            strides=[3, 3],
        ),
        helper.make_node("Reshape", ["p2", "reshape.shape"], ["flat"], name="reshape_0"),
        helper.make_node("MatMul", ["flat", "fc.weight"], ["mm"], name="matmul_0"),
        helper.make_node("Add", ["mm", "fc.bias"], ["logits"], name="add_3"),
    ]

    graph = helper.make_graph(
        nodes=nodes,
        name="mnist_cnn",
        inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 28, 28])],
        outputs=[helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 10])],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="moon-onnx-fixture-gen",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 9
    return model


def main() -> None:
    rng = np.random.default_rng(2026)
    # 模拟一张 [0, 1] 归一化的 28x28 灰度手写数字图像
    sample_img = rng.uniform(0.0, 1.0, size=(1, 1, 28, 28)).astype(np.float32)

    save_model_and_golden(build_mnist_mlp(), "mnist_mlp", sample_img)
    save_model_and_golden(build_mnist_cnn(), "mnist_cnn", sample_img)


if __name__ == "__main__":
    main()
