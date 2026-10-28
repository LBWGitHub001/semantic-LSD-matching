"""
Concrete per-layer computational graphs for the matching networks.

Each node is one real operation in the forward (Conv, BatchNorm, residual Add,
MultiheadAttention, Log-Sinkhorn, ...), with the tensor shapes that the code
actually produces and a semantic note grounded in that implementation.
"""

from __future__ import annotations

from typing import Any


def _conv_params(cin: int, cout: int, k: int, bias: bool = False) -> int:
    return cout * cin * k * k + (cout if bias else 0)


def _bn_params(channels: int) -> int:
    return channels * 2


def _linear_params(cin: int, cout: int, bias: bool = True) -> int:
    return cin * cout + (cout if bias else 0)


def _ln_params(dim: int) -> int:
    return dim * 2


def _mha_params(dim: int) -> int:
    # PyTorch MultiheadAttention: in_proj (3d, d) + bias 3d, out_proj (d, d) + bias d.
    return 4 * dim * dim + 4 * dim


class _Graph:
    def __init__(self) -> None:
        self.nodes: list[dict[str, Any]] = []
        self.edges: list[dict[str, str]] = []
        self._ids: set[str] = set()

    def node(self, **fields: Any) -> str:
        node_id = fields["id"]
        if node_id in self._ids:
            raise ValueError(f"duplicate node id {node_id}")
        self._ids.add(node_id)
        fields.setdefault("params", 0)
        fields.setdefault("weight", "")
        fields.setdefault("formula", "")
        fields.setdefault("spec", "")
        fields.setdefault("frozen", False)
        fields.setdefault("col", 0)
        fields.setdefault("row", 0)
        self.nodes.append(fields)
        return node_id

    def get(self, node_id: str) -> dict[str, Any]:
        for node in self.nodes:
            if node["id"] == node_id:
                return node
        raise KeyError(node_id)

    def edge(self, src: str, dst: str, kind: str = "data", label: str = "") -> None:
        self.edges.append({"from": src, "to": dst, "kind": kind, "label": label})

    def chain(self, ids: list[str]) -> None:
        for src, dst in zip(ids, ids[1:]):
            self.edge(src, dst, "data", self.get(src)["shape_out"])

    def place(self, ids: list[str], col: int, start_row: int) -> int:
        for offset, node_id in enumerate(ids):
            node = self.get(node_id)
            node["col"] = col
            node["row"] = start_row + offset
        return start_row + len(ids) - 1


def _residual_block(
    graph: _Graph,
    prefix: str,
    channels: int,
    spatial: str,
    module: str,
    role: str,
) -> list[str]:
    """BasicResidualBlock: Conv-BN-ReLU-Conv-BN, then Add with the block input, then ReLU."""
    shape = f"(N, {channels}, {spatial})"
    conv_p = _conv_params(channels, channels, 3, bias=False)
    ids = [
        graph.node(
            id=f"{prefix}_conv1",
            op="Conv2d",
            kind="conv",
            module=f"{module}.conv1",
            spec="3×3 · stride 1 · 无偏置",
            shape_in=shape,
            shape_out=shape,
            weight=f"W {channels}×{channels}×3×3",
            params=conv_p,
            formula="Y = W ∗ X，padding=1，分辨率不变",
            source="models/object_matcher/xfeat_object_matcher.py:25",
            meaning=(
                f"{role}残差块的第一层 3×3。它不改变 {channels} 通道，也不下采样，"
                "只在当前尺度上混合邻域。无偏置，因为后面的 BatchNorm 会重新学习偏移。"
                "这一层负责把边缘、角点这类局部响应组合成稍大的结构。"
            ),
        ),
        graph.node(
            id=f"{prefix}_bn1",
            op="BatchNorm2d",
            kind="norm",
            module=f"{module}.bn1",
            spec=f"{channels} 通道",
            shape_in=shape,
            shape_out=shape,
            weight=f"γ, β 各 {channels}",
            params=_bn_params(channels),
            formula="(X − μ) / √(σ²+ε) · γ + β",
            source="models/object_matcher/xfeat_object_matcher.py:26",
            meaning=(
                "把这一层的通道拉回零均值、单位方差，再学习缩放 γ 和平移 β。"
                "训练时用当前 batch 的统计量，推理时用滑动平均。它让后面的 ReLU 不会因为尺度漂移而整片饱和。"
            ),
        ),
        graph.node(
            id=f"{prefix}_relu1",
            op="ReLU",
            kind="act",
            module=f"{module}.relu",
            spec="inplace",
            shape_in=shape,
            shape_out=shape,
            formula="max(0, X)",
            source="models/object_matcher/xfeat_object_matcher.py:33",
            meaning=(
                "负响应直接截断。边缘卷积常常一侧为正、一侧为负，ReLU 只留下正激活。"
                "这是残差块里唯一放在相加之前的非线性；相加之前如果再来一次 ReLU，恒等支路就对不齐。"
            ),
        ),
        graph.node(
            id=f"{prefix}_conv2",
            op="Conv2d",
            kind="conv",
            module=f"{module}.conv2",
            spec="3×3 · stride 1 · 无偏置",
            shape_in=shape,
            shape_out=shape,
            weight=f"W {channels}×{channels}×3×3",
            params=conv_p,
            formula="Y = W ∗ X，padding=1",
            source="models/object_matcher/xfeat_object_matcher.py:28",
            meaning=(
                "残差块的第二层 3×3，仍然保持分辨率。两层 3×3 的感受野约等于一层 5×5，"
                "但参数更少，中间还插了一层 ReLU。它学习的是“要加回到恒等支路上的残差”，不是整幅特征本身。"
            ),
        ),
        graph.node(
            id=f"{prefix}_bn2",
            op="BatchNorm2d",
            kind="norm",
            module=f"{module}.bn2",
            spec=f"{channels} 通道",
            shape_in=shape,
            shape_out=shape,
            weight=f"γ, β 各 {channels}",
            params=_bn_params(channels),
            formula="(X − μ) / √(σ²+ε) · γ + β",
            source="models/object_matcher/xfeat_object_matcher.py:34",
            meaning=(
                "第二层卷积之后先归一化，再和恒等支路相加。注意这里没有紧跟 ReLU："
                "代码是 `self.bn2(self.conv2(out))`，相加发生在非线性之前，这样残差可以是负的修正量。"
            ),
        ),
        graph.node(
            id=f"{prefix}_add",
            op="Add",
            kind="merge",
            module=f"{module}.forward",
            spec="恒等捷径 + 残差",
            shape_in=shape,
            shape_out=shape,
            formula="Y = F(X) + X",
            source="models/object_matcher/xfeat_object_matcher.py:35",
            meaning=(
                "把进入本块的特征原样加回来。即使两层 3×3 还没学好，梯度也能沿捷径直接回到浅层。"
                "对匹配来说，这保住了已经检测到的边界，不会被后面的卷积洗掉。"
            ),
        ),
        graph.node(
            id=f"{prefix}_relu2",
            op="ReLU",
            kind="act",
            module=f"{module}.relu",
            spec="inplace · 与块内共用",
            shape_in=shape,
            shape_out=shape,
            formula="max(0, X)",
            source="models/object_matcher/xfeat_object_matcher.py:35",
            meaning=(
                "相加之后的 ReLU，是这个残差块的输出。"
                "同一个 `self.relu` 模块被用了两次，它没有参数，两次调用只是两次截断。"
            ),
        ),
    ]
    graph.chain(ids)
    return ids


def _down_stage(
    graph: _Graph,
    prefix: str,
    cin: int,
    cout: int,
    spatial_in: str,
    spatial_out: str,
    module: str,
    role: str,
) -> tuple[list[str], str]:
    """Conv stride-2 + BN + ReLU, then a same-resolution residual block. Returns ids and the tensor id that laterals read."""
    shape_in = f"(N, {cin}, {spatial_in})"
    shape_mid = f"(N, {cout}, {spatial_out})"
    stem = [
        graph.node(
            id=f"{prefix}_conv",
            op="Conv2d",
            kind="conv",
            module=f"{module}.0",
            spec=f"3×3 · stride 2 · {cin}→{cout}",
            shape_in=shape_in,
            shape_out=shape_mid,
            weight=f"W {cout}×{cin}×3×3 · 无偏置",
            params=_conv_params(cin, cout, 3, bias=False),
            formula="Y = W ∗ X，stride=2，padding=1",
            source="models/object_matcher/xfeat_object_matcher.py:99",
            meaning=(
                f"{role}这是本阶段唯一的下采样。3×3 卷积步长 2，把空间尺寸减半，通道从 {cin} 抬到 {cout}。"
                "步长放在进入残差块之前，而不是放在残差块内部，这样捷径分支不用再做投影，可以直接相加。"
            ),
        ),
        graph.node(
            id=f"{prefix}_bn",
            op="BatchNorm2d",
            kind="norm",
            module=f"{module}.1",
            spec=f"{cout} 通道",
            shape_in=shape_mid,
            shape_out=shape_mid,
            weight=f"γ, β 各 {cout}",
            params=_bn_params(cout),
            formula="(X − μ) / √(σ²+ε) · γ + β",
            source="models/object_matcher/xfeat_object_matcher.py:100",
            meaning="下采样卷积没有偏置，这一层 BatchNorm 同时完成归一化和可学习的通道偏移，避免半分辨率特征的幅度随深度爆炸。",
        ),
        graph.node(
            id=f"{prefix}_relu",
            op="ReLU",
            kind="act",
            module=f"{module}.2",
            spec="inplace",
            shape_in=shape_mid,
            shape_out=shape_mid,
            formula="max(0, X)",
            source="models/object_matcher/xfeat_object_matcher.py:101",
            meaning="下采样之后立刻截断负值。这个输出既是残差块的输入，也是残差相加时的恒等支路。",
        ),
    ]
    graph.chain(stem)
    block = _residual_block(
        graph,
        prefix=f"{prefix}_res",
        channels=cout,
        spatial=spatial_out,
        module=f"{module}.3",
        role=role,
    )
    graph.edge(stem[-1], block[0], "data", stem[-1] and graph.get(stem[-1])["shape_out"])
    graph.edge(stem[-1], block[-2], "skip", "")
    return stem + block, block[-1]


def _prenorm_block(
    graph: _Graph,
    prefix: str,
    d_model: int,
    nhead: int,
    module: str,
    query_from: str,
    key_from: str,
    role: str,
) -> list[str]:
    """AttentionBlock: Pre-Norm MHA + residual, then Pre-Norm FFN + residual."""
    shape = f"(B, M, {d_model})"
    head_dim = d_model // nhead
    ids = [
        graph.node(
            id=f"{prefix}_ln1",
            op="LayerNorm",
            kind="norm",
            module=f"{module}.norm1",
            spec=f"{d_model} 维",
            shape_in=shape,
            shape_out=shape,
            weight=f"γ, β 各 {d_model}",
            params=_ln_params(d_model),
            formula="对最后一维做归一化",
            source="models/object_matcher/xfeat_object_matcher.py:232",
            meaning=(
                f"{role}这是 Pre-Norm：先进 LayerNorm，再算注意力，残差加在归一化之前的 token 上。"
                "同一套 norm1 会作用到 Query、Key、Value。和线段网络里“先残差再 Norm”的 Post-Norm 不是同一种稳定方式。"
            ),
        ),
        graph.node(
            id=f"{prefix}_mha",
            op="MultiheadAttention",
            kind="attn",
            module=f"{module}.mha",
            spec=f"{nhead} 头 · head_dim={head_dim}",
            shape_in=shape,
            shape_out=shape,
            weight=f"in_proj {3 * d_model}×{d_model} + out_proj {d_model}×{d_model}",
            params=_mha_params(d_model),
            formula=f"softmax(QKᵀ / √{head_dim}) V",
            source="models/object_matcher/xfeat_object_matcher.py:235",
            meaning=(
                f"{role}{nhead} 个头各自在 {head_dim} 维里做缩放点积，再拼回去。"
                f"Query 来自{query_from}，Key 和 Value 来自{key_from}。"
                "头与头之间不共享投影。batch_first=True，所以张量是 (B, M, D) 而不是 (M, B, D)。"
            ),
        ),
        graph.node(
            id=f"{prefix}_add1",
            op="Add",
            kind="merge",
            module=f"{module}.forward",
            spec="注意力残差",
            shape_in=shape,
            shape_out=shape,
            formula="X ← X + Attn(LN(X))",
            source="models/object_matcher/xfeat_object_matcher.py:236",
            meaning="注意力的输出只作为增量加回原 token。外观和几何已经写进 token，注意力只负责按邻里关系改写，不应该覆盖它们。",
        ),
        graph.node(
            id=f"{prefix}_ln2",
            op="LayerNorm",
            kind="norm",
            module=f"{module}.norm2",
            spec=f"{d_model} 维",
            shape_in=shape,
            shape_out=shape,
            weight=f"γ, β 各 {d_model}",
            params=_ln_params(d_model),
            formula="对最后一维做归一化",
            source="models/object_matcher/xfeat_object_matcher.py:237",
            meaning="前馈网络之前的第二层 Pre-Norm。它和 norm1 是两个独立模块，不共享 γ、β。",
        ),
        graph.node(
            id=f"{prefix}_fc1",
            op="Linear",
            kind="linear",
            module=f"{module}.mlp.0",
            spec=f"{d_model} → {d_model * 2}",
            shape_in=shape,
            shape_out=f"(B, M, {d_model * 2})",
            weight=f"W {d_model * 2}×{d_model} + bias",
            params=_linear_params(d_model, d_model * 2),
            formula="Y = XWᵀ + b",
            source="models/object_matcher/xfeat_object_matcher.py:226",
            meaning=(
                "逐 token 的通道扩展，不混合不同目标。注意力是目标之间的线性组合，这一层给每个目标自己的非线性容量。"
                f"隐藏宽度是 {d_model * 2}，比常见的 4 倍前馈更窄，用来压参数量。"
            ),
        ),
        graph.node(
            id=f"{prefix}_relu",
            op="ReLU",
            kind="act",
            module=f"{module}.mlp.1",
            spec="inplace",
            shape_in=f"(B, M, {d_model * 2})",
            shape_out=f"(B, M, {d_model * 2})",
            formula="max(0, X)",
            source="models/object_matcher/xfeat_object_matcher.py:227",
            meaning="前馈中间层的 ReLU。负半轴被丢掉，所以扩展后的通道只有一半方向能传到下一层线性。",
        ),
        graph.node(
            id=f"{prefix}_fc2",
            op="Linear",
            kind="linear",
            module=f"{module}.mlp.2",
            spec=f"{d_model * 2} → {d_model}",
            shape_in=f"(B, M, {d_model * 2})",
            shape_out=shape,
            weight=f"W {d_model}×{d_model * 2} + bias",
            params=_linear_params(d_model * 2, d_model),
            formula="Y = XWᵀ + b",
            source="models/object_matcher/xfeat_object_matcher.py:228",
            meaning="把前馈隐藏层压回 d_model，才能和残差支路相加。这一层没有再接归一化，归一化留给下一层的 Pre-Norm。",
        ),
        graph.node(
            id=f"{prefix}_add2",
            op="Add",
            kind="merge",
            module=f"{module}.forward",
            spec="前馈残差",
            shape_in=shape,
            shape_out=shape,
            formula="X ← X + MLP(LN(X))",
            source="models/object_matcher/xfeat_object_matcher.py:237",
            meaning="前馈增量加回注意力之后的 token。这一层的输出才是该注意力块交给下一块的表示。",
        ),
    ]
    graph.chain(ids)
    # Attention residual comes from the tensor entering the block (caller adds that
    # edge onto add1). The FFN residual comes from the attention block's own sum.
    graph.edge(ids[2], ids[7], "skip", "")
    return ids


def _object_transformer(
    graph: _Graph,
    d_model: int,
    num_layers: int,
    nhead: int,
    col0: int,
    col1: int,
    start_row: int,
    src: str,
) -> tuple[str, str]:
    """Two-view graph transformer. Returns the final token ids of view 0 and view 1."""
    prev0 = src
    prev1 = src
    row = start_row
    last0 = src
    last1 = src
    for layer_idx in range(num_layers):
        self0 = _prenorm_block(
            graph, f"l{layer_idx}_self0", d_model, nhead,
            module=f"transformer.layers.{layer_idx}.self0",
            query_from="本视角（view 0）",
            key_from="本视角（view 0）",
            role=f"第 {layer_idx} 层 view0 自注意力。",
        )
        self1 = _prenorm_block(
            graph, f"l{layer_idx}_self1", d_model, nhead,
            module=f"transformer.layers.{layer_idx}.self1",
            query_from="本视角（view 1）",
            key_from="本视角（view 1）",
            role=f"第 {layer_idx} 层 view1 自注意力。权重与 self0 独立，不共享。",
        )
        graph.place(self0, col0, row)
        graph.place(self1, col1, row)
        graph.edge(prev0, self0[0], "data", graph.get(prev0)["shape_out"])
        graph.edge(prev1, self1[0], "data", graph.get(prev1)["shape_out"])
        graph.edge(prev0, self0[2], "skip", "")
        graph.edge(prev1, self1[2], "skip", "")

        cross_row = row + len(self0)
        cross0 = _prenorm_block(
            graph, f"l{layer_idx}_cross0", d_model, nhead,
            module=f"transformer.layers.{layer_idx}.cross0",
            query_from="view 0",
            key_from="view 1（另一视角的自注意力输出）",
            role=f"第 {layer_idx} 层 view0 交叉注意力。",
        )
        cross1 = _prenorm_block(
            graph, f"l{layer_idx}_cross1", d_model, nhead,
            module=f"transformer.layers.{layer_idx}.cross1",
            query_from="view 1",
            key_from="view 0（另一视角的自注意力输出）",
            role=f"第 {layer_idx} 层 view1 交叉注意力。权重与 cross0 独立。",
        )
        graph.place(cross0, col0, cross_row)
        graph.place(cross1, col1, cross_row)
        graph.edge(self0[-1], cross0[0], "data", graph.get(self0[-1])["shape_out"])
        graph.edge(self1[-1], cross1[0], "data", graph.get(self1[-1])["shape_out"])
        graph.edge(self0[-1], cross0[2], "skip", "")
        graph.edge(self1[-1], cross1[2], "skip", "")
        graph.edge(self1[-1], cross0[1], "cross", "K,V ← view1")
        graph.edge(self0[-1], cross1[1], "cross", "K,V ← view0")
        prev0, prev1 = cross0[-1], cross1[-1]
        last0, last1 = prev0, prev1
        row = cross_row + len(cross0) + 1
    return last0, last1


def _sinkhorn_head(
    graph: _Graph,
    d_model: int,
    col: int,
    row: int,
    src0: str,
    src1: str,
    iters: int = 20,
) -> str:
    shape = f"(B, M, {d_model})"
    q = graph.node(
        id="proj0",
        op="Linear",
        kind="linear",
        module="matcher.proj0",
        spec=f"{d_model} → {d_model} · 无偏置",
        shape_in=shape,
        shape_out=shape,
        weight=f"W {d_model}×{d_model}",
        params=_linear_params(d_model, d_model, bias=False),
        formula="Q = X₀ W₀",
        source="models/object_matcher/xfeat_object_matcher.py:289",
        meaning="把 view 0 的 token 投成查询向量。没有偏置，相似度只反映方向，不给某一类目标一个固定加分。",
    )
    k = graph.node(
        id="proj1",
        op="Linear",
        kind="linear",
        module="matcher.proj1",
        spec=f"{d_model} → {d_model} · 无偏置",
        shape_in=shape,
        shape_out=shape,
        weight=f"W {d_model}×{d_model}",
        params=_linear_params(d_model, d_model, bias=False),
        formula="K = X₁ W₁",
        source="models/object_matcher/xfeat_object_matcher.py:290",
        meaning="view 1 使用另一套投影。两边不共享 W，因为两个视角的 token 分布不必对称到同一套线性变换。",
    )
    score = graph.node(
        id="bmm",
        op="BMM",
        kind="op",
        module="matcher.forward",
        spec=f"÷ √{d_model}",
        shape_in=shape,
        shape_out="(B, M0, M1)",
        formula=f"S = Q Kᵀ / √{d_model}",
        source="models/object_matcher/xfeat_object_matcher.py:291",
        meaning=(
            "批量矩阵乘得到每一对目标的亲和度。除以 √d_model 是缩放点积，避免维度变大时点积幅度把后面的 softmax 推饱和。"
            "这一步还没有匹配约束，一行可以同时喜欢多个目标。"
        ),
    )
    dust = graph.node(
        id="dustbin",
        op="Pad",
        kind="op",
        module="matcher.dustbin_score",
        spec="可学习标量",
        shape_in="(B, M0, M1)",
        shape_out="(B, M0+1, M1+1)",
        weight="dustbin_score ∈ ℝ",
        params=1,
        formula="右侧一列、底下一行、右下角填入同一标量",
        source="models/object_matcher/xfeat_object_matcher.py:293",
        meaning=(
            "给分数矩阵加一列和一行垃圾桶。出画、被挡住、只在一个视角出现的区域应该流进这里，而不是被强迫配给某个错误目标。"
            "这个标量是可学习的，初始为 0。它不是“第 M+1 个物体”的特征，只是一条拒绝匹配的得分。"
        ),
    )
    sink = graph.node(
        id="sinkhorn",
        op="LogSinkhorn",
        kind="sinkhorn",
        module="log_sinkhorn_iterations",
        spec=f"{iters} 次 · FP32",
        shape_in="(B, M0+1, M1+1)",
        shape_out="(B, M0+1, M1+1)",
        formula="u ← log μ − logsumexp(Z+v)；v ← log ν − logsumexp(Z+u)",
        source="models/object_matcher/xfeat_object_matcher.py:266",
        meaning=(
            f"在对数空间交替做行、列归一化 {iters} 次，把亲和度变成双随机矩阵：每个目标的匹配概率加起来为 1，每个被匹配目标也是。"
            "实现强制用 FP32，因为 FP16 的 logsumexp 会超过 65504。边际分布给垃圾桶更大的质量，允许多个落单目标同时被拒绝。"
        ),
    )
    prob = graph.node(
        id="exp",
        op="Exp",
        kind="op",
        module="matcher.forward",
        spec="clamp logP ∈ [−100, 0]",
        shape_in="(B, M0+1, M1+1)",
        shape_out="(B, M0+1, M1+1)",
        formula="P = exp(clamp(log P, −100, 0))",
        source="models/object_matcher/xfeat_object_matcher.py:314",
        meaning=(
            "对数分配矩阵指数化成概率，输出就是 prob_matrix。最后一行和最后一列是垃圾桶。"
            "推理时再做双向阈值，而不是在这一层做 argmax。"
        ),
    )
    graph.place([q], col, row)
    graph.place([k], col + 1, row)
    graph.place([score, dust, sink, prob], col, row + 1)
    graph.edge(src0, q, "data", graph.get(src0)["shape_out"])
    graph.edge(src1, k, "data", graph.get(src1)["shape_out"])
    graph.edge(q, score, "data", shape)
    graph.edge(k, score, "data", shape)
    graph.chain([score, dust, sink, prob])
    return prob


def _line_encoder(graph: _Graph, line_dim: int) -> list[str]:
    shape_k = "(B, M, K, 5)"
    shape_h = f"(B, M, K, {line_dim})"
    shape_p = f"(B, M, {line_dim})"
    ids = [
        graph.node(
            id="line_in",
            op="Input",
            kind="input",
            module="PatchLineGeometryEncoder",
            spec="端点或 5D",
            shape_in="(B, M, K, 4) 或 (B, M, K, 5)",
            shape_out="(B, M, K, 4|5)",
            formula="K 为每个切片保留的直线槽，默认 8",
            source="models/object_matcher/xfeat_line_matcher.py:93",
            meaning=(
                "每个目标切片带上落在框内的直线。4 维是切片归一化端点 [u1,v1,u2,v2]，5 维则已经是描述子。"
                "K 维是槽位，不是一个卷积通道；空槽由 valid_mask 标出，不参与池化。"
            ),
        ),
        graph.node(
            id="to5d",
            op="Geom5D",
            kind="op",
            module="endpoints_to_5d",
            spec="仅当输入是 4D",
            shape_in="(B, M, K, 4)",
            shape_out=shape_k,
            formula="[cᵤ, cᵥ, ‖Δ‖, sin 2θ, cos 2θ]",
            source="models/object_matcher/xfeat_line_matcher.py:75",
            meaning=(
                "线段没有方向：端点对调会让 θ 差 180°，但 sin(2θ) 和 cos(2θ) 不变。"
                "所以网络吃的是中心、长度和二倍角，而不是有向端点。长度用切片归一化坐标，不随原图分辨率变化。"
            ),
        ),
        graph.node(
            id="line_fc1",
            op="Linear",
            kind="linear",
            module="line_encoder.line_mlp.0",
            spec=f"5 → 64",
            shape_in=shape_k,
            shape_out="(B, M, K, 64)",
            weight="W 64×5 + bias",
            params=_linear_params(5, 64),
            formula="Y = XWᵀ + b",
            source="models/object_matcher/xfeat_line_matcher.py:56",
            meaning="每根直线独立过同一层线性变换，K 根线之间还没有交互。这一层把 5 个数抬到可以做归一化的宽度。",
        ),
        graph.node(
            id="line_ln1",
            op="LayerNorm",
            kind="norm",
            module="line_encoder.line_mlp.1",
            spec="64",
            shape_in="(B, M, K, 64)",
            shape_out="(B, M, K, 64)",
            weight="γ, β 各 64",
            params=_ln_params(64),
            formula="对 64 维描述子归一化",
            source="models/object_matcher/xfeat_line_matcher.py:57",
            meaning="按单根直线的 64 维做归一化，而不是在 K 根线之间做。这样一根很长的线和一根很短的线进入 ReLU 之前幅度可比。",
        ),
        graph.node(
            id="line_relu",
            op="ReLU",
            kind="act",
            module="line_encoder.line_mlp.2",
            spec="inplace",
            shape_in="(B, M, K, 64)",
            shape_out="(B, M, K, 64)",
            formula="max(0, X)",
            source="models/object_matcher/xfeat_line_matcher.py:58",
            meaning="直线 MLP 里唯一的非线性，夹在两层线性中间。第二层线性之后只有 LayerNorm，没有再接 ReLU。",
        ),
        graph.node(
            id="line_fc2",
            op="Linear",
            kind="linear",
            module="line_encoder.line_mlp.3",
            spec=f"64 → {line_dim}",
            shape_in="(B, M, K, 64)",
            shape_out=shape_h,
            weight=f"W {line_dim}×64 + bias",
            params=_linear_params(64, line_dim),
            formula="Y = XWᵀ + b",
            source="models/object_matcher/xfeat_line_matcher.py:59",
            meaning=f"把每根直线映射成 {line_dim} 维。到这里 K 根线仍然是一个集合，顺序没有意义。",
        ),
        graph.node(
            id="line_ln2",
            op="LayerNorm",
            kind="norm",
            module="line_encoder.line_mlp.4",
            spec=f"{line_dim}",
            shape_in=shape_h,
            shape_out=shape_h,
            weight=f"γ, β 各 {line_dim}",
            params=_ln_params(line_dim),
            formula="对最后一维归一化",
            source="models/object_matcher/xfeat_line_matcher.py:60",
            meaning="line_mlp 的最后一层是 LayerNorm，不是 ReLU。池化之前的直线特征因此可以带负值，Max 池化才能在正负方向上挑选显著直线。",
        ),
        graph.node(
            id="line_max",
            op="Max",
            kind="pool",
            module="line_encoder.forward",
            spec="沿 K · 屏蔽空槽",
            shape_in=shape_h,
            shape_out=shape_p,
            formula="max(h + (−1e4)·¬mask)",
            source="models/object_matcher/xfeat_line_matcher.py:133",
            meaning=(
                "DeepSets 的最大池化。空槽被填成 −1e4 而不是 −1e9，是为了待在 FP16 能表示的范围内。"
                "Max 保留切片里最显著的那根线，比如门框的长边。"
            ),
        ),
        graph.node(
            id="line_mean",
            op="Mean",
            kind="pool",
            module="line_encoder.forward",
            spec="沿 K · 按有效根数",
            shape_in=shape_h,
            shape_out=shape_p,
            formula="Σ(h · mask) / count",
            source="models/object_matcher/xfeat_line_matcher.py:139",
            meaning="平均池化保留整个直线集合的统计，而不只是最长的一根。分母是有效直线数，空槽不加进来，也不会把特征稀释成零。",
        ),
        graph.node(
            id="line_cat",
            op="Concat",
            kind="merge",
            module="line_encoder.forward",
            spec="max ‖ mean",
            shape_in=shape_p,
            shape_out=f"(B, M, {2 * line_dim})",
            formula="[max ‖ mean]",
            source="models/object_matcher/xfeat_line_matcher.py:141",
            meaning="最大池化和平均池化拼在一起，既保留最强直线，也保留集合的平均几何。这是置换不变的，直线列表怎么排序都得到同一个向量。",
        ),
        graph.node(
            id="line_pool_fc",
            op="Linear",
            kind="linear",
            module="line_encoder.pool_proj.0",
            spec=f"{2 * line_dim} → {line_dim}",
            shape_in=f"(B, M, {2 * line_dim})",
            shape_out=shape_p,
            weight=f"W {line_dim}×{2 * line_dim} + bias",
            params=_linear_params(2 * line_dim, line_dim),
            formula="Y = XWᵀ + b",
            source="models/object_matcher/xfeat_line_matcher.py:65",
            meaning="把两路池化压回直线 token 的宽度，才能和视觉 token 拼接。每个切片此时只剩一个直线向量。",
        ),
        graph.node(
            id="line_pool_ln",
            op="LayerNorm",
            kind="norm",
            module="line_encoder.pool_proj.1",
            spec=f"{line_dim}",
            shape_in=shape_p,
            shape_out=shape_p,
            weight=f"γ, β 各 {line_dim}",
            params=_ln_params(line_dim),
            formula="对直线 token 归一化",
            source="models/object_matcher/xfeat_line_matcher.py:66",
            meaning="池化投影后的归一化，让“线很多的柜子”和“只有一条边的门”在幅度上可以和视觉分支比较。",
        ),
        graph.node(
            id="line_pool_relu",
            op="ReLU",
            kind="act",
            module="line_encoder.pool_proj.2",
            spec="inplace",
            shape_in=shape_p,
            shape_out=shape_p,
            formula="max(0, X)",
            source="models/object_matcher/xfeat_line_matcher.py:67",
            meaning="直线分支的最后一次非线性。没有直线的切片不会走这条路径，它们在下一步被可学习的空 token 替换。",
        ),
        graph.node(
            id="line_where",
            op="Where",
            kind="op",
            module="line_encoder.null_line_token",
            spec="无直线则换空 token",
            shape_in=shape_p,
            shape_out=shape_p,
            weight=f"null_line_token ∈ ℝ^{line_dim}",
            params=line_dim,
            formula="has_lines ? pooled : null",
            source="models/object_matcher/xfeat_line_matcher.py:146",
            meaning=(
                "白墙上的区域可能一根线都检不出来。可学习的 null_line_token 占住这一维，"
                "避免用全零向量把“没有线”和“线特征崩成零”混成同一件事。空 token 对所有无直线切片共享。"
            ),
        ),
    ]
    graph.chain(ids[:8])
    graph.edge(ids[6], ids[8], "data", shape_h)
    graph.edge(ids[7], ids[9], "data", shape_p)
    graph.edge(ids[8], ids[9], "data", shape_p)
    graph.chain(ids[9:])
    return ids


def _geo_encoder(graph: _Graph, out_dim: int) -> list[str]:
    shape = f"(B, M, {out_dim})"
    ids = [
        graph.node(
            id="geo_in",
            op="Input",
            kind="input",
            module="ObjectGeometryEncoder",
            spec="像素框",
            shape_in="(B, M, 4)",
            shape_out="(B, M, 4)",
            formula="[x1, y1, x2, y2]，原图像素坐标",
            source="models/object_matcher/xfeat_object_matcher.py:200",
            meaning="边界框还不是网络里的 4 维嵌入。编码器会用图像高宽把它展开成与分辨率无关的布局向量。",
        ),
        graph.node(
            id="geo_9d",
            op="Layout9D",
            kind="op",
            module="ObjectGeometryEncoder.forward",
            spec="4 个角点 → 9 维",
            shape_in="(B, M, 4)",
            shape_out="(B, M, 9)",
            formula="[x1/W, y1/H, x2/W, y2/H, cx, cy, w, h, area]",
            source="models/object_matcher/xfeat_object_matcher.py:212",
            meaning=(
                "代码实际构造的是 9 维，不是 4 维：两个角点、中心、宽高和面积，全部除以图像尺寸。"
                "面积项让“小开关”和“整面墙”在几何分支上就分开，不必只靠外观。"
            ),
        ),
        graph.node(
            id="geo_fc1",
            op="Linear",
            kind="linear",
            module="geo_encoder.mlp.0",
            spec="9 → 64",
            shape_in="(B, M, 9)",
            shape_out="(B, M, 64)",
            weight="W 64×9 + bias",
            params=_linear_params(9, 64),
            formula="Y = XWᵀ + b",
            source="models/object_matcher/xfeat_object_matcher.py:194",
            meaning="布局 MLP 的第一层。每个目标独立编码，这一层还不看别的目标，相对位置要等后面的自注意力。",
        ),
        graph.node(
            id="geo_relu",
            op="ReLU",
            kind="act",
            module="geo_encoder.mlp.1",
            spec="inplace",
            shape_in="(B, M, 64)",
            shape_out="(B, M, 64)",
            formula="max(0, X)",
            source="models/object_matcher/xfeat_object_matcher.py:195",
            meaning="布局编码的非线性。第二层线性之后只有 LayerNorm，没有第二个 ReLU。",
        ),
        graph.node(
            id="geo_fc2",
            op="Linear",
            kind="linear",
            module="geo_encoder.mlp.2",
            spec=f"64 → {out_dim}",
            shape_in="(B, M, 64)",
            shape_out=shape,
            weight=f"W {out_dim}×64 + bias",
            params=_linear_params(64, out_dim),
            formula="Y = XWᵀ + b",
            source="models/object_matcher/xfeat_object_matcher.py:196",
            meaning=f"压成 {out_dim} 维布局 token，与直线 token 同宽，方便和视觉特征拼接。",
        ),
        graph.node(
            id="geo_ln",
            op="LayerNorm",
            kind="norm",
            module="geo_encoder.mlp.3",
            spec=f"{out_dim}",
            shape_in=shape,
            shape_out=shape,
            weight=f"γ, β 各 {out_dim}",
            params=_ln_params(out_dim),
            formula="对布局 token 归一化",
            source="models/object_matcher/xfeat_object_matcher.py:197",
            meaning="让框的几何向量和卷积池化出来的视觉向量处于相近的幅度，否则拼接后线性层会偏向数值更大的那一路。",
        ),
    ]
    graph.chain(ids)
    return ids


def _visual_tower(graph: _Graph, use_pyramid: bool, feat_dim: int) -> str:
    """Shared CNN. Returns the id of the visual token (B, M, C)."""
    graph.node(
        id="vis_in",
        op="Input",
        kind="input",
        module="extract_visual_tokens",
        spec="区域切片",
        shape_in="(B, M, 3, 112, 112)",
        shape_out="(N, 3, 112, 112)",
        formula="N = B·M，view 成一批独立切片",
        source="models/object_matcher/xfeat_line_matcher.py:367",
        meaning=(
            "匹配的视觉单元是目标切片，不是整幅图。进入卷积前被展成 N=B·M 张 112×112 的 RGB。"
            "骨干在切片之间不通信；跨目标、跨视角的关系全部留给后面的注意力。"
        ),
    )
    stage1, c1 = _down_stage(
        graph, "s1", 3, 32, "112×112", "56×56",
        "visual_backbone.stage1",
        "Stride 2，P1 的底。",
    )
    stage2, c2 = _down_stage(
        graph, "s2", 32, 64, "56×56", "28×28",
        "visual_backbone.stage2",
        "Stride 4，部件尺度。",
    )
    stage3, c3 = _down_stage(
        graph, "s3", 64, 128, "28×28", "14×14",
        "visual_backbone.stage3",
        "Stride 8，切片级语义。",
    )
    graph.edge("vis_in", stage1[0], "data", "(N, 3, 112, 112)")
    graph.edge(c1, stage2[0], "data", graph.get(c1)["shape_out"])
    graph.edge(c2, stage3[0], "data", graph.get(c2)["shape_out"])
    graph.place(["vis_in"] + stage1 + stage2 + stage3, 0, 0)

    if not use_pyramid:
        shape64 = "(N, 64, 14×14)"
        tail = [
            graph.node(
                id="s4_conv",
                op="Conv2d",
                kind="conv",
                module="visual_backbone.stage4.0",
                spec="1×1 · 128→64 · 无偏置",
                shape_in="(N, 128, 14×14)",
                shape_out=shape64,
                weight="W 64×128×1×1",
                params=_conv_params(128, 64, 1, bias=False),
                formula="Y = W ∗ X",
                source="models/object_matcher/xfeat_object_matcher.py:70",
                meaning="单尺度骨干的通道投影。不做 FPN，只保留 stride 8 的一张特征图，细边缘和粗语义挤在同一张 14×14 里。",
            ),
            graph.node(
                id="s4_bn",
                op="BatchNorm2d",
                kind="norm",
                module="visual_backbone.stage4.1",
                spec="64",
                shape_in=shape64,
                shape_out=shape64,
                weight="γ, β 各 64",
                params=_bn_params(64),
                formula="(X − μ) / √(σ²+ε) · γ + β",
                source="models/object_matcher/xfeat_object_matcher.py:71",
                meaning="1×1 投影后的归一化。",
            ),
            graph.node(
                id="s4_relu",
                op="ReLU",
                kind="act",
                module="visual_backbone.stage4.2",
                spec="inplace",
                shape_in=shape64,
                shape_out=shape64,
                formula="max(0, X)",
                source="models/object_matcher/xfeat_object_matcher.py:72",
                meaning="单尺度特征图的最后一次非线性，随后被池化成一个向量。",
            ),
            graph.node(
                id="pool",
                op="AdaptiveAvgPool2d",
                kind="pool",
                module="visual_pool",
                spec="输出 1×1",
                shape_in=shape64,
                shape_out=f"(B, M, {feat_dim})",
                formula="空间平均后 view 回 (B, M, C)",
                source="models/object_matcher/xfeat_line_matcher.py:376",
                meaning="把 14×14 收成一个向量。单尺度模式下没有 P1/P2 的细节支路，小结构只能靠这个向量里还没被平均掉的部分。",
            ),
        ]
        graph.chain(tail)
        start = graph.get(c3)["row"] + 1
        graph.place(tail, 0, start)
        graph.edge(c3, tail[0], "data", graph.get(c3)["shape_out"])
        return tail[-1]

    lat3 = graph.node(
        id="lat3",
        op="Conv2d",
        kind="conv",
        module="visual_backbone.lat3",
        spec=f"1×1 · 128→{feat_dim}",
        shape_in="(N, 128, 14×14)",
        shape_out=f"(N, {feat_dim}, 14×14)",
        weight=f"W {feat_dim}×128×1×1 + bias",
        params=_conv_params(128, feat_dim, 1, bias=True),
        formula="P3 = lat3(C3)",
        source="models/object_matcher/xfeat_object_matcher.py:120",
        meaning=(
            "FPN 的最粗一层。1×1 只改通道、不改 14×14 的网格，把 stride 8 的语义对齐到统一的 "
            f"{feat_dim} 维。P3 没有 3×3 平滑，因为它已经是金字塔的顶端。"
        ),
    )
    up_p3 = graph.node(
        id="up_p3",
        op="Interpolate",
        kind="op",
        module="F.interpolate",
        spec="nearest · 14 → 28",
        shape_in=f"(N, {feat_dim}, 14×14)",
        shape_out=f"(N, {feat_dim}, 28×28)",
        formula="nearest upsample 到 C2 的高宽",
        source="models/object_matcher/xfeat_object_matcher.py:142",
        meaning=(
            "最近邻上采样把 P3 的语义网格复制到 P2 的分辨率，没有可学习参数。"
            "复制会留下棋盘状台阶，所以下一步相加之后必须接 3×3 平滑，而不是直接当特征用。"
        ),
    )
    lat2 = graph.node(
        id="lat2",
        op="Conv2d",
        kind="conv",
        module="visual_backbone.lat2",
        spec=f"1×1 · 64→{feat_dim}",
        shape_in="(N, 64, 28×28)",
        shape_out=f"(N, {feat_dim}, 28×28)",
        weight=f"W {feat_dim}×64×1×1 + bias",
        params=_conv_params(64, feat_dim, 1, bias=True),
        formula="lat2(C2)",
        source="models/object_matcher/xfeat_object_matcher.py:121",
        meaning="把 stride 4 的 C2 投到和上采样后的 P3 相同的通道数，否则两者不能相加。这一层保留部件轮廓，语义来自另一路。",
    )
    add_p2 = graph.node(
        id="add_p2",
        op="Add",
        kind="merge",
        module="XFeatPyramidBackbone.forward",
        spec="横向 + 自顶向下",
        shape_in=f"(N, {feat_dim}, 28×28)",
        shape_out=f"(N, {feat_dim}, 28×28)",
        formula="lat2(C2) + ↑P3",
        source="models/object_matcher/xfeat_object_matcher.py:142",
        meaning="同一空间位置上，局部部件特征和粗语义相加。这是 FPN 的核心：细网格拿到了深一层才有的类别信息。",
    )
    sm2 = graph.node(
        id="sm2_conv",
        op="Conv2d",
        kind="conv",
        module="visual_backbone.smooth2.0",
        spec=f"3×3 · {feat_dim}→{feat_dim} · 无偏置",
        shape_in=f"(N, {feat_dim}, 28×28)",
        shape_out=f"(N, {feat_dim}, 28×28)",
        weight=f"W {feat_dim}×{feat_dim}×3×3",
        params=_conv_params(feat_dim, feat_dim, 3, bias=False),
        formula="Y = W ∗ X，padding=1",
        source="models/object_matcher/xfeat_object_matcher.py:126",
        meaning="平滑最近邻上采样留下的块状伪影，并让相加后的两路特征真正混合。无偏置，紧跟 BatchNorm。",
    )
    sm2_bn = graph.node(
        id="sm2_bn",
        op="BatchNorm2d",
        kind="norm",
        module="visual_backbone.smooth2.1",
        spec=f"{feat_dim}",
        shape_in=f"(N, {feat_dim}, 28×28)",
        shape_out=f"(N, {feat_dim}, 28×28)",
        weight=f"γ, β 各 {feat_dim}",
        params=_bn_params(feat_dim),
        formula="(X − μ) / √(σ²+ε) · γ + β",
        source="models/object_matcher/xfeat_object_matcher.py:127",
        meaning="P2 平滑卷积后的归一化。",
    )
    sm2_relu = graph.node(
        id="sm2_relu",
        op="ReLU",
        kind="act",
        module="visual_backbone.smooth2.2",
        spec="inplace · 输出 P2",
        shape_in=f"(N, {feat_dim}, 28×28)",
        shape_out=f"(N, {feat_dim}, 28×28)",
        formula="P2 = ReLU(BN(smooth2(lat2(C2) + ↑P3)))",
        source="models/object_matcher/xfeat_object_matcher.py:142",
        meaning="这就是 P2。28×28、64 维，对应切片里的部件几何。它还会被上采样，去补充更细的 P1。",
    )
    lat1 = graph.node(
        id="lat1",
        op="Conv2d",
        kind="conv",
        module="visual_backbone.lat1",
        spec=f"1×1 · 32→{feat_dim}",
        shape_in="(N, 32, 56×56)",
        shape_out=f"(N, {feat_dim}, 56×56)",
        weight=f"W {feat_dim}×32×1×1 + bias",
        params=_conv_params(32, feat_dim, 1, bias=True),
        formula="lat1(C1)",
        source="models/object_matcher/xfeat_object_matcher.py:122",
        meaning="最细的 C1 只有 32 通道。1×1 把它对齐到 64 维，好和上采样的 P2 相加。这里还留着 stride 2 的边缘。",
    )
    up_p2 = graph.node(
        id="up_p2",
        op="Interpolate",
        kind="op",
        module="F.interpolate",
        spec="nearest · 28 → 56",
        shape_in=f"(N, {feat_dim}, 28×28)",
        shape_out=f"(N, {feat_dim}, 56×56)",
        formula="nearest upsample 到 C1 的高宽",
        source="models/object_matcher/xfeat_object_matcher.py:143",
        meaning="把 P2 复制到 56×56。同样是最近邻、无参数，块状伪影交给 smooth1 的 3×3 处理。",
    )
    add_p1 = graph.node(
        id="add_p1",
        op="Add",
        kind="merge",
        module="XFeatPyramidBackbone.forward",
        spec="细边缘 + 部件语义",
        shape_in=f"(N, {feat_dim}, 56×56)",
        shape_out=f"(N, {feat_dim}, 56×56)",
        formula="lat1(C1) + ↑P2",
        source="models/object_matcher/xfeat_object_matcher.py:143",
        meaning="56×56 的细网格加上来自 P2 的语义。弱纹理墙面上，真正可区分的往往是这些细边缘，而不是 P3 的粗类别。",
    )
    sm1 = graph.node(
        id="sm1_conv",
        op="Conv2d",
        kind="conv",
        module="visual_backbone.smooth1.0",
        spec=f"3×3 · {feat_dim}→{feat_dim} · 无偏置",
        shape_in=f"(N, {feat_dim}, 56×56)",
        shape_out=f"(N, {feat_dim}, 56×56)",
        weight=f"W {feat_dim}×{feat_dim}×3×3",
        params=_conv_params(feat_dim, feat_dim, 3, bias=False),
        formula="Y = W ∗ X，padding=1",
        source="models/object_matcher/xfeat_object_matcher.py:131",
        meaning="P1 的抗混叠平滑，作用和 smooth2 相同，权重不共享。",
    )
    sm1_bn = graph.node(
        id="sm1_bn",
        op="BatchNorm2d",
        kind="norm",
        module="visual_backbone.smooth1.1",
        spec=f"{feat_dim}",
        shape_in=f"(N, {feat_dim}, 56×56)",
        shape_out=f"(N, {feat_dim}, 56×56)",
        weight=f"γ, β 各 {feat_dim}",
        params=_bn_params(feat_dim),
        formula="(X − μ) / √(σ²+ε) · γ + β",
        source="models/object_matcher/xfeat_object_matcher.py:132",
        meaning="P1 平滑后的归一化。",
    )
    sm1_relu = graph.node(
        id="sm1_relu",
        op="ReLU",
        kind="act",
        module="visual_backbone.smooth1.2",
        spec="inplace · 输出 P1",
        shape_in=f"(N, {feat_dim}, 56×56)",
        shape_out=f"(N, {feat_dim}, 56×56)",
        formula="P1 = ReLU(BN(smooth1(lat1(C1) + ↑P2)))",
        source="models/object_matcher/xfeat_object_matcher.py:143",
        meaning="这就是 P1。56×56、64 维，三个尺度里分辨率最高的一张，池化后成为视觉 token 的前 64 维。",
    )
    pool_ids = []
    for name, src_shape, note in (
        ("pool_p1", f"(N, {feat_dim}, 56×56)", "P1 细边缘被平均成 64 维。空间细节没有了，但通道里仍保留边缘统计。"),
        ("pool_p2", f"(N, {feat_dim}, 28×28)", "P2 部件形状收成第二个 64 维。"),
        ("pool_p3", f"(N, {feat_dim}, 14×14)", "P3 粗语义收成第三个 64 维。三路用的是同一个 AdaptiveAvgPool2d，没有参数，也不共享仿射。"),
    ):
        pool_ids.append(graph.node(
            id=name,
            op="AdaptiveAvgPool2d",
            kind="pool",
            module="visual_pool",
            spec="1×1",
            shape_in=src_shape,
            shape_out=f"(B, M, {feat_dim})",
            formula="空间平均，再 view 成 (B, M, C)",
            source="models/object_matcher/xfeat_line_matcher.py:370",
            meaning=note + " 参数量为 0：三次调用是同一个模块。",
            params=0,
        ))
    cat = graph.node(
        id="vis_cat",
        op="Concat",
        kind="merge",
        module="extract_visual_tokens",
        spec="P1 ‖ P2 ‖ P3",
        shape_in=f"(B, M, {feat_dim})",
        shape_out=f"(B, M, {3 * feat_dim})",
        formula=f"[{feat_dim} ‖ {feat_dim} ‖ {feat_dim}] → {3 * feat_dim}",
        source="models/object_matcher/xfeat_line_matcher.py:373",
        meaning=(
            f"三个尺度拼成 {3 * feat_dim} 维视觉 token。拼接而不是相加，是为了让后面的线性层自己决定细边缘和粗语义各占多少，"
            "而不是在池化前就把它们混成一个向量。"
        ),
    )
    fpn_ids = [lat3, up_p3, lat2, add_p2, sm2, sm2_bn, sm2_relu, lat1, up_p2, add_p1, sm1, sm1_bn, sm1_relu, *pool_ids, cat]
    c3_row = graph.get(c3)["row"]
    graph.place(fpn_ids, 1, c3_row)
    graph.edge(c3, lat3, "data", graph.get(c3)["shape_out"])
    graph.edge(lat3, up_p3, "data", graph.get(lat3)["shape_out"])
    graph.edge(c2, lat2, "lateral", graph.get(c2)["shape_out"])
    graph.edge(lat2, add_p2, "data", graph.get(lat2)["shape_out"])
    graph.edge(up_p3, add_p2, "data", graph.get(up_p3)["shape_out"])
    graph.chain([add_p2, sm2, sm2_bn, sm2_relu])
    graph.edge(c1, lat1, "lateral", graph.get(c1)["shape_out"])
    graph.edge(sm2_relu, up_p2, "data", graph.get(sm2_relu)["shape_out"])
    graph.edge(lat1, add_p1, "data", graph.get(lat1)["shape_out"])
    graph.edge(up_p2, add_p1, "data", graph.get(up_p2)["shape_out"])
    graph.chain([add_p1, sm1, sm1_bn, sm1_relu])
    graph.edge(sm1_relu, pool_ids[0], "data", graph.get(sm1_relu)["shape_out"])
    graph.edge(sm2_relu, pool_ids[1], "data", graph.get(sm2_relu)["shape_out"])
    graph.edge(lat3, pool_ids[2], "data", graph.get(lat3)["shape_out"])
    for pid in pool_ids:
        graph.edge(pid, cat, "data", graph.get(pid)["shape_out"])
    return cat


def _fuse_and_match(
    graph: _Graph,
    sources: list[tuple[str, str]],
    in_dim: int,
    d_model: int,
    num_layers: int,
    nhead: int,
    fuse_module: str,
    fuse_meaning: str,
) -> None:
    parts = " ‖ ".join(name for _, name in sources)
    fused_in = graph.node(
        id="fuse_cat",
        op="Concat",
        kind="merge",
        module=fuse_module,
        spec=parts,
        shape_in="见各支路",
        shape_out=f"(B, M, {in_dim})",
        formula=parts,
        source="models/object_matcher/xfeat_line_matcher.py:419",
        meaning=fuse_meaning,
    )
    proj = graph.node(
        id="fuse_fc",
        op="Linear",
        kind="linear",
        module=f"{fuse_module}.0" if "fusion" in fuse_module or fuse_module.endswith("proj") else f"{fuse_module}.0",
        spec=f"{in_dim} → {d_model}",
        shape_in=f"(B, M, {in_dim})",
        shape_out=f"(B, M, {d_model})",
        weight=f"W {d_model}×{in_dim} + bias",
        params=_linear_params(in_dim, d_model),
        formula="Y = XWᵀ + b",
        source="models/object_matcher/xfeat_line_matcher.py:348",
        meaning=(
            f"多模态拼接后的唯一一次降维，得到图网络使用的 token。宽度 {d_model} 必须能被头数 {nhead} 整除。"
            "从这一层开始，视觉、直线和布局不再分通道，只剩一个向量。"
        ),
    )
    ln = graph.node(
        id="fuse_ln",
        op="LayerNorm",
        kind="norm",
        module=f"{fuse_module}.1",
        spec=f"{d_model}",
        shape_in=f"(B, M, {d_model})",
        shape_out=f"(B, M, {d_model})",
        weight=f"γ, β 各 {d_model}",
        params=_ln_params(d_model),
        formula="对 token 最后一维归一化",
        source="models/object_matcher/xfeat_line_matcher.py:349",
        meaning="进入注意力之前把 token 幅度拉齐。否则某一模态数值偏大时，点积会被它主导。",
    )
    relu = graph.node(
        id="fuse_relu",
        op="ReLU",
        kind="act",
        module=f"{fuse_module}.2",
        spec="inplace",
        shape_in=f"(B, M, {d_model})",
        shape_out=f"(B, M, {d_model})",
        formula="max(0, X)",
        source="models/object_matcher/xfeat_line_matcher.py:350",
        meaning=(
            "融合投影的最后一层非线性。这个输出会复制到两个视角列：编码器权重是共享的，"
            "但对 view 0 和 view 1 各前向一次。从下一层起，self0 与 self1 的权重不再共享。"
        ),
    )
    anchor = max(graph.get(src)["row"] for src, _ in sources)
    col = max(graph.get(src)["col"] for src, _ in sources) + 1
    graph.place([fused_in, proj, ln, relu], col, anchor)
    for src, _name in sources:
        graph.edge(src, fused_in, "data", graph.get(src)["shape_out"])
    graph.chain([fused_in, proj, ln, relu])
    t0, t1 = _object_transformer(
        graph, d_model, num_layers, nhead,
        col0=col, col1=col + 1, start_row=anchor + 5, src=relu,
    )
    sink_row = max(graph.get(t0)["row"], graph.get(t1)["row"]) + 2
    _sinkhorn_head(graph, d_model, col, sink_row, t0, t1)


def build_xfeat_line_graph(use_pyramid: bool = True, d_model: int = 128, num_layers: int = 2, nhead: int = 4) -> dict[str, Any]:
    feat_dim = 64
    line_dim = 64
    graph = _Graph()
    vis = _visual_tower(graph, use_pyramid=use_pyramid, feat_dim=feat_dim)
    lines = _line_encoder(graph, line_dim)
    geos = _geo_encoder(graph, feat_dim)
    vis_row = graph.get(vis)["row"]
    graph.place(lines, 2, vis_row - len(lines) + 1)
    graph.place(geos, 3, vis_row - len(geos) + 1)
    vis_dim = 3 * feat_dim if use_pyramid else feat_dim
    in_dim = vis_dim + line_dim + feat_dim
    _fuse_and_match(
        graph,
        [(vis, f"视觉 {vis_dim}D"), (lines[-1], f"直线 {line_dim}D"), (geos[-1], f"布局 {feat_dim}D")],
        in_dim=in_dim,
        d_model=d_model,
        num_layers=num_layers,
        nhead=nhead,
        fuse_module="fusion_proj",
        fuse_meaning=(
            "把视觉 token、直线 token 和 9 维布局 token 在通道维拼起来。"
            "三路缺一不可：白墙靠直线，重复家具靠外观，出画与否还要看框在画面中的位置。"
        ),
    )
    mode = "FPN P1+P2+P3" if use_pyramid else "单尺度 stride 8"
    return _pack(
        graph,
        title=f"XFeatLineMatcher · {mode}",
        assumption=(
            f"形状按代码默认：切片 112×112，N=B·M，直线槽 K，d_model={d_model}，注意力 {num_layers} 层 × {nhead} 头。"
            "视觉骨干、直线编码、布局编码在两个视角上权重共享，图中只展开一次；交叉注意力才交换两个视角。"
        ),
        jumps=_jumps(graph, [
            ("输入切片", "vis_in"),
            ("下采样", "s1_conv"),
            ("FPN" if use_pyramid else "单尺度池化", "lat3" if use_pyramid else "pool"),
            ("直线 5D", "to5d"),
            ("布局 9D", "geo_9d"),
            ("融合", "fuse_cat"),
            ("自注意力", "l0_self0_mha"),
            ("交叉注意力", "l0_cross0_mha"),
            ("垃圾桶", "dustbin"),
            ("Sinkhorn", "sinkhorn"),
        ]),
    )


def build_xfeat_object_graph(use_pyramid: bool = True, d_model: int = 128, num_layers: int = 2, nhead: int = 4) -> dict[str, Any]:
    feat_dim = 64
    graph = _Graph()
    vis = _visual_tower(graph, use_pyramid=use_pyramid, feat_dim=feat_dim)
    # Object matcher names the backbone `backbone` and the projection `token_proj`.
    for node in graph.nodes:
        node["module"] = node["module"].replace("visual_backbone", "backbone").replace("visual_pool", "pool")
        if "xfeat_line_matcher.py" in node.get("source", ""):
            node["source"] = node["source"].replace(
                "models/object_matcher/xfeat_line_matcher.py:367",
                "models/object_matcher/xfeat_object_matcher.py:387",
            ).replace(
                "models/object_matcher/xfeat_line_matcher.py:370",
                "models/object_matcher/xfeat_object_matcher.py:390",
            ).replace(
                "models/object_matcher/xfeat_line_matcher.py:376",
                "models/object_matcher/xfeat_object_matcher.py:396",
            ).replace(
                "models/object_matcher/xfeat_line_matcher.py:373",
                "models/object_matcher/xfeat_object_matcher.py:393",
            )
    vis_dim = 3 * feat_dim if use_pyramid else feat_dim
    _fuse_and_match(
        graph,
        [(vis, f"视觉 {vis_dim}D")],
        in_dim=vis_dim,
        d_model=d_model,
        num_layers=num_layers,
        nhead=nhead,
        fuse_module="token_proj",
        fuse_meaning=(
            "XFeatObjectMatcher 默认 with_geometry=False，边界框不进入网络，这里只有视觉 token。"
            "构造时若打开 with_geometry，会在这一拼接上再接 64 维 ObjectGeometryEncoder。"
        ),
    )
    # Retarget fusion source lines to the object matcher file.
    for node_id in ("fuse_cat", "fuse_fc", "fuse_ln", "fuse_relu"):
        node = graph.get(node_id)
        node["source"] = "models/object_matcher/xfeat_object_matcher.py:364"
    return _pack(
        graph,
        title="XFeatObjectMatcher · 默认不含布局分支",
        assumption=(
            f"类默认 use_pyramid={use_pyramid}、with_geometry=False。切片 112×112，d_model={d_model}。"
            "图中的卷积和注意力与直线模型共用同一套模块定义，但没有直线编码，融合层是 token_proj。"
        ),
        jumps=_jumps(graph, [
            ("输入切片", "vis_in"),
            ("FPN" if use_pyramid else "池化", "lat3" if use_pyramid else "pool"),
            ("投影", "fuse_fc"),
            ("自注意力", "l0_self0_mha"),
            ("交叉注意力", "l0_cross0_mha"),
            ("Sinkhorn", "sinkhorn"),
        ]),
    )


def _bottleneck(
    graph: _Graph,
    prefix: str,
    cin: int,
    mid: int,
    cout: int,
    stride: int,
    h_in: int,
    w_in: int,
    module: str,
    frozen: bool,
    src_id: str,
) -> tuple[list[str], int, int]:
    h_out = h_in // stride
    w_out = w_in // stride
    spatial_in = f"{h_in}×{w_in}"
    spatial_mid = f"{h_out}×{w_out}" if stride != 1 else spatial_in
    spatial_out = f"{h_out}×{w_out}"
    shape_in = f"(B, {cin}, {spatial_in})"
    shape_mid = f"(B, {mid}, {spatial_mid})"
    shape_out = f"(B, {cout}, {spatial_out})"

    def mark(fields: dict[str, Any]) -> dict[str, Any]:
        fields["frozen"] = frozen
        return fields

    ids = [
        graph.node(**mark(dict(
            id=f"{prefix}_conv1",
            op="Conv2d",
            kind="conv",
            module=f"{module}.conv1",
            spec=f"1×1 · {cin}→{mid} · stride 1",
            shape_in=shape_in,
            shape_out=f"(B, {mid}, {spatial_in})",
            weight=f"W {mid}×{cin}×1×1 · 无偏置",
            params=_conv_params(cin, mid, 1, bias=False),
            formula="Y = W ∗ X",
            source="torchvision.models.resnet.Bottleneck",
            meaning=(
                f"瓶颈块 {prefix} 的降维 1×1。先把 {cin} 通道压到 {mid}，后面的 3×3 就不用在全通道上做。"
                "这是 ResNet-50 的标准瓶颈，不是本仓库自己写的 BasicResidualBlock。"
            ),
        ))),
        graph.node(**mark(dict(
            id=f"{prefix}_bn1",
            op="BatchNorm2d",
            kind="norm",
            module=f"{module}.bn1",
            spec=f"{mid}",
            shape_in=f"(B, {mid}, {spatial_in})",
            shape_out=f"(B, {mid}, {spatial_in})",
            weight=f"γ, β 各 {mid}",
            params=_bn_params(mid),
            formula="(X − μ) / √(σ²+ε) · γ + β",
            source="torchvision.models.resnet.Bottleneck",
            meaning="1×1 降维后的 BatchNorm。ReLU 在它后面，3×3 之前。",
        ))),
        graph.node(**mark(dict(
            id=f"{prefix}_relu1",
            op="ReLU",
            kind="act",
            module=f"{module}.relu",
            spec="inplace · 块内复用",
            shape_in=f"(B, {mid}, {spatial_in})",
            shape_out=f"(B, {mid}, {spatial_in})",
            formula="max(0, X)",
            source="torchvision.models.resnet.Bottleneck",
            meaning="Bottleneck 里同一个 ReLU 模块被用三次。这一次位于 3×3 之前。",
        ))),
        graph.node(**mark(dict(
            id=f"{prefix}_conv2",
            op="Conv2d",
            kind="conv",
            module=f"{module}.conv2",
            spec=f"3×3 · {mid}→{mid} · stride {stride}",
            shape_in=f"(B, {mid}, {spatial_in})",
            shape_out=shape_mid,
            weight=f"W {mid}×{mid}×3×3 · 无偏置",
            params=_conv_params(mid, mid, 3, bias=False),
            formula=f"Y = W ∗ X，stride={stride}，padding=1",
            source="torchvision.models.resnet.Bottleneck",
            meaning=(
                "空间混合发生在窄通道上。ResNet 把下采样步长放在这个 3×3 上，而不是放在第一个 1×1。"
                + ("这一块把特征图高宽减半。" if stride == 2 else "这一块保持分辨率。")
            ),
        ))),
        graph.node(**mark(dict(
            id=f"{prefix}_bn2",
            op="BatchNorm2d",
            kind="norm",
            module=f"{module}.bn2",
            spec=f"{mid}",
            shape_in=shape_mid,
            shape_out=shape_mid,
            weight=f"γ, β 各 {mid}",
            params=_bn_params(mid),
            formula="(X − μ) / √(σ²+ε) · γ + β",
            source="torchvision.models.resnet.Bottleneck",
            meaning="3×3 之后的 BatchNorm，后面还有一次 ReLU，然后才是升维 1×1。",
        ))),
        graph.node(**mark(dict(
            id=f"{prefix}_relu2",
            op="ReLU",
            kind="act",
            module=f"{module}.relu",
            spec="inplace",
            shape_in=shape_mid,
            shape_out=shape_mid,
            formula="max(0, X)",
            source="torchvision.models.resnet.Bottleneck",
            meaning="3×3 与升维 1×1 之间的 ReLU。升维卷积看到的已经是非负特征。",
        ))),
        graph.node(**mark(dict(
            id=f"{prefix}_conv3",
            op="Conv2d",
            kind="conv",
            module=f"{module}.conv3",
            spec=f"1×1 · {mid}→{cout}",
            shape_in=shape_mid,
            shape_out=shape_out,
            weight=f"W {cout}×{mid}×1×1 · 无偏置",
            params=_conv_params(mid, cout, 1, bias=False),
            formula="Y = W ∗ X",
            source="torchvision.models.resnet.Bottleneck",
            meaning=f"把瓶颈扩回 {cout} 通道，才能和捷径相加。expansion=4，所以输出通道是中间宽度的 4 倍。",
        ))),
        graph.node(**mark(dict(
            id=f"{prefix}_bn3",
            op="BatchNorm2d",
            kind="norm",
            module=f"{module}.bn3",
            spec=f"{cout}",
            shape_in=shape_out,
            shape_out=shape_out,
            weight=f"γ, β 各 {cout}",
            params=_bn_params(cout),
            formula="(X − μ) / √(σ²+ε) · γ + β",
            source="torchvision.models.resnet.Bottleneck",
            meaning="升维之后先归一化再相加，相加之前没有 ReLU，残差可以是负的。",
        ))),
    ]
    graph.chain(ids)
    has_down = cin != cout or stride != 1
    if has_down:
        down = [
            graph.node(**mark(dict(
                id=f"{prefix}_down_conv",
                op="Conv2d",
                kind="conv",
                module=f"{module}.downsample.0",
                spec=f"1×1 · {cin}→{cout} · stride {stride}",
                shape_in=shape_in,
                shape_out=shape_out,
                weight=f"W {cout}×{cin}×1×1 · 无偏置",
                params=_conv_params(cin, cout, 1, bias=False),
                formula=f"捷径投影，stride={stride}",
                source="torchvision.models.resnet.Bottleneck",
                meaning=(
                    "捷径的通道数或步长和主分支不一致，不能直接相加。"
                    "这个 1×1 把恒等分支对齐到相同的形状。它不是额外的语义层，只为让残差公式成立。"
                ),
            ))),
            graph.node(**mark(dict(
                id=f"{prefix}_down_bn",
                op="BatchNorm2d",
                kind="norm",
                module=f"{module}.downsample.1",
                spec=f"{cout}",
                shape_in=shape_out,
                shape_out=shape_out,
                weight=f"γ, β 各 {cout}",
                params=_bn_params(cout),
                formula="(X − μ) / √(σ²+ε) · γ + β",
                source="torchvision.models.resnet.Bottleneck",
                meaning="投影捷径也带 BatchNorm，这样和主分支相加时两边都是归一化后的激活。",
            ))),
        ]
        graph.chain(down)
        graph.edge(src_id, down[0], "skip", graph.get(src_id)["shape_out"])
        identity = down[-1]
        extra = down
    else:
        identity = src_id
        extra = []
    add = graph.node(**mark(dict(
        id=f"{prefix}_add",
        op="Add",
        kind="merge",
        module=f"{module}.forward",
        spec="主分支 + 捷径",
        shape_in=shape_out,
        shape_out=shape_out,
        formula="Y = F(X) + downsample(X)",
        source="torchvision.models.resnet.Bottleneck",
        meaning="Bottleneck 的残差相加。没有投影时捷径就是块的输入；有投影时捷径是 1×1 加 BatchNorm。",
    )))
    relu = graph.node(**mark(dict(
        id=f"{prefix}_relu3",
        op="ReLU",
        kind="act",
        module=f"{module}.relu",
        spec="inplace · 块输出",
        shape_in=shape_out,
        shape_out=shape_out,
        formula="max(0, X)",
        source="torchvision.models.resnet.Bottleneck",
        meaning="相加之后的 ReLU，是这个 Bottleneck 交给下一块的特征图。",
    )))
    graph.edge(ids[-1], add, "data", shape_out)
    if has_down:
        graph.edge(identity, add, "data", shape_out)
    else:
        graph.edge(identity, add, "skip", "")
    graph.edge(add, relu, "data", shape_out)
    return ids + extra + [add, relu], h_out, w_out


def _resnet_until_layer2(
    graph: _Graph,
    in_channels: int,
    height: int,
    width: int,
    frozen_through_layer1: bool,
    freeze_all: bool,
    stem_meaning: str,
) -> tuple[str, int, int]:
    h1, w1 = height // 2, width // 2
    h2, w2 = h1 // 2, w1 // 2
    stem_frozen = freeze_all or frozen_through_layer1
    stem = [
        graph.node(
            id="stem_conv",
            op="Conv2d",
            kind="conv",
            module="backbone.0",
            spec=f"7×7 · stride 2 · {in_channels}→64 · 无偏置",
            shape_in=f"(B, {in_channels}, {height}×{width})",
            shape_out=f"(B, 64, {h1}×{w1})",
            weight=f"W 64×{in_channels}×7×7",
            params=_conv_params(in_channels, 64, 7, bias=False),
            formula="Y = W ∗ X，stride=2，padding=3",
            source="models/line_transformer/line_transformer.py:236",
            meaning=stem_meaning,
            frozen=stem_frozen,
        ),
        graph.node(
            id="stem_bn",
            op="BatchNorm2d",
            kind="norm",
            module="backbone.1",
            spec="64",
            shape_in=f"(B, 64, {h1}×{w1})",
            shape_out=f"(B, 64, {h1}×{w1})",
            weight="γ, β 各 64",
            params=_bn_params(64),
            formula="(X − μ) / √(σ²+ε) · γ + β",
            source="models/line_transformer/line_transformer.py:237",
            meaning="ResNet stem 的 BatchNorm，接在 7×7 卷积后面。",
            frozen=stem_frozen,
        ),
        graph.node(
            id="stem_relu",
            op="ReLU",
            kind="act",
            module="backbone.2",
            spec="inplace",
            shape_in=f"(B, 64, {h1}×{w1})",
            shape_out=f"(B, 64, {h1}×{w1})",
            formula="max(0, X)",
            source="models/line_transformer/line_transformer.py:238",
            meaning="Stem 的 ReLU。MaxPool 在它后面，不在卷积前面。",
            frozen=stem_frozen,
        ),
        graph.node(
            id="stem_pool",
            op="MaxPool2d",
            kind="pool",
            module="backbone.3",
            spec="3×3 · stride 2",
            shape_in=f"(B, 64, {h1}×{w1})",
            shape_out=f"(B, 64, {h2}×{w2})",
            formula="窗口 3×3，stride 2，padding 1",
            source="models/line_transformer/line_transformer.py:239",
            meaning=(
                "Stem 的第二次下采样，无参数。到这里总步长已经是 4。"
                "后面的 layer1 不再下采样，layer2 的第一块才把步长变成 8。"
            ),
            frozen=stem_frozen,
        ),
    ]
    graph.chain(stem)
    graph.place(stem, 0, 0)
    h, w = h2, w2
    previous = stem[-1]
    # layer1: 3 blocks, planes=64, out=256, stride 1
    layer1_frozen = freeze_all or frozen_through_layer1
    specs = [(64, 64, 256, 1)] + [(256, 64, 256, 1)] * 2
    last = previous
    for index, (cin, mid, cout, stride) in enumerate(specs):
        ids, h, w = _bottleneck(
            graph, f"l1b{index}", cin, mid, cout, stride, h, w,
            module=f"backbone.4.{index}",
            frozen=layer1_frozen,
            src_id=last,
        )
        start = graph.get(last)["row"] + 1
        graph.place(ids, 0, start)
        graph.edge(last, ids[0], "data", graph.get(last)["shape_out"])
        last = ids[-1]
    # layer2: 4 blocks, first stride 2, planes=128, out=512
    specs2 = [(256, 128, 512, 2)] + [(512, 128, 512, 1)] * 3
    for index, (cin, mid, cout, stride) in enumerate(specs2):
        ids, h, w = _bottleneck(
            graph, f"l2b{index}", cin, mid, cout, stride, h, w,
            module=f"backbone.5.{index}",
            frozen=freeze_all,
            src_id=last,
        )
        start = graph.get(last)["row"] + 1
        graph.place(ids, 0, start)
        graph.edge(last, ids[0], "data", graph.get(last)["shape_out"])
        last = ids[-1]
    # Annotate that layer3/layer4 are absent.
    graph.get(last)["meaning"] += (
        " 骨干到 layer2 为止，没有 layer3/layer4，所以特征图停在 stride 8、512 通道。"
        "再往下采样，沿线段取 7 个点时就对不齐了。"
    )
    return last, h, w


def _postnorm_layer(
    graph: _Graph,
    prefix: str,
    d_model: int,
    nhead: int,
    row: int,
    col: int,
    src0: str,
    src1: str,
) -> tuple[str, str]:
    """One MultiHeadAttentionBlock. Weights are shared across the two views."""
    shape = f"(B, N, {d_model})"
    head_dim = d_model // nhead
    ff = d_model * 2
    self_attn = graph.node(
        id=f"{prefix}_self",
        op="MultiheadAttention",
        kind="attn",
        module=f"{prefix}.self_attn",
        spec=f"自注意力 · {nhead} 头 · 两视角共用",
        shape_in=shape,
        shape_out=shape,
        weight=f"in_proj {3 * d_model}×{d_model} + out_proj {d_model}×{d_model}",
        params=_mha_params(d_model),
        formula=f"softmax(QKᵀ / √{head_dim}) V，Q=K=V=本视角",
        source="models/line_transformer/line_transformer.py:188",
        meaning=(
            "同一层里，view 0 和 view 1 调用的是同一个 self_attn 模块，权重共享。"
            "这和物体网络里 self0/self1 两套独立权重不同。自注意力只看本图的线段之间谁平行、谁相邻。"
        ),
    )
    norm1 = graph.node(
        id=f"{prefix}_norm1",
        op="LayerNorm",
        kind="norm",
        module=f"{prefix}.norm1",
        spec="Post-Norm",
        shape_in=shape,
        shape_out=shape,
        weight=f"γ, β 各 {d_model}",
        params=_ln_params(d_model),
        formula="LN(X + SelfAttn(X))",
        source="models/line_transformer/line_transformer.py:189",
        meaning=(
            "Post-Norm：先残差相加，再归一化。代码是 `norm1(desc + attn_out)`，归一化包住了残差和。"
            "物体网络的 AttentionBlock 相反，是先 Norm 再注意力。"
        ),
    )
    cross = graph.node(
        id=f"{prefix}_cross",
        op="MultiheadAttention",
        kind="attn",
        module=f"{prefix}.cross_attn",
        spec=f"交叉注意力 · {nhead} 头 · 两方向共用",
        shape_in=shape,
        shape_out=shape,
        weight=f"in_proj {3 * d_model}×{d_model} + out_proj {d_model}×{d_model}",
        params=_mha_params(d_model),
        formula="view0 以 view1 为 K/V，随即对调再算一次",
        source="models/line_transformer/line_transformer.py:200",
        meaning=(
            "同一个 cross_attn 先算 view0←view1，再算 view1←view0。两次调用共享权重。"
            "这是两条线段第一次互相看见的地方。若某一侧没有线段，代码会跳过这一步。"
        ),
    )
    norm2 = graph.node(
        id=f"{prefix}_norm2",
        op="LayerNorm",
        kind="norm",
        module=f"{prefix}.norm2",
        spec="Post-Norm",
        shape_in=shape,
        shape_out=shape,
        weight=f"γ, β 各 {d_model}",
        params=_ln_params(d_model),
        formula="LN(X + CrossAttn(X))",
        source="models/line_transformer/line_transformer.py:202",
        meaning="交叉注意力的残差之后做 LayerNorm。norm2 与 norm1、norm3 是三个独立模块。",
    )
    fc1 = graph.node(
        id=f"{prefix}_fc1",
        op="Linear",
        kind="linear",
        module=f"{prefix}.linear1",
        spec=f"{d_model} → {ff}",
        shape_in=shape,
        shape_out=f"(B, N, {ff})",
        weight=f"W {ff}×{d_model} + bias",
        params=_linear_params(d_model, ff),
        formula="Y = XWᵀ + b",
        source="models/line_transformer/line_transformer.py:207",
        meaning="逐线段的前馈扩展。两个视角再次共用这一层线性，不各备一套。",
    )
    relu = graph.node(
        id=f"{prefix}_relu",
        op="ReLU",
        kind="act",
        module=f"{prefix}.forward",
        spec="F.relu",
        shape_in=f"(B, N, {ff})",
        shape_out=f"(B, N, {ff})",
        formula="max(0, X)",
        source="models/line_transformer/line_transformer.py:207",
        meaning="前馈中间的 ReLU，用的是函数式 F.relu，不是一个带参数的模块。",
    )
    drop = graph.node(
        id=f"{prefix}_drop",
        op="Dropout",
        kind="op",
        module=f"{prefix}.dropout",
        spec="p=0.1",
        shape_in=f"(B, N, {ff})",
        shape_out=f"(B, N, {ff})",
        formula="训练时按 0.1 丢弃",
        source="models/line_transformer/line_transformer.py:171",
        meaning="只作用在前馈隐藏层，注意力权重上的 dropout 用的是 MultiheadAttention 自己的默认值。推理时这一层是恒等。",
    )
    fc2 = graph.node(
        id=f"{prefix}_fc2",
        op="Linear",
        kind="linear",
        module=f"{prefix}.linear2",
        spec=f"{ff} → {d_model}",
        shape_in=f"(B, N, {ff})",
        shape_out=shape,
        weight=f"W {d_model}×{ff} + bias",
        params=_linear_params(ff, d_model),
        formula="Y = XWᵀ + b",
        source="models/line_transformer/line_transformer.py:207",
        meaning="前馈压回 d_model，才能和残差相加。",
    )
    norm3 = graph.node(
        id=f"{prefix}_norm3",
        op="LayerNorm",
        kind="norm",
        module=f"{prefix}.norm3",
        spec="Post-Norm",
        shape_in=shape,
        shape_out=shape,
        weight=f"γ, β 各 {d_model}",
        params=_ln_params(d_model),
        formula="LN(X + FFN(X))",
        source="models/line_transformer/line_transformer.py:208",
        meaning="这一层块的输出。两个视角都写成这个形状，下一层继续用同一结构、但是新的一套权重。",
    )
    ids = [self_attn, norm1, cross, norm2, fc1, relu, drop, fc2, norm3]
    graph.place(ids, col, row)
    graph.edge(src0, self_attn, "data", "view0")
    graph.edge(src1, self_attn, "data", "view1")
    graph.chain(ids)
    return norm3, norm3


def _line_geometry(graph: _Graph, d_model: int, col: int, row: int) -> str:
    ids = [
        graph.node(
            id="geom_in",
            op="Input",
            kind="input",
            module="LineGeometryEncoder",
            spec="像素端点",
            shape_in="(B, N, 4)",
            shape_out="(B, N, 4)",
            formula="[x1, y1, x2, y2]",
            source="models/line_transformer/line_transformer.py:78",
            meaning="线段坐标是原图像素，还不是描述子。编码器内部会除以图像宽高和对角线。",
        ),
        graph.node(
            id="geom_5d",
            op="Geom5D",
            kind="op",
            module="LineGeometryEncoder.forward",
            spec="中心 / 长度 / 二倍角",
            shape_in="(B, N, 4)",
            shape_out="(B, N, 5)",
            formula="[cx/W, cy/H, ‖Δ‖/diag, sin 2θ, cos 2θ]",
            source="models/line_transformer/line_transformer.py:103",
            meaning=(
                "和切片直线编码器同一个二倍角，但归一化基准不同：中心除以宽高，长度除以图像对角线。"
                "端点对调不改变这 5 个数，所以匹配不会因为 LSD 端点顺序而翻号。"
            ),
        ),
        graph.node(
            id="geom_fc1",
            op="Linear",
            kind="linear",
            module="geometry_encoder.mlp.0",
            spec="5 → 64",
            shape_in="(B, N, 5)",
            shape_out="(B, N, 64)",
            weight="W 64×5 + bias",
            params=_linear_params(5, 64),
            formula="Y = XWᵀ + b",
            source="models/line_transformer/line_transformer.py:72",
            meaning="几何 MLP 的第一层，每条线段独立。",
        ),
        graph.node(
            id="geom_relu",
            op="ReLU",
            kind="act",
            module="geometry_encoder.mlp.1",
            spec="inplace",
            shape_in="(B, N, 64)",
            shape_out="(B, N, 64)",
            formula="max(0, X)",
            source="models/line_transformer/line_transformer.py:73",
            meaning="几何分支中间的 ReLU。后面一层线性直接接到 d_model，再加 LayerNorm。",
        ),
        graph.node(
            id="geom_fc2",
            op="Linear",
            kind="linear",
            module="geometry_encoder.mlp.2",
            spec=f"64 → {d_model}",
            shape_in="(B, N, 64)",
            shape_out=f"(B, N, {d_model})",
            weight=f"W {d_model}×64 + bias",
            params=_linear_params(64, d_model),
            formula="Y = XWᵀ + b",
            source="models/line_transformer/line_transformer.py:74",
            meaning=f"几何向量对齐到 {d_model}，才能和视觉采样向量逐元素相加，而不是拼接。",
        ),
        graph.node(
            id="geom_ln",
            op="LayerNorm",
            kind="norm",
            module="geometry_encoder.mlp.3",
            spec=f"{d_model}",
            shape_in=f"(B, N, {d_model})",
            shape_out=f"(B, N, {d_model})",
            weight=f"γ, β 各 {d_model}",
            params=_ln_params(d_model),
            formula="对几何 token 归一化",
            source="models/line_transformer/line_transformer.py:75",
            meaning="相加之前的归一化。视觉支路的投影末尾也有 LayerNorm，两边幅度才加得起来。",
        ),
    ]
    graph.chain(ids)
    graph.place(ids, col, row)
    return ids[-1]


def _sampler(graph: _Graph, in_channels: int, d_model: int, col: int, row: int, feat_id: str) -> str:
    ids = [
        graph.node(
            id="sample_pts",
            op="Lerp",
            kind="op",
            module="LineVisualSampler.forward",
            spec="K=7 等分",
            shape_in="(B, N, 4)",
            shape_out="(B, N, 7, 2)",
            formula="(1−t)·p1 + t·p2，t=0,1/6,…,1",
            source="models/line_transformer/line_transformer.py:142",
            meaning="沿每条线段均匀取 7 个点，包含两个端点。采样是几何插值，还没有读特征图。",
        ),
        graph.node(
            id="grid_sample",
            op="GridSample",
            kind="op",
            module="F.grid_sample",
            spec="bilinear · border",
            shape_in=f"(B, {in_channels}, H/8, W/8)",
            shape_out=f"(B, N, 7, {in_channels})",
            formula="像素坐标线性映射到 [−1, 1]，align_corners=True",
            source="models/line_transformer/line_transformer.py:151",
            meaning=(
                "在骨干特征图上双线性取值。坐标先除以原图宽高再映射到 grid_sample 的 [−1, 1]，"
                "所以特征图是 stride 8 时，点会落在对应的粗网格上。越界使用边界值，不补零。"
            ),
        ),
        graph.node(
            id="sample_mean",
            op="Mean",
            kind="pool",
            module="LineVisualSampler.forward",
            spec="沿 K=7",
            shape_in=f"(B, N, 7, {in_channels})",
            shape_out=f"(B, N, {in_channels})",
            formula="mean over 7 samples",
            source="models/line_transformer/line_transformer.py:156",
            meaning="7 个采样点做平均，使线段描述与点的顺序无关。端点对调不会改变这 7 个点的集合平均。",
        ),
        graph.node(
            id="sample_fc1",
            op="Linear",
            kind="linear",
            module="sampler.proj.0",
            spec=f"{in_channels} → {d_model}",
            shape_in=f"(B, N, {in_channels})",
            shape_out=f"(B, N, {d_model})",
            weight=f"W {d_model}×{in_channels} + bias",
            params=_linear_params(in_channels, d_model),
            formula="Y = XWᵀ + b",
            source="models/line_transformer/line_transformer.py:116",
            meaning=f"把采样到的 {in_channels} 维外观压到 d_model。这一层是可训练的，即使骨干被冻结，线段外观仍能适配匹配任务。",
        ),
        graph.node(
            id="sample_relu",
            op="ReLU",
            kind="act",
            module="sampler.proj.1",
            spec="inplace",
            shape_in=f"(B, N, {d_model})",
            shape_out=f"(B, N, {d_model})",
            formula="max(0, X)",
            source="models/line_transformer/line_transformer.py:117",
            meaning="视觉投影中间的 ReLU。",
        ),
        graph.node(
            id="sample_fc2",
            op="Linear",
            kind="linear",
            module="sampler.proj.2",
            spec=f"{d_model} → {d_model}",
            shape_in=f"(B, N, {d_model})",
            shape_out=f"(B, N, {d_model})",
            weight=f"W {d_model}×{d_model} + bias",
            params=_linear_params(d_model, d_model),
            formula="Y = XWᵀ + b",
            source="models/line_transformer/line_transformer.py:118",
            meaning="第二层线性保持宽度，准备和几何向量相加。",
        ),
        graph.node(
            id="sample_ln",
            op="LayerNorm",
            kind="norm",
            module="sampler.proj.3",
            spec=f"{d_model}",
            shape_in=f"(B, N, {d_model})",
            shape_out=f"(B, N, {d_model})",
            weight=f"γ, β 各 {d_model}",
            params=_ln_params(d_model),
            formula="对视觉 token 归一化",
            source="models/line_transformer/line_transformer.py:119",
            meaning="视觉支路的末尾归一化。下一步是和几何 token 逐元素相加，不是拼接。",
        ),
    ]
    graph.place(ids, col, row)
    graph.edge(feat_id, ids[1], "data", graph.get(feat_id)["shape_out"])
    graph.edge("geom_in", ids[0], "data", "(B, N, 4)")
    graph.chain(ids)
    return ids[-1]


def _add_tokens(graph: _Graph, d_model: int, col: int, row: int, vis_id: str, geom_id: str) -> str:
    node = graph.node(
        id="tok_add",
        op="Add",
        kind="merge",
        module="forward",
        spec="视觉 + 几何",
        shape_in=f"(B, N, {d_model})",
        shape_out=f"(B, N, {d_model})",
        formula="token = vis + geom",
        source="models/line_transformer/line_transformer.py:275",
        meaning=(
            "视觉采样和几何编码逐元素相加，而不是像 XFeatLineMatcher 那样拼接后再投影。"
            "相加要求两边已经对齐到同一宽度，也意味着网络不能把某一维专门留给直线长度。"
        ),
    )
    graph.place([node], col, row)
    graph.edge(vis_id, node, "data", f"(B, N, {d_model})")
    graph.edge(geom_id, node, "data", f"(B, N, {d_model})")
    return node


def _line_match_head(
    graph: _Graph,
    d_model: int,
    col: int,
    row: int,
    src: str,
    canonical_255: bool,
    exponentiate: bool,
) -> None:
    shape = f"(B, N, {d_model})"
    proj = graph.node(
        id="desc_proj",
        op="Linear",
        kind="linear",
        module="proj0 / proj1",
        spec=f"{d_model} → {d_model} · 两视角各一套",
        shape_in=shape,
        shape_out=shape,
        weight=f"两套 W {d_model}×{d_model} + bias",
        params=_linear_params(d_model, d_model) * 2,
        formula="desc = proj(token)",
        source="models/line_transformer/line_transformer.py:258",
        meaning="proj0 和 proj1 是两个独立线性层，分别作用在两个视角。图中画成一个节点，参数按两套计算。",
    )
    norm = graph.node(
        id="l2norm",
        op="L2Norm",
        kind="op",
        module="F.normalize",
        spec="沿通道",
        shape_in=shape,
        shape_out=shape,
        formula="desc / ‖desc‖₂",
        source="models/mask_line_matcher/mask_line_matcher.py:151",
        meaning="描述子归一化之后，后面的矩阵乘就是余弦相似度，数值被限制在 [−1, 1]，不再受向量长度影响。",
    )
    score = graph.node(
        id="cosine",
        op="BMM",
        kind="op",
        module="forward",
        spec="余弦",
        shape_in=shape,
        shape_out="(B, N0, N1)",
        formula="S = desc0 · desc1ᵀ",
        source="models/line_transformer/line_transformer.py:309",
        meaning="每条线段对每条线段一个分数。这里还没有互斥约束。",
    )
    dust = graph.node(
        id="dustbin",
        op="Pad",
        kind="op",
        module="log_optimal_transport",
        spec="常数 0 · 不可学习",
        shape_in="(B, N0, N1)",
        shape_out="(B, N0+1, N1+1)",
        formula="新增的行、列、角点填 0",
        source="models/line_transformer/line_transformer.py:43",
        meaning=(
            "线段网络的垃圾桶是常数 0，不是 XFeat 里那个可学习的 dustbin_score。"
            "拒绝匹配的强度因此不能单独训练，只能靠 Sinkhorn 的边际分布间接体现。"
        ),
    )
    sink = graph.node(
        id="sinkhorn",
        op="LogSinkhorn",
        kind="sinkhorn",
        module="log_sinkhorn_iterations",
        spec="20 次",
        shape_in="(B, N0+1, N1+1)",
        shape_out="(B, N0+1, N1+1)",
        formula="对数空间行列归一化",
        source="models/line_transformer/line_transformer.py:23",
        meaning=(
            "与物体网络相同的对数 Sinkhorn，但边际分布的构造不同：普通行先减去 log(N)，"
            "垃圾桶位置再用两侧线段数量的比例。迭代 20 次，输出仍是对数分配。"
        ),
    )
    ids = [proj, norm, score, dust, sink]
    if exponentiate:
        ids.append(graph.node(
            id="exp",
            op="Exp",
            kind="op",
            module="forward",
            spec="exp",
            shape_in="(B, N0+1, N1+1)",
            shape_out="(B, N0+1, N1+1)",
            formula="P = exp(log P)",
            source="models/mask_line_matcher/mask_line_matcher.py:159",
            meaning="指数化得到含垃圾桶的分配矩阵。左上角 N0×N1 的子块才是线段对线段的概率。",
        ))
    if canonical_255:
        ids.append(graph.node(
            id="matrix_255",
            op="Pad",
            kind="op",
            module="forward",
            spec="写入 255×255",
            shape_in="(B, N0, N1)",
            shape_out="(B, 255, 255)",
            formula="matrix[:, :N0, :N1] = P_sub，其余为 0",
            source="models/mask_line_matcher/mask_line_matcher.py:163",
            meaning=(
                "这不是一个可学习层。为了固定接口，概率子块被抄进一张 255×255 的零矩阵。"
                "超过 255 条的线段会被丢掉。零填充不代表那些位置匹配概率是 0 的预测，只是没有线段。"
            ),
        ))
    graph.place(ids, col, row)
    graph.edge(src, proj, "data", shape)
    graph.chain(ids)


def build_line_transformer_graph(d_model: int = 128, num_layers: int = 4, nhead: int = 4) -> dict[str, Any]:
    height, width = 480, 640
    graph = _Graph()
    feat, _h, _w = _resnet_until_layer2(
        graph,
        in_channels=3,
        height=height,
        width=width,
        frozen_through_layer1=True,
        freeze_all=True,
        stem_meaning=(
            "LineMatcherNet 的入口是 RGB 整图，不是切片。7×7 卷积来自 ImageNet 预训练的 ResNet-50，"
            "并且 freeze_backbone=True 时，从这一层到 layer2 的全部权重都不更新。"
        ),
    )
    geom = _line_geometry(graph, d_model, col=1, row=0)
    vis = _sampler(graph, 512, d_model, col=2, row=0, feat_id=feat)
    token = _add_tokens(graph, d_model, col=3, row=max(graph.get(geom)["row"], graph.get(vis)["row"]) + 1, vis_id=vis, geom_id=geom)
    src0 = src1 = token
    row = graph.get(token)["row"] + 2
    last = token
    for index in range(num_layers):
        out0, out1 = _postnorm_layer(
            graph, f"layers.{index}", d_model, nhead, row, 3, src0, src1,
        )
        src0 = src1 = out0
        last = out0
        row = graph.get(out0)["row"] + 2
    _line_match_head(graph, d_model, 3, row, last, canonical_255=False, exponentiate=False)
    return _pack(
        graph,
        title="LineMatcherNet · ResNet-50 至 layer2",
        assumption=(
            f"形状按 480×640 的输入计算，特征图到 layer2 为 512×60×80（总步长 8）。"
            f"d_model={d_model}，Transformer {num_layers} 层。骨干默认全部冻结，采样投影和注意力仍训练。"
        ),
        jumps=_jumps(graph, [
            ("Stem", "stem_conv"),
            ("Layer1", "l1b0_conv1"),
            ("Layer2", "l2b0_conv2"),
            ("沿线采样", "grid_sample"),
            ("二倍角", "geom_5d"),
            ("相加", "tok_add"),
            ("自注意力", "layers.0_self"),
            ("Sinkhorn", "sinkhorn"),
        ]),
    )


def build_mask_line_graph(d_model: int = 128, num_layers: int = 4, nhead: int = 4) -> dict[str, Any]:
    height, width = 480, 640
    graph = _Graph()
    feat, _h, _w = _resnet_until_layer2(
        graph,
        in_channels=6,
        height=height,
        width=width,
        frozen_through_layer1=True,
        freeze_all=False,
        stem_meaning=(
            "第一层卷积被改成 6 通道：前 3 个通道拷贝 ResNet-50 的 RGB 预训练权重，"
            "后 3 个通道（EDT、sin2θ、cos2θ）用 RGB 卷积核的通道均值初始化。"
            "freeze_backbone_stages≥1 时，stem 和 layer1 冻结，layer2 仍训练。"
        ),
    )
    graph.get("stem_conv")["source"] = "models/mask_line_matcher/mask_line_matcher.py:58"
    graph.get("stem_conv")["weight"] = "W 64×6×7×7"
    graph.get("stem_conv")["params"] = _conv_params(6, 64, 7, bias=False)
    geom = _line_geometry(graph, d_model, col=1, row=0)
    vis = _sampler(graph, 512, d_model, col=2, row=0, feat_id=feat)
    token = _add_tokens(graph, d_model, col=3, row=max(graph.get(geom)["row"], graph.get(vis)["row"]) + 1, vis_id=vis, geom_id=geom)
    graph.get("tok_add")["source"] = "models/mask_line_matcher/mask_line_matcher.py:144"
    src = token
    row = graph.get(token)["row"] + 2
    last = token
    for index in range(num_layers):
        out0, _out1 = _postnorm_layer(graph, f"layers.{index}", d_model, nhead, row, 3, src, src)
        src = out0
        last = out0
        row = graph.get(out0)["row"] + 2
    _line_match_head(graph, d_model, 3, row, last, canonical_255=True, exponentiate=True)
    # 6-channel input note sits on a dedicated node above the stem by shifting? Stem already starts at 0.
    # The stem meaning covers it. Add an input node in column 1 is taken by geometry.
    # Insert a field node by rewriting stem input meaning only — already done.
    return _pack(
        graph,
        title="MaskLineFeatMatchNet · 6 通道 ResNet-50 至 layer2",
        assumption=(
            "若传入的是 RGB，forward 会先调用 create_6ch_input_tensor：EDT 截断在 60 像素，朝向是 sin2θ/cos2θ。"
            f"图中空间尺寸按 480×640 计算。d_model={d_model}。stem 与 layer1 冻结，layer2 可训练。"
        ),
        jumps=_jumps(graph, [
            ("6 通道 Stem", "stem_conv"),
            ("Layer2", "l2b0_conv2"),
            ("EDT/朝向", "geom_5d"),
            ("沿线采样", "grid_sample"),
            ("相加", "tok_add"),
            ("交叉注意力", "layers.0_cross"),
            ("255 矩阵", "matrix_255"),
        ]),
    )


def _jumps(graph: _Graph, pairs: list[tuple[str, str]]) -> list[dict[str, str]]:
    have = {node["id"] for node in graph.nodes}
    return [{"label": label, "node": node_id} for label, node_id in pairs if node_id in have]


def _pack(graph: _Graph, title: str, assumption: str, jumps: list[dict[str, str]]) -> dict[str, Any]:
    for node in graph.nodes:
        if not node.get("meaning"):
            raise ValueError(f"node {node['id']} missing meaning")
    return {
        "title": title,
        "assumption": assumption,
        "nodes": graph.nodes,
        "edges": graph.edges,
        "jumps": jumps,
        "param_count": sum(int(node.get("params") or 0) for node in graph.nodes),
    }


def build_semantic_line_graph(use_pyramid: bool = True, d_model: int = 128, num_layers: int = 2, nhead: int = 4) -> dict[str, Any]:
    """Compact interactive layer graph for SemanticLineMatcher dual outputs."""
    feat_dim = 64
    line_dim = 64
    graph = _Graph()
    vis = _visual_tower(graph, use_pyramid=use_pyramid, feat_dim=feat_dim)
    geos = _geo_encoder(graph, feat_dim)
    vis_row = graph.get(vis)["row"]
    graph.place(geos, 2, vis_row - len(geos) + 1)

    along0 = graph.node(
        id="along_sample",
        op="GridSample",
        kind="op",
        module="line_encoder.sample_along_lines",
        spec="T=8 along-line",
        shape_in="fmap + lines(B,M,K,5)",
        shape_out="(N, K, T, C)",
        formula="p(t)=(1-t)p1+t p2",
        source="models/semantic_line_matcher/line_encoder.py",
        meaning="在区域特征图上沿每条直线均匀取 T 个点（默认 8），得到纹理序列。",
    )
    along1 = graph.node(
        id="along_conv1d",
        op="Conv1d",
        kind="conv",
        module="line_encoder.conv1d",
        spec=f"C → {line_dim}",
        shape_in="(N·K, C, T)",
        shape_out=f"(B, M, K, {line_dim})",
        formula="mean_t Conv1d(x)",
        source="models/semantic_line_matcher/line_encoder.py",
        meaning="一维卷积 + 均值池化，把沿线纹理压成直线向量 f_ℓ。",
    )
    graph.place([along0, along1], 3, vis_row - 1)
    graph.edge(vis, along0, "data", graph.get(vis)["shape_out"])
    graph.edge(along0, along1, "data", graph.get(along0)["shape_out"])

    vis_dim = 3 * feat_dim if use_pyramid else feat_dim
    _fuse_and_match(
        graph,
        [(vis, f"视觉 {vis_dim}D"), (geos[-1], f"布局 {feat_dim}D")],
        in_dim=vis_dim + feat_dim,
        d_model=d_model,
        num_layers=num_layers,
        nhead=nhead,
        fuse_module="region_fusion",
        fuse_meaning="区域向量 f_r 与 bbox 几何融合，进入区域 Transformer + Sinkhorn，输出 P_region。",
    )

    cat = graph.node(
        id="line_cat",
        op="ConcatProj",
        kind="merge",
        module="line_desc_proj",
        spec=f"[f_ℓ ‖ f_r] → {d_model}",
        shape_in=f"({line_dim}+{vis_dim})",
        shape_out=f"(N_flat, {d_model})",
        formula="Proj([f_line || f_r])",
        source="models/semantic_line_matcher/model.py",
        meaning="同一区域的 f_r 广播到各条线，与沿线纹理拼接后投影为直线匹配描述子。",
    )
    line_sink = graph.node(
        id="line_sinkhorn",
        op="LogSinkhorn",
        kind="match",
        module="line_matcher",
        spec="dustbin OT",
        shape_in=f"(N0,{d_model})×(N1,{d_model})",
        shape_out="(N0+1, N1+1)",
        formula="Sinkhorn(S/τ)",
        source="models/semantic_line_matcher/model.py",
        meaning="对展平有效直线做最优传输；损失另含线→语义 JSD 与区域邻接保持。",
    )
    sink_row = max(n["row"] for n in graph.nodes) + 2
    graph.place([cat, line_sink], 4, sink_row)
    graph.edge(along1, cat, "data", graph.get(along1)["shape_out"])
    graph.edge(vis, cat, "data", graph.get(vis)["shape_out"])
    graph.edge(cat, line_sink, "data", graph.get(cat)["shape_out"])

    mode = "FPN" if use_pyramid else "单尺度"
    return _pack(
        graph,
        title=f"SemanticLineMatcher · {mode} · 双输出",
        assumption=(
            f"区域与直线并行：d_model={d_model}，注意力 {num_layers}×{nhead}。"
            "直线描述子=沿线纹理‖区域特征；邻接几何只在损失中，不进打分。"
        ),
        jumps=_jumps(graph, [
            ("输入切片", "vis_in"),
            ("视觉骨干", "s1_conv"),
            ("沿线采样", "along_sample"),
            ("区域融合", "fuse_cat"),
            ("区域 Sinkhorn", "sinkhorn"),
            ("直线拼接", "line_cat"),
            ("直线 Sinkhorn", "line_sinkhorn"),
        ]),
    )


def build_layer_graph(model_name: str, use_pyramid: bool = True, d_model: int = 128) -> dict[str, Any]:
    """Return the concrete layer graph for a model alias."""
    name = model_name.lower()
    if name in {"semantic_line_matcher", "semantic_line"}:
        return build_semantic_line_graph(use_pyramid=use_pyramid, d_model=d_model)
    if name in {"xfeat_line_matcher", "xfeat_line", "line_object", "line_matcher"}:
        return build_xfeat_line_graph(use_pyramid=use_pyramid, d_model=d_model)
    if name in {"xfeat_object", "xfeat", "object"}:
        return build_xfeat_object_graph(use_pyramid=use_pyramid, d_model=d_model)
    if name in {"mask_line", "mask"}:
        return build_mask_line_graph(d_model=d_model)
    if name in {"line_transformer", "linefeat"}:
        return build_line_transformer_graph(d_model=d_model)
    raise ValueError(f"Unknown model for layer graph: {model_name}")
