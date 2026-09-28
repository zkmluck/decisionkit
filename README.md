# decisionkit

**状态进,分布出,零 token 解码,没有 GPU。** 一个用 numpy 写成的"类型化决策"参考实现:给它一段无结构状态和几道结构化问题,它直接返回每个候选的概率分布——不生成答案文本,不下载模型权重,不需要显卡。

## 为什么做这个

[agent-jev](https://github.com/malevrigns/agent-jev) 那类 System One 决策模型证明了路线:很多 Agent 步骤要的不是写作,而是一个判断,那就别让模型把答案一个 token 一个 token 地写出来,直接对候选打分。但那条路要 Qwen3-0.6B 加 CUDA 环境。

这里补的是它旁边的空位:**没有显卡、不想下载几百 MB 权重、只想先把"状态进分布出"这条链路跑通、训起来、量出来的时候,用什么?** 答案是这份东西——它不打算跟决策模型比效果,它是一份 CPU 基线 + 契约参考实现,顺带把指标、校准和服务端一起给全。

## 三个原语

每个原语都返回**完整分布**,不只给赢家。

| 原语 | 你问 | 你送 | 你拿到 |
| --- | --- | --- | --- |
| `boolean` | 一个命题 | 可选的 `criteria` 描述真假 | `value`、`probability`、两个候选的质量 |
| `choice` | 这些选项里选哪个 | 2–255 个候选(映射或列表) | `value`、`description`、`top_probability`、`margin`、完整分布 |
| `score` | 落在这把有序刻度的哪一档 | 2–10 档、从低到高 | `level`(argmax)、`score` = Σ i·Pᵢ、`legend` |

## 契约里几条硬规矩

- **候选顺序不可能影响结果**:每题所有候选共用一个权重向量,置换候选只是置换输出。有测试守着。
- **不确定就说不确定**:top 概率或 margin 没过阈值,`status` 是 `review` 而不是硬猜——这条借自 [gui-verifier](https://github.com/xuxufei12/gui-verifier) 的处理。
- **传输用的 id 不进模型输入**:问题 id、案例 id 只用于对齐答案,不参与打分。
- **结构化的 state 会被规范化**:对象/数组统一按 key 排序序列化,同一份状态不因书写顺序而不同分。
- **批量与上限显式**:一个请求最多 32 个状态、128 道题、1024 条候选路径,超了报错,不静默截断。

## 快速开始

```bash
cd decisionkit
python -m venv .venv && . .venv/bin/activate    # Windows: .venv\Scripts\Activate.ps1
pip install -e .

decisionkit synth --out data/synthetic --cases 400
decisionkit train --train data/synthetic/train.jsonl \
                  --validation data/synthetic/validation.jsonl \
                  --output models/linear.npz
decisionkit evaluate --data data/synthetic/test.jsonl --model models/linear.npz
decisionkit verify --model models/linear.npz
decisionkit serve --model models/linear.npz --port 8765
```

不想装包也行:`PYTHONPATH=src python -m decisionkit ...`。

## 实测数字

上面那几条命令在本机(Windows、Python 3.10、纯 CPU、无 GPU)的真实输出:

```text
synth      400 个案例 → train 280 / validation 60 / test 60
train      16 epoch,每 epoch 756 次更新,训练损失 0.781 → 0.616
eval       60 个案例 / 159 道决策
           温度(boolean / choice / score)= 1.0362 / 0.9651 / 1.0362
```

| 指标 | 未校准 | 校准后 |
| --- | ---: | ---: |
| 与教师 argmax 的一致率 | 1.000 | 1.000 |
| Macro-F1 | 1.000 | 1.000 |
| Brier(按候选求和)↓ | 0.0557 | 0.0576 |
| NLL ↓ | 0.2051 | 0.2088 |
| ECE(10 等宽)↓ | 0.1838 | 0.1868 |
| 判定覆盖率 | 1.000 | 1.000 |

每案例延迟 p50 1.82 ms、p95 2.74 ms(校准前);**生成 token 数 0**。

## 这些数字怎么读

**合成任务上的 100% 什么都不能证明。** 那批数据是按固定提示词生成的,模型学到的就是那些提示词;它证明的是流水线通了——特征、损失、训练、校准、指标、服务能串起来——不是任何真实任务上的能力。真实数据请只看 NLL、Brier 和校准曲线。

**ECE 在这里有个天然地板。** 标签是按 0.85/0.15 平滑过的软分布,而 ECE 现在是拿"预测最大项"和"是否命中 argmax"比:一个完全学对的模型,置信度也只会收敛到 0.85 附近,对硬标签算 ECE 自然有约 0.15 的地板。所以 ECE 要跟 NLL 一起看,单独看会误判。

**校准在这份数据上几乎没动。** 温度拟合出来接近 1,因为模型本来就不算过度自信;`calibration.py` 的价值在模型过度自信时才显现(比如拿硬标签训练)。这里把它跑通、把接口留好,比假装它带来了提升更诚实。

## 输入输出

请求见 [examples/request.json](examples/request.json)。最小形态:

```json
{
  "state": {"notes": "the last run reported FAILED with a traceback"},
  "questions": [
    {"id": "next", "type": "choice", "question": "Which action should run next?",
     "options": {"read": "read the failing test", "patch": "patch the implementation"}},
    {"id": "risk", "type": "score", "question": "How risky is the next command?",
     "levels": ["0 - read only", "3 - destructive"]}
  ],
  "min_probability": 0.6
}
```

HTTP 返回的形状(与 agent-jev 一类接口保持一致):

```json
{"api_version": "decisionkit.v1", "model": "decisionkit-linear",
 "results": [{"id": "0", "answers": [
   {"id": "next", "type": "choice", "distribution": {"read": 0.886, "patch": 0.018},
    "value": "read", "description": "read the failing test",
    "top_probability": 0.886, "margin": 0.835, "status": "decided", "generated_tokens": 0}]}],
 "usage": {"questions": 2, "candidate_paths": 3, "input_tokens": 214,
           "generated_tokens": 0, "wall_ms": 1.8}}
```

## 目录

| 路径 | 是什么 |
| --- | --- |
| `src/decisionkit/contract.py` | 请求契约与校验;id 不进输入 |
| `src/decisionkit/features.py` | 哈希特征(状态/问题/候选,加候选×状态关键词的交互项) |
| `src/decisionkit/train.py` | 软交叉熵 + Brier 的精确梯度训练,只需要 numpy |
| `src/decisionkit/calibration.py` | 按原语拟合温度;ECE |
| `src/decisionkit/model.py` | 打分、softmax、组装三种答案、`review` 判定 |
| `src/decisionkit/metrics.py` | 一致率、Macro-F1、Brier、NLL、ECE、覆盖率、延迟分位 |
| `src/decisionkit/service.py`、`client.py` | 只绑 127.0.0.1 的 HTTP 服务与标准库客户端 |
| `src/decisionkit/synth.py` | 确定性合成数据与按案例切分 |
| `src/decisionkit/cli.py` | `synth` / `train` / `evaluate` / `serve` / `verify` |
| `tests/` | 21 个测试:契约拒绝、置换等变、度量手算、训练效果、服务往返 |

## 限制

- 模型是**线性打分器 + 哈希特征**,不是 transformer。同义词、长距离推理、跨域迁移都不要指望它。
- 合成数据自造、无人工金标、无多 seed、无并发压测;这里所有数字都是本机 CPU 上的单次结果。
- 服务没有鉴权,默认只绑 `127.0.0.1`,不要直接暴露公网。
- 仓库暂时**没有选许可证**,这个留给你定。

## 测试

```bash
pip install -e '.[test]'
python -m pytest -q      # 21 passed
```

---

### English summary

`decisionkit` is a numpy-only reference implementation of typed decision
inference: give it a state and typed questions (`boolean` / `choice` / `score`)
and it returns a full probability distribution per question, with zero decoded
tokens and no GPU or downloaded weights. It ships a trainable hashed-feature
scorer, per-primitive temperature calibration, metrics (accuracy, macro-F1,
Brier, NLL, ECE, coverage, latency), a loopback HTTP service and a stdlib
client. It is positioned as the no-GPU baseline next to transformer decision
heads such as agent-jev, not as a competitor to them: the 100% figure on the
synthetic task is a pipeline check, and the ECE floor on smoothed labels is
explained above.
