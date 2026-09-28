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

# 仓库里已经带了公开评测集和一份训练好的基线模型,这两条离线就能跑:
decisionkit evaluate --data data/typed-decisions/test.jsonl --model models/typed-decisions.npz
decisionkit verify --model models/typed-decisions.npz
decisionkit serve --model models/typed-decisions.npz --port 8765

# 想从上游重建数据、自己训一遍:
python tools/import_typed_decisions.py --out data/typed-decisions
decisionkit train --train data/typed-decisions/train.jsonl \
                  --validation data/typed-decisions/calibration.jsonl \
                  --output models/typed-decisions.npz

# 不需要任何数据的自检流程:
decisionkit synth --out data/synthetic --cases 400
decisionkit train --train data/synthetic/train.jsonl \
                  --validation data/synthetic/validation.jsonl \
                  --output models/linear.npz
decisionkit evaluate --data data/synthetic/test.jsonl --model models/linear.npz
```

不想装包也行:`PYTHONPATH=src python -m decisionkit ...`。

## 真实数据上的结果

数据用 agent-jev 评测的那份公开集:`LocalLLaMA/typed-decisions`(Apache-2.0)的官方 train/test 划分,标签是教师给的软分布。**转换后的文件和一个训练好的基线模型都随仓库发布**,所以下面第一条评测命令离线就能跑;要从上游重建,跑第二段。

```bash
# 已经带在仓库里,直接评测:
decisionkit evaluate --data data/typed-decisions/test.jsonl --model models/typed-decisions.npz

# 想自己重来一遍:
python tools/import_typed_decisions.py --out data/typed-decisions
decisionkit train --train data/typed-decisions/train.jsonl \
                  --validation data/typed-decisions/calibration.jsonl \
                  --output models/typed-decisions.npz
```

数据出处、许可、上游修订版和逐文件 SHA-256 都在 [`data/typed-decisions/README.md`](data/typed-decisions/README.md) 里;重训一次得到的结果与下表中的数字逐位相同。

导入得到 1200 个官方训练案例、400 个官方测试案例,每个案例 5 道题,共 8000 道,零题被跳过。官方划分没有开发集,所以我把训练案例里每第 10 个留作温度校准(120 个),训练用剩下的 1080 个;测试集原样使用,和训练没有任何案例重叠。

```text
train   1080 案例 / 每 epoch 5400 次更新 / 16 epoch,损失 1.151 → 0.965
        温度(boolean / choice / score)= 2.110 / 1.965 / 2.265
eval    400 案例 / 2000 道决策(官方测试集)
```

| 指标 | 未校准 | 校准后 |
| --- | ---: | ---: |
| 与教师 argmax 的一致率 | 0.615 | 0.615 |
| Macro-F1 | 0.509 | 0.509 |
| Brier(按候选求和)↓ | 0.5225 | **0.5085** |
| NLL ↓ | 0.9289 | **0.8929** |
| ECE(10 等宽)↓ | 0.1060 | **0.0626** |
| 判定覆盖率(过 0.6 阈值) | 0.699 | 0.370 |
| 下了判断的那些里答对的比例 | 0.689 | **0.780** |
| 每案例延迟 p50 / p95 | 12.33 / 22.24 ms | 12.54 / 22.26 ms |

**生成 token 数 0。** 校准不改变 argmax,所以一致率不动;它改的是概率本身——ECE 从 0.106 降到 0.063,代价是敢下判断的比例从 70% 掉到 37%,而那 37% 里有 78% 是对的。"要么说准、要么承认不确定"这个取舍才是校准该带来的东西。

同一份官方测试集上,agent-jev 公布的 AgentJev-0.6B 是 **79.25%(1585/2000)**、CE 0.8494、Brier 0.0448。这份 CPU 线性基线是 **61.5%**,差 17.75 个百分点——这就是"不下载权重、不用显卡、训练一分半钟"的代价,而不是"打平"。

## 合成数据(只是流水线自检)

```bash
decisionkit synth --out data/synthetic --cases 400
decisionkit train --train data/synthetic/train.jsonl \
                  --validation data/synthetic/validation.jsonl --output models/linear.npz
decisionkit evaluate --data data/synthetic/test.jsonl --model models/linear.npz
```

400 个自造案例(280/60/60),在这上面一致率 1.000、每案例 p50 1.82 ms、生成 token 0 个。**这个 100% 什么都不能证明**:数据是按固定提示词造的,模型学到的就是那些提示词;它的用处是证明流水线通了。校准在这里几乎不动(温度 ≈ 1.04),因为模型本来就不过度自信——换成真实数据立刻不一样,见上表。

## 这些数字怎么读

**标签是教师分布,不是人工金标。** 一致率指"与公开教师分布 argmax 的一致率",和 agent-jev 表里同名字段同口径;它不等于任务成功率。

**Brier / CE / ECE 各家定义可能不同,别跨表硬比。** 这里 Brier 是"每个决策按候选求 (p−q)² 求和,再对决策取平均";CE 是软标签交叉熵;ECE 是 10 个等宽区间,拿最大概率和是否命中 argmax 比。对着别的表格引数字前先核对定义。

**和 agent-jev 不是等预算比较。** 它是 0.6B 的决策 transformer、GPU 训练,而且在 1200 个训练案例里留出 120 开发 + 120 校准;这里是线性打分器 + 哈希特征、CPU、1080 训练 + 120 校准。那 17.75 个百分点是模型类别的差距。

**覆盖率和一致率要一起看。** 0.6 这个阈值是随手定的,不是验证过的放行线;改 `min_probability` 就是在"多下判断"和"下得准"之间挪动,真正上线得用你自己的数据把这条线定下来。

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
| `tools/import_typed_decisions.py` | 把公开的 typed-decisions 数据集转成本项目的 JSONL(带重试与分页) |
| `data/typed-decisions/` | 随仓库发布的转换后评测数据、出处与许可(Apache-2.0) |
| `models/typed-decisions.npz` | 随仓库发布的基线模型:在 1080 个案例上训练 16 epoch |
| `tests/` | 26 个测试:契约拒绝、置换等变、度量手算、训练效果、服务往返、导入器映射 |

## 限制

- 模型是**线性打分器 + 哈希特征**,不是 transformer。同义词、长距离推理、跨域迁移都不要指望它。
- 合成数据自造、无人工金标、无多 seed、无并发压测;这里所有数字都是本机 CPU 上的单次结果。
- 服务没有鉴权,默认只绑 `127.0.0.1`,不要直接暴露公网。
- 随仓库发布的数据是上游 Apache-2.0(许可文本已附),**本项目自己的代码暂时没有选许可证**,这个留给你定。

## 测试

```bash
pip install -e '.[test]'
python -m pytest -q      # 26 passed
```

---

### English summary

`decisionkit` is a numpy-only reference implementation of typed decision
inference: give it a state and typed questions (`boolean` / `choice` / `score`)
and it returns a full probability distribution per question, with zero decoded
tokens and no GPU or downloaded weights. It ships a trainable hashed-feature
scorer, per-primitive temperature calibration, metrics (accuracy, macro-F1,
Brier, NLL, ECE, coverage, latency), a loopback HTTP service, a stdlib client,
and an importer for the public `LocalLLaMA/typed-decisions` benchmark.

On that benchmark's official test split (400 cases, 2000 decisions) the linear
CPU baseline agrees with the teacher argmax on **61.5%** of decisions, against
**79.25%** published for the 0.6B GPU decision model on the same split. The
point of the number is the cost: no weights to download, ~90 seconds of CPU
training, ~12 ms per case. Temperature calibration leaves the argmax alone,
cuts ECE from 0.106 to 0.063, and turns the 0.6 threshold into a 37% coverage /
78% precision operating point. The synthetic suite is only a pipeline check.
