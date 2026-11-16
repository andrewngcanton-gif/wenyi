# 最终验收操作指南（人工评分只做一次）

本文件只描述评估工具链。它不改变产品默认行为：`reflective_polish` 仍为默认关闭；在正式门槛
全部通过前，不实施、也不建议默认启用 WP3。

## 现在已自动化

- **预注册冻结**：`plan`/`run` 在翻译开始前把实验协议（比较组、聚类单位、留出章节、随机种子、
  重采样方法、全部门槛）写入 `experiment_protocol.frozen.json`，并记录内容 SHA-256。之后
  `export`/`import`/`report` 只验证该绑定，不重写。协议在产出后被改动即判为无效预注册。
- **盲评材料**：`samples/pairwise.csv`（匿名左右偏好、漏译/增译、MQM 错误数、严重误译）和
  `samples/edit_review.csv`（候选相对自身初译的改前/改后盲评）。真实组别只在单独的 key 文件。
- **聚类配对统计**：按预注册的章节/整书/连续块重采样；报告聚类数、胜/负/平、可评/缺失分母、
  点估计、95% 区间、随机种子和方法。全部有效偏好同向（全胜/全负）、候选缺陷为零或确认
  改坏事件为零时，区间为 unavailable，相应门槛为 inconclusive。聚类大小不等时，聚类层面
  的二项精确界不能证明合并段落的错误率界；全零 bootstrap 也不能作为通过证据。
- **全部门槛**：偏好、MQM 非劣化、漏译、增译、严重新增误译、改坏率、货币费用、完整性、
  证据冻结、留出复核、多重比较校正。逐项输出 passed / failed / inconclusive 及原因。
- **报告**：`report/report.json`、`report/summary.md`、`report/human_summary.csv`、
  `report/group_summary.csv`。工作流是否跑通与经验质量结论分开显示。

自动化测试、离线适配器或 HTTP stub 只能验证计算和边界，**永远不能当作经验质量通过**。

## 人工只在最终阶段做一次

评审者只填写两个 CSV，不修改 `pair_id`/`edit_id`、原文或译文：

1. `samples/pairwise.csv`：`preference`（left/right/tie/skip）、`omission`、
   `addition`、`left_mqm_errors`/`right_mqm_errors`（非负错误计数；空白=未评）、
   `left_severe`/`right_severe`（yes/no；空白=未评）、`notes`。
2. `samples/edit_review.csv`：`worsened`（yes/no/skip；空白=未评）、`severity`（low/medium/high）、
   `notes`。

空白、skip、缺失一律按未知处理，不补零、不插补。评分绑定导出时的原文/译文哈希；若之后重新
导出或重新翻译导致文本变化，导入会被整体拒绝。联合导入先验证两个 CSV，任一文件无效时两个
文件均不提交，已有评分和导入摘要保持原样。报告重新校验磁盘上当前保存的协议、key、文本与
评分绑定，不能凭先前导入成功或摘要中的状态继续认定有效。评分来源是调用方自报，工具无法验证评审者身份，
因此不能称为“双盲”，只能称“匿名随机盲评”。

## 正式运行前仍需准备

- 版权清晰的源书，以及清楚记录的许可证与来源。
- 三个独立代码绑定：上游基线、冻结的 WP0、干净且已固定的候选 SHA。C/C_keep/D 绑定候选。
- 用户配置中的模型档案；D 还需要 `--heterogeneous-reflect-model` 指定的异构档案。
- provider 价格 JSON：状态 known、日期、币种、单位价格；缺价格时成本为 unknown，不会按 0 计算。
- 预算与完整工作量；不完整的运行不能用于成本比。
- 在看结果之前确定最终留出章节并在协议中声明。章节必须实际存在于源书清单中，与主样本
  分离并留下选样记录；release 还需要这些留出段落的真实配对评分、足够聚类及偏好、MQM、
  严重误译检查证据。仅把 `sampling.holdout.verified` 设为 true 不能证明完成留出复核。
- 协议中的门槛是 DRAFT 建议，必须由负责人确认后改成 `status: registered`；DRAFT 不能运行。

在协议、模型档案、许可数据和预算准备齐全之前，不启动付费正式研究；人工评分保留到最终
材料冻结后统一进行。当前工具链准备不构成经验质量结论。

## 命令

```bash
# 1) 生成 DRAFT 协议模板（只作建议，不是已批准门槛）
uv run --project tools/eval python -m wenyi_eval.run_variants protocol-template \
  --out /absolute/prereg/protocol.draft.json

# 2) 人工确认阈值、比较组、聚类、留出和种子后，把 status 改为 registered

# 3) 预注册并执行（翻译前冻结协议）
uv run --project tools/eval python -m wenyi_eval.run_variants run \
  --source /absolute/licensed/book.epub --license CC-BY-4.0 \
  --source-language ja --target-language zh \
  --groups A_prime,C,D --heterogeneous-reflect-model <profile> \
  --protocol /absolute/prereg/protocol.json \
  --baseline-code /absolute/read-only/upstream \
  --wp0-code /absolute/read-only/wp0-baseline \
  --candidate-code /absolute/candidate --candidate-sha <clean-sha> \
  --runs-dir /absolute/disposable/eval-runs --prices /absolute/prices.json \
  --candidate-executable /venv/bin/python --candidate-argv=-m --candidate-argv=wenyi_cli \
  --wp0-executable /venv/bin/python --wp0-argv=-m --wp0-argv=wenyi_cli \
  --baseline-executable /venv/bin/python --baseline-argv=-m --baseline-argv=wenyi_cli

# 4) 评审者填写两个 CSV 后导入并重新出报告
uv run --project tools/eval python -m wenyi_eval.run_variants import \
  --root /absolute/eval-root/book
uv run --project tools/eval python -m wenyi_eval.run_variants report \
  --root /absolute/eval-root/book
```

`export --root <root>` 会从当前保存的译文重建盲评包，并更新包的绑定；即使文本相同，也需要
重新导入评分。评审一旦开始，不要重新导出。只有确实需要新一批材料时才重新执行 `run`，
旧评分不能迁移到新文本。

## 结果位置

- `experiment_protocol.frozen.json`：冻结内容、哈希、冻结时间。
- `manifest.json`：代码/输入/配置/模型/种子/价格身份。
- `samples/`：盲评 CSV、私有 key、导入摘要（人工评分只在这里）。
- `report/summary.md`、`report/report.json`：逐门槛结论、区间、分母和原因。
- `report/human_summary.csv`：每个门槛一行的机器可读摘要。
- `report/group_summary.csv`、`report/cost.json`、`report/stage_costs.csv`：覆盖、失败、
  成本、准备与 Review 分账。

工具链验收应另存精确命令、协议哈希、逐门槛状态、导入分母和负向控制结果，并保留隔离的
离线运行产物。至少验证：保存的译文改变后拒绝旧评分、第二份表非法时两份评分均保持原样、
保存中断或文件损坏时阻断通过，以及重新导出后旧绑定失效。这些合成数据只能证明工具链行为，
不能替代最终的人工质量评分。

## 未解决证据如何阻断默认建议

任何 required 门槛为 `inconclusive` 或 `failed` 时，整体质量结论就不是 `passed`：

- 缺人工评分、缺 MQM/严重误译标注、缺改坏率样本 → 不通过。
- 协议缺失、DRAFT、被改动或晚于产出写入 → 不通过（缺失为 inconclusive，篡改为 failed）。
- 聚类不足、试验样本过小、全部有效偏好同向、零事件导致区间 unavailable、只有 pilot，或
  缺真实留出选样/评分证据 → inconclusive，不能通过。
- 价格/用量/币种/完整性缺失，或 A′ 正文成本为零 → 不通过。
- 任一严重新增误译、偏好显著劣势、改坏率或漏译/增译越界 → failed。

因此当前默认行为保持：产品反思润色关闭；不以自动指标或 LLM 裁判单独宣称提升；在真实门槛
通过前不启用 WP3。200–300 段 pilot 只用于检验流程与方向，不能推断整书或所有语对质量。
