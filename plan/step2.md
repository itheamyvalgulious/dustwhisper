# Step 2: GPU 主链路优化计划

## 目标

把当前 `enginedemo` 的 1920x1080、全 active、随机材料 GPU 路径从约 `145ms/frame` 压到可继续逼近 60 FPS 的结构。第一阶段目标不是靠降低分辨率、减少 active 区域、跳过 solver 或降低视觉质量，而是删除当前主链路中的重复状态搬运、重复发布、重复全屏写入和不该每帧阻塞的结构计算。

重要前提：只做“各阶段候选输出 + 最后合并一次”不会自动带来 40-60ms。这个改造本身主要删除重复桥接、发布和阶段依赖，不会减少核心物理计算。要达到大幅降耗，必须同时拆掉每帧阻塞的连通性计算、合并重复 reaction、删除中途权威世界发布，并重写反应和液体里的重复全屏状态搬运。

## 当前证据

最新有效 profile：

- `avg_ms`: 约 `144.78ms`
- `p95_ms`: 约 `146.75ms`
- GPU: `NVIDIA GeForce GTX 1650/PCIe/SSE2`
- `skipped_gpu_stages`: `[]`
- `readbacks_completed`: `0`

主要耗时：

- `reactions after optics`: 约 `28.82ms`
- `collapse`: 约 `26.73ms`
- `reactions before motion`: 约 `24.99ms`
- `liquid`: 约 `20.31ms`
- `motion`: 约 `20.00ms`
- `heat`: 约 `12.30ms`
- `optics`: 约 `6.04ms`
- `gas`: 约 `2.63ms`

这个结果说明当前瓶颈不是单个 GPU shader 的算术强度，而是主循环把多个系统当成串行世界修改器执行：每个系统读取权威世界、写临时结果、发布回权威世界，下一个系统再读刚发布的世界。

## 当前主链路问题

当前主循环顺序：

1. `collapse`
2. `gas`
3. `heat`
4. `reactions before motion`
5. `liquid_pre_motion_intent`
6. `motion`
7. `liquid`
8. `optics`
9. `reactions after optics`
10. `latch_clear`
11. `active_decay`

主要结构问题：

- `collapse` 是结构连通性判断，不应该每帧阻塞主链路。
- reaction 被拆成光学前和光学后两段，材光反应强依赖本帧 optics，导致一帧内多跑一整段 reaction。
- reaction 内部的 `timed/self/material_material/material_gas/material_light/gas_gas/gas_light` 多次读取、写出、发布完整状态。
- liquid 的核心 `liquid_tile_solve` 只有约 `1.87ms`，但整个 liquid 约 `20.31ms`，主要损耗来自 `load_bridge_inputs`、中间 copy、seam copy、placeholder copy 和 publish。
- heat 也在独立读取、生成目标、应用目标、发布状态，很多工作应和 reaction 的 cell 更新或最终合并共享。
- 当前 bridge 权威状态在一帧内被多次更新，导致数据流必须串行。

## 新数据流

每帧开始固定上一帧输入：

- 上一帧最终 `cell_core`
- 上一帧气体场
- 上一帧温度或 `cell_core` 内温度字段
- 上一帧光照结果
- 最近一次完成的 collapse 连通性结果

各系统只读上一帧输入，分别写完整候选输出：

- reaction 写 reaction 候选 `cell_core`，必要时写 gas 候选。
- heat 写 heat 候选 `cell_core` 或只写温度/相变相关字段。
- motion 写 motion 候选 `cell_core`。
- liquid 写 liquid 候选 `cell_core`。
- gas 写 gas 候选。
- optics 写下一帧使用的光照结果。
- collapse 独立异步更新最近一次可用连通性结果。

最后只做一次固定优先级合并：

- 读上一帧 `cell_core`。
- 读 reaction、heat、motion、liquid 候选。
- 按固定优先级写下一帧最终 `cell_core`。
- 无改动格子保持上一帧状态。

合并规则第一版：

- 反应导致的删除、生成、相变优先。
- FallingIsland 和粉体运动次之。
- 液体运动再次。
- heat 默认只覆盖温度、完整性、相态等字段，不随意覆盖 material。
- 无候选结果时保留上一帧。

不引入 changed mask。每个阶段遍历自身需要处理的 active tile 或全屏区域时直接查看格子内容。最终合并也直接读候选结果和上一帧状态，不靠额外 mask 决定是否访问。

## 必须真正减少的工作

### 1. collapse 脱离每帧阻塞

当前 `collapse` 约 `26.73ms`。它不应每帧阻塞主链路。

改法：

- 平均每 4 帧调度一次 collapse。
- 当前帧只使用最近一次已经完成的 collapse 结果。
- 如果新结果还没完成，就继续使用旧结果。
- 地形修改只标记 dirty 区域，不强制本帧等 collapse 完成。

预期主链路直接省接近 `20ms+`，具体取决于调度和结果消费方式。

### 2. reaction 从两段改成一段

当前：

- `reactions before motion`: 约 `24.99ms`
- `reactions after optics`: 约 `28.82ms`

改法：

- 材光反应用上一帧光照触发。
- 本帧 optics 输出供下一帧 reaction 使用。
- 删除为了本帧 optics 再跑一次 reaction 的结构。
- `timed/self/material_material/material_gas/material_light/gas_gas/gas_light` 在同一个 reaction 阶段内组织。

预期节省不是因为候选输出，而是因为不再一帧跑两段 reaction。

### 3. reaction cell 侧融合

当前 reaction 内部多组反应都像独立世界更新器：

- 上传或加载状态。
- 跑 shader。
- 写完整状态。
- 发布 cell state。
- 处理 deferred action 和 gas side effect。

改法：

- cell 侧 reaction 使用同一份上一帧 `cell_core` 输入。
- 尽量把 `self/timed/material_material/material_light` 的 cell 更新合并成一个或少数几个 shader。
- 每格只写一次 reaction 候选 `cell_core`。
- gas side effect 可以先保留独立，因为它写低分辨率 gas grid 且涉及原子累积。
- 不再每个 reaction group 都 publish 回 bridge。

目标是减少多次全屏状态读写和多次完整 cell state 发布，而不是只改变输出名字。

### 4. liquid 删除中途 bridge/copy

当前 liquid 约 `20.31ms`，但核心 `liquid_tile_solve` 约 `1.87ms`。

明显可疑开销：

- `liquid_load_bridge_inputs`: 约 `3.63ms`
- `liquid_copy_for_placeholder`: 约 `1.76ms`
- `liquid_copy_tile_solve`: 约 `1.65ms`
- `liquid_copy_seam_x`: 约 `1.61ms`
- `liquid_publish_bridge`: 约 `1.54ms`
- `liquid_buoyancy_sink/float`: 各约 `1.63ms`

改法：

- 液体读上一帧 `cell_core`，直接写 liquid 候选。
- 删除中途为了下个阶段可见而做的 publish。
- 尽量删除 tile solve 到 seam、placeholder 之间的完整状态 copy。
- buoyancy 只处理相关物态，不应把大量非液体格子当作常规工作。

目标不是减少液体 tile solve，而是删除液体链路里非物理核心的状态搬运。

### 5. heat 与 reaction/merge 共享数据流

当前 heat 约 `12.30ms`，其中 `load_bridge_inputs`、`publish_bridge_outputs`、`apply_cell_targets`、`apply_condense_cells` 都是明显串行世界修改器结构。

改法：

- heat 读上一帧状态。
- 输出 heat 候选字段。
- 相变结果交给最终合并按固定优先级处理。
- 如果 shader 规模可控，可把部分 cell heat 更新并入 reaction cell shader。

### 6. motion 暂时保持独立，但禁止中途依赖其他本帧结果

motion 约 `20.00ms`，不是第一刀最大问题，但必须改成读上一帧输入、写 motion 候选。

改法：

- motion 不再依赖本帧 liquid_pre_motion_intent 的即时结果。
- 如果需要液体流向辅助，使用上一帧流向结果或由 liquid 候选在下一帧影响 motion。
- reservation/resolve/apply 内部继续单独优化，避免按 reservation 或 cell 做大范围重复扫描。

## OpenGL/ModernGL 限制

ModernGL 当前使用 OpenGL compute。单 context 下 `program.run(...)` 是按 OpenGL 命令流提交的有序命令。ModernGL 暴露的是 `compute_shader.run()`、`ctx.memory_barrier()`、`ctx.finish()`，没有 Vulkan 那种显式多 compute queue。

因此不能依赖 `reaction.run(); liquid.run(); heat.run();` 在硬件上可靠并行。驱动可能内部做一定 overlap，但不能作为架构保证。

可以手动融合 shader，但有限制：

- 适合融合：同样按 cell 遍历、读上一帧状态、写 cell 候选的反应和部分热更新。
- 不适合硬塞：液体 tile shared memory、seam、buoyancy、粉体 reservation/resolve/apply，因为它们需要不同 dispatch 结构和不同同步边界。
- 一个 compute shader 内没有全屏所有 workgroup 的同步点，不能在一个 shader 里完整执行多个全局阶段再合并。

## 新 GPU API 方向

长期正式方向建议是 Vulkan。

理由：

- Vulkan 有显式 queue、command buffer、semaphore、barrier。
- 可以表达多个独立 compute 工作读同一输入、各自写输出、最后等待并合并。
- Vulkan 同时能负责渲染，避免 OpenGL 和计算 API 互操作复杂度。
- CUDA 性能和工具最好，但锁 NVIDIA，适合作为 NVIDIA 性能对照，不适合作为唯一正式后端。
- OpenCL 语义能表达 out-of-order queue 和 event 依赖，但生态、调试和跨平台驱动质量不如 Vulkan 适合作为游戏底座。

建议路线：

1. 短期继续在 ModernGL/OpenGL 中改数据流，验证去掉重复桥接和主链路阻塞后的收益。
2. 中期做 Vulkan compute 原型，只搬 `cell_core`、reaction、heat、merge，验证 queue 和 GPU timestamp。
3. 长期迁移 Vulkan 全模拟和渲染。
4. CUDA 只作为 NVIDIA 上的性能上限参考后端。

## 阶段性落地顺序

### 阶段 A：修主链路调度

- collapse 改成固定间隔非阻塞。
- reaction 删除光学前/光学后双段结构。
- 材光反应用上一帧光照。
- 保留现有内部 shader，先验证主链路时序变化。

验收：

- 主链路不再每帧等待 collapse 完整求解。
- profile 中不再出现两段大 reaction。
- 不能跳过 solver，只能改变依赖和调度。

### 阶段 B：reaction cell 侧融合

- 合并 cell 侧反应输出。
- 删除 reaction group 之间的中途 publish。
- 最终只写一次 reaction 候选。

验收：

- `reactions.*_publish_cell_state` 大幅减少。
- `reactions.*_upload_state.load_bridge_*` 大幅减少。
- reaction 总耗时接近当前最大单组反应加少量额外开销，而不是多组相加。

### 阶段 C：候选输出和固定合并

- reaction、heat、motion、liquid 写各自候选。
- 新增固定优先级合并 pass。
- 一帧内只发布一次最终 `cell_core`。

验收：

- 每帧 bridge 权威 `cell_core` 更新点收敛到最终合并。
- 合并 pass 目标约 `1ms` 级。
- 不通过 changed mask 做稀疏合并。

### 阶段 D：liquid/heat/motion 内部降耗

- liquid 删除中间完整 copy 和 publish。
- heat 改成候选字段或并入 reaction。
- motion reservation/resolve/apply 继续按 profile 优化，避免大范围重复扫描。

验收：

- liquid 总耗时不再被 copy/publish 主导。
- heat 不再独立发布完整世界状态。
- motion 内部热点由 pass 级 profile 直接指导。

## 预期

只做候选输出和最后合并，不足以到 40-60ms。

合理预期必须建立在以下改动同时完成：

- collapse 不再每帧阻塞。
- reaction 不再分两段。
- reaction cell 侧不再多次完整写出和发布。
- liquid/heat 不再中途桥接发布。
- 最终只做一次固定优先级合并。

完成这些以后，才有资格把主帧目标设为 `40-60ms` 并继续向 60 FPS 压。若这些结构性问题不改，只换 API 或只加候选输出，都不会解决当前 100ms 级浪费。
