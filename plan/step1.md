# 核心世界引擎实现计划

## 概要

- 使用 Python + ModernGL + venv 从空仓库搭建核心世界引擎。
- 只实现世界引擎：不做地形生成、角色、怪物、法术和玩法系统。
- 新增 enginedemo 调试入口和 HTTP 控制台。
- 所有物质、光、气体、反应表在初始化时必须通过 CPU 侧编辑接口写入 GPU/规则系统，用来验证 CPU 编辑链路。
- 目标性能：demo 在常规窗口尺寸下稳定 60 FPS 以上。

## 核心接口与架构

- 建立 oracle_game 包，包含世界状态、规则表、GPU 资源、模拟 pass、渲染、分页、CPU-GPU 桥接、HTTP 控制台和 demo。
- 定义稳定数据模型：
    - CellCore: material_id, phase, cell_flags, velocity_xy, cell_temperature, timer_pack, integrity。
    - MaterialTable, GasSpeciesTable, LightTypeTable, MaterialOpticsTable, 六类反应表和定长 ReactionBuffer。
    - WorldCommand, ReadbackRequest, ReadbackResult, ForceSource, FallingIslandRecord, PageStripeUpdate。

- CPU 侧公开编辑与注入接口：
    - update_material_table(...)
    - update_light_type_table(...)
    - update_material_optics_table(...)
    - update_gas_species_table(...)
    - update_reaction_table(...)
    - inject_material(...)
    - inject_temperature(...)
    - inject_force(...)
    - request_readback(...)

- 游戏初始化流程不得直接硬编码 GPU 表内容；必须走上述 CPU 编辑接口，把首版物质、光、气体、反应、光学参数写入运行时表。

## 实现内容

- 项目启动：
    - 创建 venv 兼容的 Python 项目配置。
    - 依赖使用 moderngl, moderngl-window, numpy, pytest。
    - 提供 python -m oracle_game.enginedemo 和 enginedemo 命令。

- 世界状态：
    - 实现固定大小 GPU 活动窗口和逻辑坐标到物理坐标映射。
    - 使用 tile = 32 x 32, chunk = 8 x 8 tiles。
    - ActiveChunkMask 和 ActiveTileTTL 独立于 CellCore。
    - 实现环形分页窗口，stripe 加载/保存先做可调用 stub。

- 规则表：
    - 首版物质包含 game_plan.md 中列出的土石、植物、金属、水、危险液体、特殊功能物质。
    - 首版气体为 air, water_gas, poison_gas, oil_gas, pollution_gas。
    - 首版光为 visible_light, holy_light, chaos_light, magic_light。
    - 反应表使用不对称语义：“如果邻居匹配 X，则自己执行 Y”。
    - 支持 emit_material, emit_light, modify_gas, convert_material, modify_temperature, harm。

- 模拟 pass：
    - 气体：共享低分辨率速度场，advection，divergence，Jacobi pressure，projection，species 平流/扩散/衰减。
    - 温度：格子热传导、环境温度 Jacobi 扩散、格子与环境换热、相变。
    - 反应：材材、材气、材光、气气、气光、自反应；光光反应保留但关闭。
    - 运动：粉体和高速材料 DDA reservation/resolve，重力、风场、稀疏力源。
    - 液体：shared memory tile 下填/找平、边界修正、粉体浮沉、placeholder 排液钩子。
    - 崩塌：事件触发 JFA 支撑传播，把失支撑结构转为 FallingIsland。
    - 光学：typed DDA ray traversal，写入 VisibleIllumination, CellOpticalDose, GasOpticalDose。
    - CPU-GPU 通信：PBO 双缓冲读回，一帧延迟，不做同帧阻塞读回。

- HTTP 控制台：
    - 在 demo 中启动本地 HTTP 服务，默认绑定 127.0.0.1。
    - 提供 JSON API：
        - 写入物质区域。
        - 读取局部 CellCore。
        - 读取温度、气体、光学 dose、速度场。
        - 编辑物质表、光表、气体表、反应表。
        - 注入温度、气体、力源、光源。
        - 暂停、继续、单步、调速、重置世界。

    - HTTP 写入通过同一套 CPU 命令队列进入 GPU，不开旁路改世界状态。

- 渲染与 demo：
    - 每种物质走贴图 atlas 渲染路径，v1 贴图用纯色。
    - 物质纹理按世界坐标重复平铺，只显示对应物质占据区域。
    - 渲染光学系统结果，能看出不同光的颜色和 dose 强弱。
    - enginedemo 支持数字键选择物质、鼠标绘制、调试视图切换、暂停、单步、重置、刷子大小和模拟速度。

## 性能与通信优化

- GPU pass 只处理 active chunk/tile，避免全图高频扫描。
- PBO 读回只允许小区域请求，读回结果延迟一帧消费。
- CPU 表编辑采用批量上传，避免每项参数单独提交 GPU。
## 测试计划

- 单元测试：
    - 初始化确实通过 CPU 编辑接口写入物质、光、气体和反应表。
    - 所有首版物质、气体、光类型存在。
    - material_id = 0 是唯一空格表达。
    - 反应表保持不对称语义。
    - convert_material 先按 harm_per_frame 损伤，再按阈值转换。
    - 环形分页坐标映射正确。
    - PBO ping/pong 不读取当前帧写入槽。

- 集成测试：
    - 启动 enginedemo。
    - HTTP 写入物质后，读回能看到对应区域改变。
    - HTTP 编辑表后，新规则在后续帧生效。
    - 水、毒液、酸液、油能按液体规则下填/找平。
    - 气体浓度非负，并能随共享流场移动。
    - 温度扩散和基础相变能运行。
    - 火焰、酸、毒、圣光、可见光、混沌光的首版反应能触发。
    - 荧光粉能发出对应光类型。
    - 失支撑固体能转为 FallingIsland。
    - 默认 demo 场景稳定 60 FPS 以上。

## 假设

- “对面的物质/光/反应”按“前面计划中定义的物质、光、反应表等规则数据”理解。
- game_plan.md 没给精确数值的反应，先用可编辑的保守默认数值，不写死在 shader 里。
- HTTP 控制台只绑定本机，不做认证系统。
- 不实现真实地形生成，只用 demo 初始场景和手动/HTTP 写入。
- 不实现角色、敌人、法术、LLM 系统，只保留未来接入接口。