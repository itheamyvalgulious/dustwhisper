# 新 GPU 落砂重构计划书

## 第一章. 世界本体与模拟总纲

### 1.1 本章目的

本章定义新 GPU 落砂世界的本体结构。所谓“本体结构”, 指的是这个世界里到底有哪些基础存在, 它们各自住在哪里, 以什么形式更新, 以及哪些东西属于世界真相, 哪些东西只是求解过程中的临时量。

这一章不讨论具体技能列表, 不讨论 UI, 不讨论 CPU 侧角色逻辑, 只定义世界模拟层的最底座。后续所有章节, 包括渲染、角色碰撞、分页、法术系统、反应系统, 都必须建立在这里定义的本体约束之上。

新架构的目标不是把旧实现换一套写法, 而是彻底放弃“相邻格交换 CA”作为核心世界观。新的世界观是:

- 世界仍然是网格驱动的, 因为这最适合 GPU 并行、局部更新和直接渲染。
- 但格子不再只是“物质编号”, 而是承载局部状态的物质实例。
- 并不是所有东西都住在格子里。气体与光学是独立传播载体。
- 世界中的运动、反应、相变、发射、崩坏, 都必须以明确的子系统边界表达, 不能回到单体 shader 中混写。

### 1.2 世界由三类正式载体组成

本项目的世界只承认三类正式传播载体:

#### 1.2.1 格子物质载体

格子是世界的主载体, 负责承载以下东西:

- 静态固体。
- 粉体。
- 液体。
- 可移动的特殊物质。
- 局部反应状态。
- 局部计时器。
- 局部速度状态。

格子是稠密、全覆盖、分页持久化的主体世界。任何会构成地形、会被挖掘、会阻挡运动、会参与动态求解、会发生局部崩坏的物质, 都必须最终落在格子层。

#### 1.2.2 气体载体

气体不住在格子里。它属于独立连续场系统, 由以下两部分组成:

- 一个共享背景流场。
- 多个气体 species 浓度场。

所有气体共享同一个背景流场, 这是正式设计决策。不同气体之间的区别, 不通过不同风场表达, 而通过各自的浓度场、扩散率、沉降率、温度耦合和反应规则表达。

气体与格子世界通过以下方式耦合:

- 相变。
- 反应。
- 热交换。
- 源项注入与汇项消耗。

#### 1.2.3 光学载体

光不是格子的附属数值, 也不是气体的一个特例。它是第三种正式传播载体。

不同类型的光拥有不同的传播、衰减、穿透、折射、反应和作用规则。光可以绑定自身反应, 也可以作为反应触发条件作用于格子和气体。

因此, 光学系统不是“亮度贴图”, 而是一个带类型的传播系统。

### 1.3 格子物质的物态定义

格子层中的物质不再只按“材料种类”区分, 还必须按“物态”区分。当前第一版明确支持以下核心物态:

- `StaticSolid`, 静态固体。
- `Powder`, 粉体。
- `Liquid`, 液体。
- `FallingIsland`, 崩坏后的整体下落块。

后续如果需要, 可以扩展出更特殊的可移动物态, 但第一版的核心重心就是这四类。

#### 1.3.1 静态固体

静态固体是地形骨架。它的基本定义是:

- 平时不参与逐帧运动。
- 不直接进入高频运动求解。
- 不直接受重力更新速度。
- 主要通过破坏、加热、侵蚀、相变或崩坏转化为其他物态。

静态固体的存在意义不是“只是不能动的材料”, 而是把“世界骨架”和“可流动物质”从一开始就拆开。这样一来, 大部分地形不会被纳入高频运动求解, GPU 主流水线的负担会显著下降。

#### 1.3.2 粉体

粉体是可运动的离散颗粒物态。它的基本特征是:

- 参与动态求解。
- 会受到重力影响。
- 可在局部堆积、滑落、坍塌。
- 会受到风场有限影响。
- 可由静态固体在破坏或崩坏时转化而来。

粉体不是“固体的附带效果”, 而是一个正式的动力学物态。沙子、碎石、灰烬、部分冰晶或魔法碎屑, 都应优先用粉体模型表达。

#### 1.3.3 液体

液体也是格子物质, 但它的运动机制独立于粉体。液体的主行为不是“单个粒子逐步位移”, 而是“局部区域内的快速重排与找平”。

因此液体:

- 可以保留局部速度或流动倾向。
- 但主更新不依赖纯粒子式逐格迁移。
- 而是依赖 tile 内 shared memory 收集、重排、回填。
- 必要时只在 tile seam 做轻量校正。

液体从一开始就与气体走两条不同的模拟路线。液体依然是高分辨率格子占据物质, 而不是低分辨率连续流体。

#### 1.3.4 FallingIsland

`FallingIsland` 是由崩坏触发产生的特殊物态。它的基本定义是:

- 仍然占据格子空间。
- 以整体平移方式运动。
- 第一版不处理旋转。
- 运动期间保持对其他物质的占位阻挡。
- 在失去整体性时可进一步解体为 `Powder`。

它不是常规材料的常驻物态, 而是静态固体在失去支撑后进入的过渡阶段。

### 1.4 静态固体与粉体必须严格分离

这是本架构的一个核心原则。

旧式落砂常把“石头”和“沙子”都看成某种相邻交换粒子, 只是移动概率不同。这种做法会导致两个问题:

- 地形骨架也被迫进入高频运动判定, 浪费大量算力。
- 世界的“崩塌”感会很弱, 因为所有东西都只是慢慢滑动, 而不是结构性失稳后整体碎解。

新架构中, 静态固体和平时运动的粉体必须被视为两个不同层级的存在:

- 静态固体代表“仍然作为结构存在的材料”。
- 粉体代表“已经失去结构支撑、转为颗粒运动的材料”。

因此, 一个石质拱顶、土墙、悬崖、立柱在正常情况下都属于静态固体。只有当它受到足够破坏, 或者失去必要支撑关系时, 它才不再被视为结构, 而先转化为 `FallingIsland`, 并在后续根据整体性决定是否进一步解体为 `Powder`。

这一定义直接决定了“崩坏系统”不是可选特效, 而是静态固体生命周期的正式阶段。

### 1.5 崩坏机制是结构判定, 不是逐格下落

#### 1.5.1 崩坏的定义

崩坏不是“某格脚下为空就开始掉”, 那只是粉体规则。

崩坏的真正定义是: 静态固体必须与某些合法支撑区域保持连通, 否则它就失去结构资格, 必须转化为 `FallingIsland` 或其他可动态处理的物态。

也就是说, 静态固体之所以能悬空, 不是因为它“恰好这一格下面没空”, 而是因为它仍然属于某个受支撑的结构整体。

#### 1.5.2 支撑的概念

本项目必须显式定义“支撑源”或“合法连通目标”。第一版至少应支持以下几类支撑源:

- 世界底部基岩或不可破坏根层。
- 明确标记为永久支撑的结构材料。
- 与地图边界或人工锚点连通的区域。
- 后续可扩展的特殊魔法支撑节点。

支撑不是局部邻接概念, 而是全局或区域连通概念。一个高悬平台是否稳定, 取决于它最终是否还能通过某条固体连通路径连接到合法支撑源。

#### 1.5.3 崩坏触发

崩坏判定不应每帧全图运行。它应当是事件驱动加区域调度的系统。典型触发条件包括:

- 地形被挖断。
- 爆炸摧毁承重区域。
- 某材料因受热、腐蚀、相变而失去结构属性。
- 支撑锚点被移除。
- 某些法术改变了材料的结构资格。

当这些事件发生时, 系统只把相关区域标记为“结构待重算”, 而不是立即全世界做一次支撑求解。

### 1.6 JFA 连通判定是崩坏系统的正式求解器

JFA 在这里不是拿来做视觉特效, 而是作为静态固体支撑传播求解器。它的目标不是求精确最短路径, 而是快速判定:

- 哪些静态固体仍然与支撑源连通。
- 哪些静态固体已经脱离支撑网络。
- 哪些区域应被转化为粉体或其他崩坏态。

崩坏求解阶段至少需要以下输入:

- 当前静态固体占据 mask。
- 支撑源 mask。
- 不可传播区域或切断边界。
- 待检测区域边界。

JFA 或其后续归约结果至少要输出:

- `SupportedMask`, 仍然连到合法支撑源的静态固体。
- `UnsupportedMask`, 已失去支撑的静态固体。
- 必要时输出岛屿标识或区域标签, 供后续分批崩坏处理使用。

JFA 虽然适合 GPU, 但它解决的是结构合法性, 不是高频局部运动。因此:

- 它不应该进入每帧主循环的常规 pass 序列。
- 它应由地形修改事件触发。
- 它可以按 dirty region、dirty chunk、dirty stripe 调度。
- 它甚至可以延迟 1-数帧执行, 只要崩坏视觉表现允许存在短暂结构迟滞。

第一版在这里明确接受近似误判。也就是说, JFA 的目标是稳定地给出“足够合理的结构断裂表现”, 而不是在细桥、薄层、狭缝支撑上追求数学上严格精确的连通结果。某些很细的连接被误判为断开, 在当前版本中属于可接受的视觉后果。

当某片静态固体被判定为 `Unsupported` 后, 它不应再继续以静态固体身份存在。系统必须先把它转化为 `FallingIsland`。

`FallingIsland` 的定义是: 一片由崩坏产生的、暂时脱离主网格结构语义、以整体平移方式下落的连通块。第一版中, `FallingIsland` 只做整体平移下落, 不处理旋转。

因此, 崩坏后的正式流程是:

- 静态固体失去支撑后, 先整体转化为 `FallingIsland`。
- `FallingIsland` 作为一个独立下落块整体运动。
- 当它重新稳定落地后, 再根据后续规则回到主网格语义。

对于松散结构, 额外引入一条解体规则:

- 如果一个 `FallingIsland` 的足够多邻居不再属于 `FallingIsland`, 则该部分失去整体性, 转化为 `Powder`。

这意味着崩坏后的结构并不总是完整落地。它可以在下落过程中逐步松散, 从整体块体演化为粉体云或碎渣堆。

因此, 崩坏后的结果不再是“直接从静态固体变成粉体”, 而是优先经历 `StaticSolid -> FallingIsland` 这一阶段, 再根据整体性与邻域条件决定是否进一步 `FallingIsland -> Powder`。

### 1.7 反应载体的正式定义

当前版本已经明确, 世界只承认三种反应载体:

- 绑定在格子物质上的反应。
- 绑定在气体上的反应。
- 绑定在光上的反应。

这是一条非常强的设计约束。以后任何新玩法机制, 最终都必须回答“它到底附着在哪个传播载体上”。

#### 1.7.1 格子反应

格子反应附着在固体、粉体、液体这些离散物质上。典型例子包括:

- 燃烧。
- 冻结。
- 腐蚀。
- 净化。
- 石化。
- 崩坏准备态。
- 法术附着状态。

格子反应的优点是位置精确、可持久化、可直接参与碰撞和相变。缺点是传播半径天然局部。

#### 1.7.2 气体反应

气体反应附着在某个 species 的浓度场上。典型例子包括:

- 毒雾扩散。
- 烟尘沉降。
- 冰雾降温。
- 圣雾净化。
- 可燃气积聚与爆燃。

气体反应适合大尺度扩散、卷动、稀释和跨区域渗透。

#### 1.7.3 光反应

光反应附着在光学传播本身上。典型例子包括:

- 热光加热。
- 冷光冻结。
- 圣光驱魔。
- 可见光照明与显形。
- 特定频段触发材料反应。

光反应的优势在于高速传播、方向性强、可穿透特定介质或被特定材质滤波。

### 1.8 发射不是第四种本体, 而是物质触发的反应

本项目不把 emitter 设计成独立世界对象系统。发射行为应被定义为:

某种物质在特定条件下, 通过反应系统向气体系统、光学系统或局部反应系统注入源项。

这意味着:

- 火焰物质可以持续发出热与可见光。
- 毒池物质可以持续向气体场注入毒雾 species。
- 圣石物质可以持续发出圣光。
- 低温结晶物质可以持续释放冷光或冷气。
- 一个飞行中的“圣光弹”, 本质上只是一个会移动、并持续触发 `emit_light` 之类反应的特殊物质团。

这样处理的优点是统一:

- 移动发射源不需要独立对象同步。
- 发射行为天然跟随材质移动。
- 不引入第二套“发射器实体”的分页、存档和桥接系统。

因此, 发射是反应系统的一部分, 不是独立本体类别。

### 1.9 力只作为稀疏输入存在, 不构成正式场层

在本章最终版本中, 力系统被明确收缩为少量稀疏源直接作用模型。

世界不维护统一的全局力场图, 也不为力建立一套持久场层。力只以以下形式存在:

- 少量稀疏局部力源。
- 爆炸冲量。
- 风扇式方向推力。
- 局部排斥或吸引。
- 其他明确受数量约束的空间驱动。

这些力源在速度更新时被按需采样并叠加到局部速度上。因此, 力在本架构中的定位是:

- 它不是第四种传播载体。
- 它不是正式连续场。
- 它只是运动求解阶段的外部输入项。

这条限制有助于长期控制系统复杂度, 防止“什么效果都加一张场图”导致底层失控。

### 1.10 世界主状态与工作缓存必须分层

这一条虽然听起来像实现细节, 实际上也属于本章内容, 因为它定义了“什么才是世界”。

#### 1.10.1 世界主状态

以下内容属于正式世界状态:

- 格子物质主状态。
- 气体浓度场。
- 共享背景流场。
- 必要的光学持久状态或传播基底。
- 物质规则表和类型表引用。
- 能被分页、保存、读回和重建的内容。

#### 1.10.2 工作缓存

以下内容不属于世界真相, 只属于求解临时量:

- 动态求解的 reservation buffer。
- 原子抢位结果。
- JFA 传播工作缓冲。
- 气体 pressure ping-pong。
- 液体 LDS 共享内存统计。
- 光传播前沿。
- 各类临时 mask 和 reduction buffer。

这条边界必须从第一章就写死。因为后面分页系统、读回系统、存档系统、调试系统, 都必须围绕这条边界展开。

### 1.11 本章确立的正式设计决策

截至目前, 第一章已经正式确立以下长期决策:

1. 世界只有三类正式传播载体: 格子物质、气体、光学。
2. 固体与粉体是不同物态, 静态固体平时不参与运动。
3. 静态固体只有在破坏、失稳或崩坏后, 才先转化为 `FallingIsland`, 并在失去整体性时进一步转化为 `Powder`。
4. 崩坏的本质是支撑连通性失效, 而不是局部脚下为空。
5. 崩坏连通求解采用 JFA 路线, 并且不进入每帧主循环; 第一版允许细连接上的近似误判。
6. 所有气体共享一个背景流场, 但各自拥有独立浓度 species。
7. 反应载体只分三类: 绑定格子、绑定气体、绑定光。
8. emitter 不是独立本体, 而是物质规则的一部分。
9. 力不是正式场层, 只作为少量稀疏外部输入存在。
10. 世界主状态与求解工作缓存必须严格分层。

## 第二章. 数据模型与光学机制

### 2.1 本章目的

本章把第一章确定的世界本体继续落到具体数据层。这里先定义逻辑数据结构, 不急于绑定最终的 GPU 贴图数量与精确位宽。也就是说, 这一章回答的是“世界必须保存什么数据, 每种数据的语义是什么”, 而不是“现在就固定成几张图”.

本章的 3 条总原则如下:

- `material_id = 0` 直接表示空格, 不单独设置 `Empty` phase。
- `phase` 只描述运动与占位语义, 不描述“有没有东西”.
- 所有跨系统通用的动态状态应优先保留固定语义, 不再使用语义过于模糊的通用 `aux` 名称。

### 2.2 CellCoreGrid

`CellCoreGrid` 是全分辨率的主格子状态。它是分页、存档、渲染、局部读回和大部分模拟 pass 共同依赖的核心数据层。

这里有一个明确约束: `CellCoreGrid` 不承载 `active` 调度状态。`active` 属于 chunk/tile 级调度元数据, 不是单格世界真相。

当前建议的逻辑字段如下:

- `material_id`
- `phase`
- `cell_flags`
- `velocity_xy`
- `cell_temperature`
- `timer_pack`
- `integrity`

#### 2.2.1 `material_id`

- `material_id = 0` 表示空格。
- 非零值表示某种真实材料实例。
- 材料的大部分共性参数不放在格子里, 而通过 `MaterialTable` 查表获得。

这样做的直接结果是: “有没有东西”由 `material_id` 决定, “它如何运动”由 `phase` 决定。

#### 2.2.2 `phase`

`phase` 只负责运动语义和占位语义。当前正式相位为:

- `StaticSolid`
- `Powder`
- `Liquid`
- `FallingIsland`

这里不再允许 `Empty` phase, 因为空格已经由 `material_id = 0` 唯一表示。

把 `FallingIsland` 放进 `phase` 是必要的。原因不是为了分类美观, 而是为了让崩坏后的整体块在下落过程中继续占据主网格空间, 从而阻止其他物质、液体或粉体错误地流入其体内。

#### 2.2.3 `cell_flags`

`cell_flags` 是少量跨系统共享的单格标志位。它们的特点是:

- 语义固定。
- 高频读取。
- 不依赖具体材料表解释。

首版建议只保留极少数真正跨系统的标志, 并直接固定 bit 含义:

- `bit0 = phase_locked`, 当前帧不允许常规规则改写 `phase`。
- `bit1 = reaction_latched`, 当前格本帧已经进入过一次 reaction resolve, 用于防止同帧循环触发或重复写入。
- `bit2 = recently_converted`, 当前格刚发生过材料转换, 供后续 pass 做跳过或补充处理。
- `bit3..bit7 = reserved`, 暂不分配给具体玩法。

这类标志不应该膨胀成第二套自定义状态机, 否则会把 `cell_flags` 重新用成模糊的万能状态包。

#### 2.2.4 `velocity_xy`

- 用于描述可动物质的局部速度。
- `Powder` 和 `FallingIsland` 明确使用它。
- `Liquid` 可以部分使用它作为局部流动倾向, 但不要求液体主求解完全服从它。
- `StaticSolid` 平时不更新它。

#### 2.2.5 `cell_temperature`

格子自身温度必须跟随材料一起移动, 因此它属于格子主状态, 不能只放在空间固定场里。

这意味着:

- 一块被加热过的炭块移动后, 热量会跟着炭块一起走。
- 一团被冰冻的材料飞出后, 低温也会跟着材料一起走。
- 格子温度在热交换 pass 中同时和周围格子、气体环境温度发生交换。

因此, 第二章正式区分两种温度:

- `cell_temperature`, 绑定在材料格子上, 跟着材料移动。
- `ambient_temperature`, 绑定在环境空间上, 由气体系统维护。

#### 2.2.6 `timer_pack` 与 `integrity`

这里正式取消 `aux0` / `aux1` 这种过于模糊的命名。当前 `CellCore` 中不再保留第二个模糊“状态包”, 而是只留下一个固定用途的 `timer_pack`, 再直接存一份当前 `integrity`。

当前建议是:

- `timer_pack` 作为“reaction 计时槽”.
- `integrity` 作为“当前完整度”.

其中:

- `timer_pack` 固定为 `4 x 8-bit reaction timer`, 可以记为 `rt0`, `rt1`, `rt2`, `rt3`。
- `integrity` 直接表示这个格子当前剩余的完整度数值。

这里进一步明确两者的边界:

- `timer_pack` 当前只服务 reaction 系统。
- 材料固定拥有 `8` 个 reaction slot, 其中 `reaction_slot0..3` 是带 timer 的 timed slot, `reaction_slot4..7` 是不带 timer 的 untimed slot。
- `rt0` 到 `rt3` 与 `reaction_slot0` 到 `reaction_slot3` 一一绑定。
- 每个 timed slot 只使用自己对应的 timer, 不共享、不抢占、不动态分配。
- `reaction_slot4..7` 没有独立 timer, 只要条件成立就会在每帧持续触发。
- `integrity` 不通过“基础值减累计伤害”间接表示, 而是直接记录当前值。

之所以仍然保留两个独立字段, 原因是:

- `timer_pack` 已经有明确职责, 就是 reaction 计时。
- `integrity` 是世界真状态, 应当像温度一样直接存取。
- 把 reaction timer 和完整度强塞进同一套解释逻辑里, 只会让两套系统互相耦合。

因此, 当前版本正式取消 `state_pack`。如果未来某类材料确实需要更大的每格局部状态, 再单独引入稀疏扩展池, 而不是提前在 `CellCore` 里预埋一个空泛状态包。

这里再定一条硬边界: `CellCore` 必须保持全材料统一格式, 不允许“不同 `material_id` 使用不同核心布局”.

原因很直接:

- 几乎所有全图 pass 都要无分支地读取 `CellCore`。
- 如果核心布局按材料变化, 每个 pass 都要先判断格式再解码, 会严重伤害实现复杂度和吞吐。
- 真正材料特有的数据, 要么放到 `MaterialTable`, 要么放到稀疏扩展池, 不能挤进主格子格式里。

因此, 第二章正式不把 `active`、`active_ttl` 或类似调度字段塞进 `CellCoreGrid`。
同理, `FallingIsland` 的 `island_id` 以及实体注入格子的 `entity_id` 这类按格归属状态, 默认也不挤进 `CellCore` 主字布局, 而是放在并行维护的辅助归属缓冲中。

#### 2.2.7 首版推荐打包方案

为了让实现阶段更直接, 当前推荐的 `CellCore` 首版物理打包如下:

- `Word0 = material_id:u16 | phase:u8 | cell_flags:u8`
- `Word1 = velocity_pack = packHalf2x16(velocity_xy)`
- `Word2 = cell_temperature:f32`
- `Word3 = timer_pack:u32`
- `Word4 = integrity:u16 | reserved0:u16`

这意味着 `CellCore` 首版是 5 个 32-bit word, 共 160 bit。

这不是理论最小值, 但它的优点是:

- 字段职责清楚。
- 实现简单。
- 读写路径稳定。
- 后续如需压缩, 也有明确的压缩目标。

### 2.3 MaterialTable

`MaterialTable` 是全系统的规则中心。格子里只保存 `material_id`, 其余大部分共性参数全部从这里查表。

当前应至少包含以下规则分组。

#### 2.3.1 身份与基础分组

- `material_id`
- `name`
- `default_phase`
- `render_group`
- `base_integrity`

这里正式把 `integrity` 提升为材料公共属性。具体含义是:

- `base_integrity` 定义该材料生成时写入 `CellCore.integrity` 的默认值, 也可视为常规上限。
- 格子本地直接保存当前 `integrity`。
- `harm`、撞击、崩坏或特殊规则都直接修改这份当前 `integrity`。

各字段的角色再明确如下:

- `default_phase`, 该材料在生成、刷入、或纯转换后默认进入的相位。
- `render_group`, 渲染层按哪一套可视规则解释这类材料。
- `base_integrity`, 该材料的公共完整度上限。

#### 2.3.2 运动规则

- `density`
- `gravity_scale`
- `wind_coupling`
- `drag_scale`
- `friction`
- `elasticity`
- `max_dda_step`
- `powder_solver_kind`
- `liquid_solver_kind`
- `falling_island_break_kind`

这里的重点是: 材料不直接决定“每帧怎么写 shader”, 而是决定“在某类求解器里如何响应”.

这组字段的具体含义如下:

- `density`, 决定重力竞争、粉体在液体中的浮沉趋势, 以及液液或液粉相遇时的相对轻重关系。
- `gravity_scale`, 该材料受全局重力的倍率。
- `wind_coupling`, 该材料被共享背景流场加速的强弱。
- `drag_scale`, 该材料局部速度衰减的强弱。
- `friction`, 与支撑面、障碍物或液体接触时的切向耗能强度。
- `elasticity`, 法向碰撞后的回弹系数。
- `max_dda_step`, 单帧内允许申请的最大位移格数, 用于限制 DDA 目标距离。
- `powder_solver_kind`, 该材料进入 `Powder` 相位时采用哪一类粉体求解规则。
- `liquid_solver_kind`, 该材料进入 `Liquid` 相位时采用哪一类液体求解规则。
- `falling_island_break_kind`, 整体下落块在边缘剥落、撞击或邻域稀疏化时采用哪类破碎规则。

#### 2.3.3 结构规则

- `is_structural`
- `is_support_anchor`
- `collapse_behavior`
- `collapse_generation`
- `powder_generation`

这组字段决定它能不能作为静态结构存在, 失稳后进入哪种崩坏流程, 以及最终会不会更容易解体为粉体。

这里各字段的角色是:

- `is_structural`, 该材料是否允许以 `StaticSolid` 身份参与连通与支撑判定。
- `is_support_anchor`, 该材料是否可作为 JFA 支撑源。
- `collapse_behavior`, 失去支撑后是立刻进入 `FallingIsland`, 延迟崩塌, 还是完全免疫常规崩塌。
- `collapse_generation`, 开始崩塌时切换成的材料类型, 用于把“完整石块”转成“下落石块”等过渡材料。
- `powder_generation`, `FallingIsland` 进一步碎裂成粉体时使用的最终材料类型。

#### 2.3.4 热学规则

- `heat_capacity`
- `conductivity`
- `ambient_exchange_rate`
- `melt_point`
- `boil_point`
- `melt_to_material`
- `freeze_to_material`
- `boil_to_gas_species`

这里正式采用你的修正: 固液相变只保留一个 `melt_point`, 不再把 `freeze_point` 单独拆出。也就是说:

- 当温度高于 `melt_point` 时, 固体可熔化为液体。
- 当温度重新低于 `melt_point` 且规则允许时, 液体可凝固回固体。

如果未来确实需要熔化和凝固的迟滞差异, 再另加偏置字段, 但当前版本先不预埋双阈值。

另外:

- `heat_capacity`, 材料温度被同样能量输入改变的难易程度。
- `conductivity`, 与邻格材料交换热量的效率。
- `ambient_exchange_rate`, 与环境温度场交换热量的效率。
- `melt_point`, 固液相变阈值。
- `boil_point`, 用于液体转气体, 或某些固体高温挥发的阈值入口。
- `melt_to_material`, 熔化后应该变成什么材料。
- `freeze_to_material`, 凝固后应该变成什么材料。
- `boil_to_gas_species`, 沸腾后应该向哪一种气体 species 注入源项。

#### 2.3.5 反应规则引用

- `material_tag_mask`
- `gas_tag_mask`
- `light_tag_mask`
- `reaction_slot0`
- `reaction_slot1`
- `reaction_slot2`
- `reaction_slot3`
- `reaction_slot4`
- `reaction_slot5`
- `reaction_slot6`
- `reaction_slot7`

这里不直接定义反应结果, 只定义这类材料会参与哪些反应表匹配。

更具体地说:

- `material_tag_mask`, 供材材反应与部分自反应做标签匹配。
- `gas_tag_mask`, 供材气反应与气体吸收、腐蚀、催化等逻辑匹配。
- `light_tag_mask`, 供材光反应匹配不同光学后果。
- `reaction_slot0..7`, 该材料固定拥有的 8 个反应槽位。
- `reaction_slot0..3` 是 timed slot, 永远与 `timer_pack` 中的 `rt0..rt3` 一一对应。
- `reaction_slot4..7` 是 untimed slot, 没有对应 timer, 只按条件逐帧持续触发。
- `reaction_slotN` 的内容是指向 `ReactionBuffer` 的固定索引, 即“这个材料第 N 个反应到底做什么”。
- 相变不占用 reaction slot, 它走通用材料规则。

这里也正式取消“材料自带发射规则字段”的设计。`emit_material`、`emit_light`、气体注入等行为统一通过反应表命中后调用 `ReactionBuffer` 动作完成。

#### 2.3.6 推荐首版字段清单

为了让 `MaterialTable` 更接近可实现状态, 当前推荐首版至少包含下列字段:

- `material_id`
- `default_phase`
- `render_group`
- `density`
- `gravity_scale`
- `wind_coupling`
- `drag_scale`
- `friction`
- `elasticity`
- `max_dda_step`
- `is_structural`
- `is_support_anchor`
- `collapse_behavior`
- `collapse_generation`
- `powder_generation`
- `base_integrity`
- `heat_capacity`
- `conductivity`
- `ambient_exchange_rate`
- `melt_point`
- `boil_point`
- `melt_to_material`
- `freeze_to_material`
- `boil_to_gas_species`
- `material_tag_mask`
- `gas_tag_mask`
- `light_tag_mask`
- `reaction_slot0`
- `reaction_slot1`
- `reaction_slot2`
- `reaction_slot3`
- `reaction_slot4`
- `reaction_slot5`
- `reaction_slot6`
- `reaction_slot7`

这份清单表达的是逻辑 schema, 不是最终物理排布。实现时应坚持:

- 逻辑上所有字段都由 `material_id` 索引。
- 物理上可以拆成多张只读表或多个结构化缓冲, 例如运动表、热学表、结构表、反应表。
- 拆表的目的只是提高缓存局部性和带宽效率, 不能改变字段语义。

### 2.4 温度与气体系统

#### 2.4.1 环境温度

除了 `cell_temperature` 外, 世界还需要一层环境温度:

- `ambient_temperature`

它属于环境空间而不是材料本体, 主要服务于:

- 气体浮力与热上升流。
- 热空气与冷空气分层。
- 材料与空气的热交换。
- 区域热积累与散热。

#### 2.4.2 GasSystem

`GasSystem` 当前正式由以下部分组成:

- `flow_velocity_xy`, 所有气体共享的背景流场。
- `ambient_temperature`, 环境温度场。
- `gas_concentration[species_id]`, 各种气体 species 的独立浓度层。
- `solver_scratch`, 例如 `divergence`, `pressure_ping`, `pressure_pong` 等工作缓冲。

这里要明确两点:

- 气体 pressure 默认只是 solver scratch, 不是世界长期主状态。
- 液体是否需要额外压强辅助量, 当前还没有正式定死, 不提前塞进气体主字段。

#### 2.4.3 GasSpeciesTable

每种气体 species 至少需要:

- `species_id`
- `name`
- `diffusion_rate`
- `buoyancy`
- `decay_rate`
- `temperature_coupling`
- `condense_point`
- `condense_to_material`
- `material_reaction_tag_mask`
- `light_reaction_tag_mask`

这使得多种气体能共享一套背景流场, 但仍保持各自独立的反应与衰减特性。

### 2.5 FallingIsland 与稀疏运行时对象

由于 `FallingIsland` 已经进入 `phase`, 它在主网格中必须持续占位。与此同时, 系统仍需要一份稀疏对象表来保存“这一整块如何运动”.

当前先只锁定最小字段:

- `island_id`
- `bbox`
- `velocity_xy`
- `subcell_offset`

其中:

- 网格中的 `phase = FallingIsland` 表示该格当前属于某个下落块。
- 网格侧还必须能回答“这一格当前属于哪个 island”, 因此需要一份按格可查询的 `island_id` 状态。
- `bbox` 用于快速更新和碰撞裁剪。
- `subcell_offset` 用于累积非整数速度导致的位移余量。

这一版暂时不锁死更复杂的字段, 例如转动惯量、角速度、精确局部形状缓存等。`island_id` 默认放在与 `CellCore` 并行的辅助归属缓冲中, 不强迫把它塞进每格 `CellCore` 主字布局, 但它必须是 GPU 上可按格直接查询的正式状态。

本章仍保留少量 `ForceSource` 作为稀疏输入对象, 但它不是世界正式场层, 字段只需满足位置、方向、半径、强度和寿命即可。

### 2.6 光学系统

光学系统不是一张“亮度图”, 而是一个带类型的传播与沉积系统。光的长期结果会回写成温度、气体变化、材料反应或渲染照明, 但光本身不作为永久占据场留在世界里。

#### 2.6.1 LightTypeTable

每一种光类型至少需要:

- `light_type_id`
- `name`
- `visual_channel`
- `default_range`
- `max_bounce`
- `dose_channel_id`
- `render_style`

这里的关键是: 光类型定义的是“传播与反应类别”, 而不只是颜色。
第一版先把 `light_type` 上限正式收束到最多 `8` 种, 以控制 dose 通道数、显存占用和写带宽。

#### 2.6.2 MaterialOpticsTable

材料对光的传播属性不写在一般 `MaterialTable` 里, 而是写在一张独立的二维表:

- `MaterialOpticsTable[material_id][light_type_id]`

每个条目至少包含:

- `absorption`
- `scattering`
- `refraction`

这 3 个量描述的是传播行为本身, 而不是传播后触发什么玩法反应。也就是说:

- 材料对光的吸收、散射、折射属于光学属性。
- 光照到材料后会不会燃烧、冻结、净化、显形, 属于反应系统, 不写在这里。

#### 2.6.3 OpticalEmitterBuffer

每帧先收集本帧反应系统产出的发光请求, 再生成临时 `OpticalEmitterBuffer`。它不是独立世界实体, 而是由 `emit_light` 这类反应动作临时导出的运行时缓冲。

每个发射项至少需要:

- `light_type_id`
- `origin`
- `direction`
- `spread`
- `strength`
- `duration`

这里的 `duration` 指“该发光反应在触发后会持续多少帧继续发射”, 而不是生成一个脱离材料存在的独立长期 emitter 实体。只要对应 reaction slot 仍处于持续触发状态, 光学系统就会在每帧重新收集并生成本帧 emitter 请求。

这里的 `direction` 可以支持:

- `all`
- `up`
- `down`
- `left`
- `right`
- `speed`, 跟随材料当前速度方向

#### 2.6.4 光传播主过程

当前建议的第一版光学机制是 typed ray traversal。其流程如下:

1. 收集本帧 `emit_light` 动作产生的发光请求。
2. 生成本帧 `OpticalEmitterBuffer`。
3. 对每个 emitter 发出对应类型的 ray。
4. ray 在格子世界中逐步穿行。
5. 每命中一个格子, 查 `MaterialOpticsTable[material_id][light_type_id]`。
6. 根据吸收、散射、折射系数拆分本次能量。
7. 把被吸收的 dose 写入沉积缓冲。
8. 把散射或折射后的剩余能量继续传播, 直到强度过低或超出距离上限。

如果写成局部守恒形式, 一次命中后可视作:

\[
E_{absorb} = E \cdot a,\quad
E_{scatter} = E \cdot s,\quad
E_{refract} = E \cdot r,\quad
E_{remain} = E - E_{absorb} - E_{scatter} - E_{refract}
\]

其中 \(a, s, r\) 来自材料对该光类型的光学系数。

#### 2.6.5 OpticalDoseBuffers

当前版本不把“玩法光”和“可见光”拆成两套不同本体。每种光类型都在同一次 typed ray traversal 中同时产出玩法 dose 和渲染可见结果。

光学传播结束后, 至少需要保留 3 类结果:

- `VisibleIllumination`, 供画面和可见性逻辑使用。
- `CellOpticalDose[dose_channel_id]`, 供格子反应系统使用。
- `GasOpticalDose[dose_channel_id]`, 供气体反应系统使用。

这里的关键不是再做第二次传播, 而是让同一批命中结果同时服务玩法和渲染。实现上, 每个 `light_type` 都有自己独立的 dose 通道; 渲染层再按 `render_style` 解释这些类型化结果。

#### 2.6.6 光光反应

当前版本默认不做 `Light-Light` 反应。也就是说:

- 两束光默认只是在空间中叠加 dose 和可见度。
- 不主动做干涉、湮灭、合成等光光反应。

如果未来确实需要极重的魔法化光学, 再单独打开 `LightLightReactionTable`。

### 2.7 反应表体系

当前正式采用“传播属性和反应结果分离”的做法。传播属性由材料光学表、气体属性表、温度与运动规则决定; 真正的“遇到什么触发什么结果”全部放到反应表里。

当前建议的反应表如下:

- `MaterialMaterialReactionTable`
- `MaterialGasReactionTable`
- `MaterialLightReactionTable`
- `GasGasReactionTable`
- `GasLightReactionTable`
- `MaterialSelfReactionTable`

另外:

- `LightLightReactionTable` 预留为可选表, 默认关闭。

#### 2.7.1 配对反应表的通用字段

每条配对反应规则至少需要:

- `lhs_selector`
- `rhs_selector`
- `phase_filter`
- `min_temperature`
- `max_temperature`
- `trigger_kind`
- `threshold`
- `rate`
- `consume_policy`
- `result_ref`

其中 `trigger_kind` 可以是:

- 邻接接触
- 重叠接触
- 光照 dose 超阈值
- 气体浓度超阈值
- 温度区间
- 定时触发
- 完整度归零

这里的 `min_temperature` 与 `max_temperature` 是所有反应表共享的通用温度门槛。相变专用阈值仍由材料的 `melt_point` 和 `boil_point` 定义; 反应表上的温度区间用于描述“什么条件下触发该反应”, 例如灼烧、冷凝、净化或腐朽。

`result_ref` 在当前版本分两类解释:

- 如果结果作用在 material cell 上, 则它解释为 `target_side + trigger_slot_index`, 也就是“触发该材料的第几个固定反应槽位”。
- 如果结果作用在纯气体或纯光学载体上, 则它可以直接解释为 `ReactionBuffer` 中的动作引用。

#### 2.7.2 `MaterialSelfReactionTable`

材料自反应表用于描述不依赖第二载体就能发生的变化, 它本质上是“本地条件 -> 触发哪个 reaction slot” 的映射。典型情况包括:

- 计时器归零后衰变。
- 周期性自反应触发 `emit_material` 或 `emit_light`。
- 完整度归零后的碎裂、销毁或崩坏。
- 某些材料自身的冷却、熄灭、腐朽。

它的匹配条件主要来自:

- `material_id`
- `phase`
- `cell_temperature`
- `timer_pack`
- `base_integrity`
- `integrity`

它的结果不直接写动作, 而是写:

- `trigger_slot_index`

这样 `rt0..rt3` 和 `reaction_slot0..3` 的绑定就始终稳定, 不会出现多个反应争抢同一个 timer 的问题。与此同时, `reaction_slot4..7` 这 4 个 untimed slot 也可以由同一套条件匹配直接持续触发, 不再额外区分“边沿型 instant”与“持续型 instant”.

### 2.8 ReactionBuffer 与反应动作描述项

反应表本身只负责匹配条件。真正执行什么结果, 应由一套可复用的动作描述项定义。

当前正式引入一份只读 `ReactionBuffer`, 用于存放所有具体反应描述项。

每个反应项的逻辑结构是:

- `reaction_type`
- `payload`

其中:

- `reaction_type` 从固定动作枚举中选择。
- `payload` 根据 `reaction_type` 解释为对应参数块。

反应表命中后, 不直接内联写死结果, 而是外链到 `ReactionBuffer` 中的某个反应项。

当前版本正式采用定长 `ReactionBuffer` 结构。也就是说:

- 每条反应项占用固定大小的槽位。
- 未使用参数位允许留空。
- 不为节省少量显存而引入变长解析复杂度。

这样做的目标是:

- 简化 GPU 侧读取逻辑。
- 避免变长 payload 带来的分支和地址跳转复杂度。
- 让 reaction descriptor 在实现上更接近结构化查表, 而不是小型解释器。

在当前版本里, `ReactionBuffer` 的主要消费者是材料固定拥有的 `reaction_slot0..7`。也就是说:

- `MaterialTable.reaction_slotN` 指向 `ReactionBuffer` 中的一条动作描述。
- 若 `N` 位于 `0..3`, 则 `timer_pack.rtN` 只服务这条动作描述。
- 若 `N` 位于 `4..7`, 则该槽没有独立 timer, 条件成立时每帧直接触发。
- 触发系统只决定“何时触发第 N 槽”, 不再临时拼一个任意 reaction chain。
- 同一格同一帧若多个槽同时满足条件, 则按统一顺序全部执行, 不因某个槽已经满足就跳过其他槽。

当前先正式支持以下动作原语:

#### 2.8.1 `emit_material`

- `material`
- `duration`
- `speed`
- `velocity`
- `direction`

这里的 `duration` 同样表示“持续触发多少帧”. 若该动作被 timed slot 触发, 则由对应 `rt` 驱动其持续期; 若被 untimed slot 触发, 则只要条件仍成立, 就会逐帧持续发射。

`direction` 当前至少支持:

- `random`
- `up`
- `down`
- `left`
- `right`
- `speed`

#### 2.8.2 `emit_light`

- `type`
- `duration`
- `strength`
- `direction`
- `range`

这里的 `range` 指传播开角或光束范围, 而不是简单意义上的距离上限。
`duration` 的语义与 `emit_material` 一致: 它表示持续触发期, 而不是生成独立 emitter 实体。

#### 2.8.3 `modify_gas`

- `type`
- `speed`
- `duration`

其中:

- 正值表示注入。
- 负值表示吸收。
- `duration` 表示该气体修改动作的持续触发期; timed slot 通过 `rt` 驱动, untimed slot 只要条件成立就会逐帧重复执行。

#### 2.8.4 `convert_material`

- `material`
- `generation`

其中:

- `material = none` 表示销毁当前格子材料。
- `generation` 用于控制生成世代或后续衍生限制。

#### 2.8.5 `modify_temperature`

- `delta`
- `duration`

这类动作既可以作用在格子材料温度上, 也可以作用在环境温度上, 具体由反应表上下文决定。
`duration` 的解释与前述一致: 它表示持续触发期, 而不是独立后台过程。

#### 2.8.6 `harm`

- `value`
- `duration`

`harm` 的作用是直接减少目标格子的 `integrity`。当 `integrity <= 0` 时, 该格子后续如何碎裂、销毁或转化, 由材料自己固定的 `8` 个 reaction slot 中满足条件的槽位来处理。

### 2.9 `integrity`

完整度系统在当前版本中的正式定义如下:

- `base_integrity` 是材料公共属性, 存在于 `MaterialTable` 中。
- `integrity` 是格子局部动态状态, 直接存在于 `CellCore` 中。
- 新格子生成时, 默认把 `integrity` 初始化为该材料的 `base_integrity`。
- `harm`、撞击和其他规则直接减少 `integrity`。
- 当 `integrity <= 0` 时, 由该材料自己的 reaction slot 处理破坏后果, 而不是走单独的“累计伤害反推”机制。

这样一来:

- 不同材料可以共用统一的“受损 -> 破坏”框架。
- 材料之间仍然可以有完全不同的耐久度上限和归零后果。
- 烧坏、砸坏、腐蚀坏都可以统一归约为对 `integrity` 的减少。

### 2.10 本章确立的正式设计决策

截至目前, 第二章已经正式确立以下决策:

1. 空格由 `material_id = 0` 表示, 不设置 `Empty` phase。
2. `phase` 只描述运动和占位语义, 当前包含 `StaticSolid`, `Powder`, `Liquid`, `FallingIsland`。
3. `FallingIsland` 必须继续占据主网格空间, 并通过 `phase = FallingIsland` 保持阻挡语义; 同时需要有可按格查询的 `island_id` 状态, 默认放在与 `CellCore` 并行的辅助归属缓冲中。
4. 格子自身温度和环境温度必须分离: `cell_temperature` 跟随材料移动, `ambient_temperature` 绑定在空间上。
5. `CellCore` 不再使用模糊的 `aux0/aux1` 命名, 当前保留 `cell_flags`、`timer_pack` 和 `integrity`。
6. `timer_pack` 固定为 4 个 reaction timer; 材料固定拥有 `8` 个 reaction slot, 其中 `reaction_slot0..3` 与 `rt0..rt3` 绑定, `reaction_slot4..7` 为无 timer 的持续触发槽位。
7. 当前正式取消 `state_pack`。若未来确实需要更大的材料局部状态, 再走稀疏扩展。
8. 格子的当前完整度直接存为 `integrity`, 不再使用 `damage_accum` 这类累计伤害反推方案。
9. 材料热学公共属性当前至少包含 `melt_point`、`boil_point` 以及对应的相变目标, 相变属于通用材料规则, 不走 reaction。
10. 所有反应表统一带 `min_temperature` 和 `max_temperature` 作为通用温度门槛。
11. 发射不再写进 `MaterialTable`; `emit_material`、`emit_light`、气体注入等行为统一通过反应表命中后调用 `ReactionBuffer` 动作完成。
12. 材料对光的传播属性采用独立的 `MaterialOpticsTable[material_id][light_type_id]`, 至少包含吸收、散射、折射三项。
13. 光传播属性与反应结果分离: 光学表负责传播, 反应表负责玩法后果; 同一套 typed ray traversal 同时产出玩法 dose 和渲染结果。
14. 当前正式反应表包含材材、材气、材光、气气、气光、自反应 6 类, 光光反应只做可选预留。
15. `ReactionBuffer` 中不再支持 `chain` 类型; 每个反应项只描述一个确定动作。
16. `ReactionBuffer` 采用定长槽位结构, 不使用变长 payload。
17. 第一版 `light_type` 数量上限正式收束到最多 `8` 种。

## 第三章. 固有相互作用

### 3.1 本章目的

本章定义世界中各种载体之间天然存在、且不依赖具体技能脚本的相互作用规则。这里的重点不是某个法术做什么, 而是世界本身在热、相变、气体、液体、固体和实体参与下会自然发生什么。

本章的目标是建立“世界为什么会这样演化”的底层规则边界, 并为第四章中的各个 GPU pass 提供明确职责。

### 3.2 物质的相变规则

相变是材料在温度、环境和完整度共同作用下改变物态或材料类型的过程。它不属于单纯渲染效果, 而是正式的世界状态转化。

当前版本中, 相变至少包括以下几类:

- 固体熔化为液体。
- 液体凝固为固体。
- 液体汽化为气体源项。
- 气体冷凝回液体或固体材料。
- 特殊材料在热或冷条件下转化为另一种材料。

#### 3.2.1 熔化与凝固

每种材料通过 `melt_point` 定义固液相变基准。

- 当 `cell_temperature` 高于 `melt_point` 且材料允许熔化时, 触发固体到液体的转化。
- 当液体温度重新回落到 `melt_point` 以下且周围条件允许时, 可触发液体到固体的回转。

这里不默认引入凝固和熔化双阈值迟滞。若以后需要特殊材料行为, 再通过额外材料规则补充。

#### 3.2.2 汽化与冷凝

液体与气体之间的相变不直接把“气体”塞回格子, 而是通过格子材料和气体浓度场之间的源汇交换完成。

- 液体或固体在高温下按自身材料的 `boil_point` 与 `boil_to_gas_species` 通用规则, 把自身部分质量转换成某种气体 species 的浓度注入。
- 气体在满足自身 `condense_point` 以及局部环境条件时, 可按 `condense_to_material` 通用规则反向在格子层生成液体或固体材料。

因此, 气体与格子之间的相变本质上属于“材料减少 / 生成 + 浓度场增减”的双侧交换。

#### 3.2.3 特殊转化

某些相变不只是物态变化, 而是材料变化。例如:

- 木头被烧成炭。
- 冰晶被圣光净化成透明晶体。
- 腐蚀后的金属转化为脆化残渣。

这类行为不再算基础相变, 而是特殊材料转化, 统一通过反应表和 `convert_material` 实现。

### 3.3 气体与格子物质的相互影响

气体与格子世界之间的关系不是单向“穿过去”, 而是双向耦合。

#### 3.3.1 气体对格子物质的影响

气体可以通过以下方式影响固体、粉体和液体:

- 改变材料温度。
- 触发腐蚀、毒化、净化、冻结等反应。
- 对轻质粉体施加风耦合与浮力影响。
- 改变液体表面附近的蒸发或冷凝速率。

这里的关键约束是: 气体对格子的机械影响主要通过共享流场和反应系统间接实现, 而不是给每种气体都建立一套独立的力学系统。

#### 3.3.2 格子物质对气体的影响

格子物质也会持续影响气体系统:

- 高温材料向环境释放热量。
- 可挥发材料通过通用沸腾/挥发规则向气体场注入 species。
- 冷源材料降低周围环境温度。
- 某些材料吸收、过滤或催化特定气体。

这部分主要通过通用热相变规则和材气反应表共同完成。

### 3.4 热交换

热交换是世界中最基础的连续过程之一。它存在于 3 个层面:

- 格子与格子之间。
- 格子与环境之间。
- 光或气体带来的热量沉积。

#### 3.4.1 格子与格子之间

相邻格子之间按材料导热率交换热量。热交换不依赖材料是否可移动, 静态固体、粉体、液体都参与。

主规则可以抽象为:

\[
Q_{cell-cell} \propto k \cdot (T_{neighbor} - T_{self})
\]

其中 \(k\) 由材料导热能力决定。

#### 3.4.2 格子与环境之间

每个格子同时与所在位置的 `ambient_temperature` 做热交换。这样:

- 热石头会加热周围空气。
- 冷物体会在热空气中逐渐回温。
- 大面积热源可以诱发局部热空气上升。

#### 3.4.3 光与气体带来的热量沉积

光学系统和气体系统都可以向格子或环境写入温度变化:

- 热光可直接升高材料温度。
- 冷光可降低材料温度。
- 高温气体可提升接触材料和环境的温度。
- 某些气体反应可以放热或吸热。

因此, 热交换是跨格子、气体、光学三个载体的统一桥梁。

### 3.5 液体与固体/粉体之间的力学关系

液体与固体、粉体之间的相互作用不走完整连续体流固耦合, 而走适合格子世界的离散规则近似。

当前第一版至少明确以下行为:

- 液体受固体边界约束而铺平、积蓄和泄流。
- 固体阻挡液体占据。
- 液体可沿固体表面流动。
- 液体可对固体施加局部侵蚀、冷却或加热作用。
- 液体对浸没其中的 `Powder` 施加基于密度的浮力。

其中“液体与固体之间的力”在第一版里主要体现为:

- 固体边界对液体流向的约束。
- 液体在接触表面对局部材料触发反应。
- 液体对 `Powder` 的浮沉通过材料密度和液体 solver 的规则实现。

当前不把“液体推动整块刚体”作为基础能力。

#### 3.5.1 液体对 `Powder` 的浮力

液体中的 `Powder` 不应只按空气中的重力规则处理, 而应额外受到液体介质的浮力影响。

基本原则是:

- `Powder` 的有效沉浮由其材料密度与当前液体密度比较决定。
- 密度更大的 `Powder` 倾向于下沉。
- 密度更小的 `Powder` 倾向于上浮。
- 接近中性的 `Powder` 可表现为悬浮、缓慢沉降或缓慢上浮。

这类浮力不要求引入完整连续体浮力方程, 而是在粉体求解和液体重排之间通过规则化的垂向偏置来实现。

#### 3.5.2 液体是否需要独立“流动”

液体不能只有静态找平而完全没有流动倾向, 否则会出现以下问题:

- 瀑布和喷流缺少方向惯性。
- 斜坡泄流会显得像瞬移重排。
- 实体或外力扰动后的水体缺少余波和继续流走的趋势。

因此第一版的正式结论是:

- 液体的主求解器仍然是“局部重排与找平”。
- 但液体仍保留有限的局部流动倾向。
- 这种流动倾向不实现为独立的高成本液体场, 而是通过通用 DDA 运动阶段实现, 液体在该阶段以随机或半随机方向尝试有限位移。
- 当前版本不引入液体压力求解, 未来如有需要再追加。

也就是说, 液体需要“流动”, 但不需要一套独立于重排逻辑之外的完整流体运动框架。

### 3.6 实体与环境中的力

这里的“实体”指角色、敌人和其他 CPU 侧主导的稀疏对象。

实体不进入格子世界做逐格材料求解, 但它们必须感受到环境中的力和阻力。

#### 3.6.1 实体受哪些环境影响

实体至少应受以下因素影响:

- 重力。
- 气体背景流场造成的风推力。
- 稀疏 `ForceSource` 造成的局部推拉。
- 液体中的阻力与浮沉倾向。
- 温度、毒气、圣光等造成的持续状态影响。

#### 3.6.2 实体与格子环境的耦合方式

实体不直接成为格子物质, 但会通过以下方式与环境耦合:

- 实体的真实形状与局部损伤状态由外部实体系统维护, 这是权威状态。
- 每帧模拟前, CPU 按当前实体形状把对应格子注入 `CoreCell`, 并为这些注入格子附带可查询的 `entity_id` 状态。
- 这些实体格子阻挡液体和常规运动, 但不参与 `JFA` 支撑/崩坏判定。
- 一旦注入后, 它们在本帧内和普通格子一样参与热交换、`harm`、反应、材料转换和环境交互。
- 帧末 CPU 读回这些格子及其周边环境结果, 再把受损、转化、缺块等结果回写到外部实体状态。
- 若某些实体格子在本帧中被破坏、转换或清空, 这些变化会影响下一帧的实体形状注入, 从而允许实现断肢、缺块等效果。

也就是说, 实体受环境影响是真实存在的, 但这种影响通过 CPU/GPU 协同桥接完成, 而不是把角色直接塞进材料模拟器里。

## 第四章. 算法流程

### 4.1 本章目的

第三章定义了世界中“会发生什么”, 第四章则定义“这些事情按什么算法顺序发生”。本章关注的是 GPU 和 CPU 如何协同推进每个 tick, 各个求解器如何分工, 分页如何在不复制整窗的前提下滚动, 以及各个子系统的具体处理流程。

#### 4.1.1 二级 active 调度

为了避免每个 pass 都全图扫描, 运行时正式采用两级 active 调度:

- `ActiveChunkMask`
- `ActiveTileTTL`

其基本原则如下:

- 世界事件首先激活 chunk 级区域。
- 只有 active chunk 内部才继续细分 active tile。
- 各个局部 pass 优先处理 `ActiveTileTTL > 0` 的 tile, 而不是直接扫描整张世界图。
- pass 写回后负责标记自己和相邻区域的下一帧 active 状态。
- active 状态是带衰减的短时工作集, 不是永久标记。

这里的 `ActiveTileTTL` 是独立调度缓冲, 不属于 `CellCoreGrid`。它按 tile 记录活跃衰减计数, 而不是按 cell 记录。

当前建议的层级尺寸基准如下:

- `tile = 32 x 32` cells
- `chunk = 8 x 8 tiles = 256 x 256` cells

这样划分的原因是:

- `tile` 足够小, 适合 shared memory 局部求解和 seam correction。
- `chunk` 足够大, 适合作为切页、存取和 coarse active 的基本单位。

常见的 active 触发来源包括:

- 地形修改。
- 爆炸和法术。
- 粉体、液体、`FallingIsland` 发生位置变化。
- 光、热、气体在局部出现显著变化。
- 切页时新进入的 stripe。
- 实体 placeholder 占位写入。

因此, “活跃优化”不是维护全图活跃格子列表, 而是维护二级活跃区域, 再在 active tile 内部临时找真正活跃的行段或局部单元。

active 区域的退活规则正式定义如下:

- 某个 tile 在本帧没有发生格子改写、没有新的局部源项注入、没有来自相邻 tile 的边界传播需求时, 其 active 衰减计数减一。
- 某个 tile 只要再次发生材料、液体、气体、光学、placeholder 或温度变化, 就会把自己的 active 计数重置为活跃态。
- 当 tile 的 active 衰减计数降到 0 时, 它不再被视为 active tile。
- 当某个 chunk 内所有 tile 都已退活, 且该 chunk 不处于新切页进入、待保存、待加载或待修正状态时, 它从 `ActiveChunkMask` 中清除。

这里故意不把衰减帧数写死, 但正式要求它必须大于 1 帧, 以避免边界传播和 placeholder 修正在相邻帧之间抖动失活。

### 4.2 切页与循环世界窗口

世界运行时不直接维护一个会被整体复制的大地图缓冲, 而是维护一个环形的活动窗口。

#### 4.2.1 基本原则

- 活动窗口在物理显存上固定大小。
- 当前版本采用“缓存窗口大于实际活跃窗口”的循环窗口实现。
- 推荐把缓存窗口开到实际活跃范围的 2 倍, 以换取更简单的滚动与写入逻辑。
- 世界坐标通过逻辑起始位置映射到缓存窗口坐标。
- 相机或关注中心跨过阈值时, 只推进逻辑起始位置, 不搬动窗口内原有内容。

#### 4.2.2 滚动流程

1. 检测关注中心是否越过切页阈值。
2. 若越过, 推进逻辑起始位置。
3. 标记离开窗口的 stripe 为待保存。
4. 标记新进入窗口的 stripe 为待加载或待生成。
5. 把离开的 chunk 和 tile 从 active 集合中清出。
6. 把新 stripe 写入缓存窗口中当前 head 或 tail 对应区域。
7. 把新进入的 chunk 标记为 active, 并在其内部初始化 active tile。
8. 对应的气体、环境温度和其他场数据使用同样的循环映射逻辑。

这种方式的目标是让切页成本变成“增量进入和离开的条带更新”, 而不是“整窗复制”。

### 4.3 光线追踪流程

光学系统每个 tick 执行 typed ray traversal。

#### 4.3.1 输入

- 本帧扫描得到的 `OpticalEmitterBuffer`
- `CellCoreGrid`
- `MaterialOpticsTable`
- `LightTypeTable`

#### 4.3.2 主过程

1. 从每个 emitter 生成一组初始 ray。
2. 对每条 ray 执行 2D DDA 或等价网格穿行。
3. 逐格读取当前材料对该光类型的吸收、散射、折射参数。
4. 计算能量分配。
5. 写入 `VisibleIllumination` 与各种 `OpticalDoseBuffer`。
6. 若仍有剩余能量且未超过最大 bounce, 继续传播下一段。

#### 4.3.3 输出

- 可见照明结果。
- 材料光学剂量。
- 气体光学剂量。

### 4.4 DDA 运动流程

这里的 DDA 指可动物质在格子中的定向位移申请流程。当前它主要服务于粉体中的稀疏高速颗粒和其他需要轨迹扫描的物质, 而不是所有致密堆积都依赖它单独求解。

#### 4.4.1 输入

- `CellCoreGrid`
- 当前格子的 `velocity_xy`
- 重力、风和稀疏力源叠加后的目标速度

#### 4.4.2 主流程

1. 根据 `velocity_xy * dt` 计算目标位移。
2. 对目标位移施加材料级最大位移上限, 即使前方完全无阻挡, 单次更新也不能无限制前进。
3. 从起点朝“想移动到的位置”执行 DDA 扫描。
4. 找到第一个阻挡点之前的最远合法落点。
5. 若整条扫描路径无障碍, 也只移动到本次允许的目标终点, 不继续超程穿行。
6. 向 reservation buffer 提交位移意图。
7. 在 resolve 阶段统一执行成功的迁移并处理失败者的速度衰减、反弹或停滞。

#### 4.4.3 边界

- DDA 不是致密粉体整体沉降的唯一算法。
- DDA 更适合稀疏、高速、飞散和抛射态的材料运动。
- DDA 的核心语义是“先确定想去哪里, 再沿路径检查是否能到”, 而不是“只要没有阻挡就无限前进”。

### 4.5 JFA 崩塌处理流程

JFA 流程不进入每帧主循环, 而按结构破坏事件调度。

#### 4.5.1 触发

- 地形被挖断。
- 支撑材料消失。
- 高温、腐蚀或反应使结构资格改变。

#### 4.5.2 主流程

1. 生成当前待检测区域的 `StaticSolid` 占据 mask。
2. 生成支撑源 mask。
3. 对区域运行 JFA 风格的支撑传播。
4. 输出 `SupportedMask` 和 `UnsupportedMask`。
5. 将 `UnsupportedMask` 对应区域整体标记为 `FallingIsland`。
6. 后续在普通 tick 中推进 `FallingIsland` 的整体平移下落。
7. 若岛体局部失去足够多同类邻居, 则该部分转化为 `Powder`。

### 4.6 液体平铺流程

液体不依赖慢速邻格压力扩散来表现“找平”, 而采用独立的下填与平铺 pass。液体本身的有限流动不在这里单独实现, 而交给通用 DDA 运动阶段。

#### 4.6.1 输入

- 活动窗口中的液体格。
- 固体边界与阻挡信息。
- 局部速度或流动倾向。
- `ActiveChunkMask` 与 `ActiveTileTTL`。

#### 4.6.2 主流程

1. 先从 `ActiveChunkMask` 压缩出本帧需要处理的 active chunk。
2. 只在 active chunk 内继续解析 `ActiveTileTTL`, 得到本帧 `TTL > 0` 的 active tile 列表。
3. 每个 active tile 把本 tile 数据读入 shared memory。
4. 在 tile 内按行扫描连续液体段和其下一行的可达空位段。
5. 可达空位的判定要求从当前液体段到目标空位之间不存在不可穿透占位阻隔, 其中至少包括 `StaticSolid`、`FallingIsland` 和实体 placeholder。
6. 若某液体段满足下填条件, 则把该段的一部分液体直接搬运到下一行可达空位中。
7. 一次性回写 tile 内新的液体分布。
8. 标记本 tile 与相邻 tile 的边界变动, 交给后续 `seam correction pass`。
9. `seam correction pass` 只处理 tile 边界几行或几列, 解决因 tile 局部求解导致的假墙和边界断流问题。
10. 根据液体分布是否变化, 更新下一帧的 active chunk 与 active tile TTL。

这里的 `seam correction pass` 指的是专门修正 tile 边界连续性的轻量 pass。它不重跑整个液体 solver, 只负责让相邻 tile 之间本应连通的液体在边界上真正流通。

#### 4.6.3 液体与 `Powder` 的浮力处理

液体 pass 或紧邻它的耦合 pass 需要额外处理浸没 `Powder` 的浮沉:

1. 在 active tile 内识别被液体包围或部分浸没的 `Powder`。
2. 比较该 `Powder` 材料密度与液体密度。
3. 生成对应的垂向偏置:
   - 重于液体则向下沉。
   - 轻于液体则向上浮。
   - 接近中性则偏置减弱。
4. 把这个偏置交给粉体更新或局部交换逻辑使用。

#### 4.6.4 液体与实体 placeholder 的排液修正

实体通过每帧写入 placeholder 物质来获得真实占位。液体系统需要在主液体重排后增加一个轻量排液修正阶段:

1. 收集本帧新写入或发生位移的 placeholder 区域。
2. 只在这些区域及其相邻 tile 上运行排液修正。
3. 把 placeholder 体积内原本被液体占据的部分批量搬运到角色两侧、且与角色等宽的可达范围内。
4. 由于承接区域面积通常扩大到原占位宽度的约 2 倍, 因此回填量按相对系数重新分配到左右两侧。
5. 对被排开的液体施加一个远离角色、同时向上向外的初始速度。
6. 若局部没有足够空间, 则把多余液体保留为下一帧继续修正的 active 区域。

这个 pass 的目标不是全图重算液体, 而是保证实体进入水体时能产生真实的占液和排液效果。
这里再明确 placeholder 的边界:

- placeholder 对液体和常规运动表现为真实阻挡占位。
- placeholder 不参与 `JFA` 支撑传播, 不作为结构支撑源。
- placeholder 在本帧内参与热交换、反应、`harm` 和材料转换。
- placeholder 的归属由并行的 `entity_id` 状态指向对应实体。
- placeholder 的最终受损、转换和缺块结果在帧末读回并回写到外部实体状态, 再由下一帧重新注入。

### 4.7 气体系统与 Navier-Stokes Jacobi 求解

气体系统运行在低分辨率共享流场上, 采用简化 Navier-Stokes 路线。

#### 4.7.1 主字段

- `flow_velocity_xy`
- `ambient_temperature`
- `gas_concentration[species_id]`
- `pressure_ping/pong`
- `divergence`

#### 4.7.2 主流程

1. 注入本帧外部冲量和边界源项。
2. 对 `flow_velocity_xy` 做 advection。
3. 计算散度 `divergence`。
4. 用 Jacobi 迭代求解 pressure。
5. 用 pressure projection 修正速度场。
6. 用共享流场 advect 各个气体 species 浓度。
7. 处理扩散、衰减、温度耦合和气体反应。

### 4.8 GPU-CPU 协同架构

GPU 对材料世界、气体场、光学传播和主要环境状态保持权威。CPU 对角色、敌人、AI、战斗数值和高层玩法保持权威。

#### 4.8.1 设计目标

GPU-CPU 协同的核心目标不是“让 CPU 随时拿到最新全图”, 而是:

- 保持 GPU 对世界模拟的持续推进权。
- 让 CPU 以真异步方式拿到实体周边足够小、足够新的局部环境快照。
- 避免读回操作把 GPU 主队列同步卡死。

因此, 本架构默认采用基于 PBO 双缓冲或三缓冲的异步读回。fence 不作为基础方案, 只作为未来在高负载下验证或兜底的升级选项。

#### 4.8.2 CPU 到 GPU

CPU 向 GPU 上传的主要内容包括:

- 实体位置、包围盒和运动状态。
- 稀疏力源和高层世界命令。
- 必要的生成、挖掘、破坏、施法指令。

其中实体上传还应包括 placeholder 写入信息, 使 GPU 能在本帧材料与液体流程中正确占位和排液。

#### 4.8.3 GPU 到 CPU

GPU 向 CPU 提供的主要内容包括:

- 角色周围局部格子原始块读回。
- 局部环境温度、气体浓度或必要环境字段读回。
- 必要的调试和观测缓冲。

这些读回不走同步 `glReadPixels` 式阻塞路径, 而走 PBO 异步路径。

#### 4.8.4 PBO 真异步读回

当前版本中, 每个实体或每组实体共享一组循环使用的 PBO 读回槽位。基础形式优先采用双缓冲:

- `ReadbackRequest`
- `ReadbackPBO[ping]`
- `ReadbackPBO[pong]`
- `ReadbackResult`

基本原则如下:

- GPU 在第 `N` 帧末尾把局部区域拷贝到当前写入 PBO。
- CPU 在第 `N+1` 帧只尝试读取上一帧写入完成的另一个 PBO。
- CPU 永远不读取当前帧刚发起传输的那个槽位。
- GPU 和 CPU 通过 `ping/pong` 槽位交替, 避免同一缓冲在同一帧被同时写和读。

这就是当前版本的真异步基线。当前版本把“`N+1` 帧读回不阻塞”视为必须达成的性能目标, 而不是可选优化。只要 CPU 从不去 map 当前帧刚发出的读回槽位, 并且读回区域保持足够小, 这条路径就应稳定工作在一帧延迟模式下。

如果后续实测发现 GPU 队列积压, 导致上一帧 PBO 在下一帧仍偶尔未完成, 这优先被视为 GPU 负载或帧率预算超标问题, 应先优化 pass 或下调频率目标。`fence` 只作为诊断和兜底手段, 不作为基础依赖。

#### 4.8.5 一帧时序

标准一帧协同时序如下:

1. CPU 消费上一帧已完成的 `ReadbackResult`。
2. CPU 用这些结果推进实体碰撞、AI、技能和状态逻辑。
3. CPU 上传实体位置、placeholder 信息和高层命令。
4. GPU 执行本帧世界模拟 pass。
5. GPU 在帧尾对每个需要观测的实体区域发起局部读回到当前写入 PBO。
6. 下一帧 CPU 再读取上一帧对应的 PBO。

默认目标是一帧延迟。如果 GPU 命令队列没有异常积压, 小区域读回应当能够稳定满足这一延迟。

#### 4.8.6 读回内容与区域

每个实体每帧只请求自己周边的小区域, 不请求整图。局部读回至少可以包含:

- `material_id`
- `phase`
- `cell_temperature`
- 必要时的 `cell_flags`
- 必要时的 `timer_pack`
- 必要时的 `integrity`
- 局部 `ambient_temperature`
- 局部气体浓度
- 局部光学 dose 或其他必要光学量
- 必要时的液体占据与 `FallingIsland` 占据信息

这样 CPU 能做:

- 地形碰撞判断
- 液体接触与浮沉判断
- 毒气、高温、圣光等环境状态判断
- 局部技能目标判定

#### 4.8.7 真异步的约束

要保证这套 PBO 读回在基础版本里尽量保持真异步, 必须满足以下约束:

- 不在发起读回的同一帧立即 map 该 PBO。
- 不复用当前帧仍在写入的同一个 PBO 槽位。
- 读回区域保持小而稳定, 避免大量零碎同步请求。
- 读回命令放在世界模拟之后、但不与 CPU 当帧逻辑形成强同步链。
- 如果后续发现上一帧 PBO 在下一帧偶发未完成, 再引入 fence 做显式跳过或降级处理。

#### 4.8.8 异步原则

- CPU 不直接改写原始格子块。
- CPU 每帧只读回每个实体周边的小区域环境块。
- 读回默认接受一帧延迟。
- CPU 使用异步读回得到的局部快照做碰撞、AI 和技能逻辑。
- GPU 独立推进世界模拟。
- 双方通过命令上传和局部观测而不是共享可写世界状态协同。

#### 4.8.9 完整帧执行顺序

当前版本的一帧正式顺序如下。

1. CPU 读取上一帧对应的 PBO, 取得各实体周边的局部环境快照。
2. CPU 基于这些一帧延迟的快照推进实体碰撞、AI、技能、状态和相机关注中心。
3. CPU 生成本帧的高层世界命令, 包括挖掘、破坏、施法、生成和稀疏力源更新。
4. CPU 根据实体最新位置生成 placeholder 占位写入描述, 并确定下一次局部读回的中心区域。
5. 如果关注中心触发切页, CPU 同时更新逻辑窗口起始位置, 并准备本帧需要进入和离开的 stripe 元数据。
6. CPU 向 GPU 上传实体状态、placeholder、世界命令、稀疏力源和已准备好的 stripe 更新。
7. GPU 先应用切页相关的 stripe 写入和初始化, 再清理上一帧 placeholder 残留并写入当前帧 placeholder。
8. GPU 根据 stripe、placeholder、世界命令和局部变化种子更新 `ActiveChunkMask` 与 `ActiveTileTTL`。
9. GPU 执行气体与环境温度 pass, 更新共享流场、环境温度和各气体 species。
10. GPU 执行格子热交换与通用相变 pass, 更新 `cell_temperature` 以及熔化、凝固、沸腾、冷凝等基础相变结果。
11. GPU 执行材料本地 reaction trigger pass, 基于 `timer_pack`、`integrity` 和局部条件触发各材料自己的 `reaction_slot0..7`。
12. GPU 执行 `FallingIsland` 整体平移更新和其他需要 DDA 的材料位移申请与 resolve。
13. GPU 执行液体主 pass, 包括行段下填、有限 DDA 随机流动、`Powder` 浮沉耦合和 seam correction。
14. GPU 执行 placeholder 排液修正 pass, 只处理本帧实体新占位附近的活跃区域。
15. GPU 执行光学发射与光线传播 pass, 写入可见照明和各类光学 dose。
16. GPU 执行材材、材气、材光、气气、气光等反应 resolve, 消费 `ReactionBuffer` 和各类 dose, 统一落地 `convert_material`、`harm`、`modify_temperature`、`modify_gas` 等结果。
17. GPU 执行渲染相关输出。
18. GPU 在帧尾对每个需要观测的实体区域发起局部读回到当前写入 PBO。
19. 下一帧 CPU 读取这些 PBO, 周期重复。

这条时序强调 3 件事:

- CPU 始终消费旧快照, 不等待 GPU 当帧完成。
- GPU 先完成世界更新, 再发起局部读回。
- 切页、placeholder、active 标记都被视为“本帧正式输入”, 而不是旁路逻辑。

#### 4.8.10 切页插入时序

切页不是独立的大模式切换, 而是插入到普通帧流程中的增量步骤。其标准流程如下:

1. CPU 在实体与相机更新后, 判断关注中心是否越过窗口阈值。
2. 若越过, 只更新逻辑窗口起始位置, 不搬移缓存窗口内部现有内容。
3. CPU 计算哪些 stripe 将离开 2 倍缓存窗口的有效覆盖区, 并把这些 stripe 加入待保存队列。
4. CPU 计算哪些新 stripe 将进入有效覆盖区, 并把它们加入待加载或待生成队列。
5. 已经准备好的进入 stripe 在本帧上传给 GPU, 其物理写入位置由逻辑起始位置和取模映射共同决定。
6. GPU 在本帧最早阶段把这些新 stripe 写入缓存窗口的 head 或 tail 区域。
7. 新进入 stripe 所在 chunk 被立即标记为 active, 其相邻 tile 也获得初始 active TTL, 以保证地形、液体、气体和光学边界能尽快收敛。
8. 离开窗口的 stripe 在保存完成后即可从 active 集合和相关暂存结构中清出。

这套流程的目标是:

- 世界滚动只引入增量 stripe 更新。
- 不发生整窗复制。
- 新进入区域天然成为下一帧局部求解的热点区域。

#### 4.8.11 各阶段主要输入输出

为了让实现阶段更容易拆分 pass, 当前一帧内各阶段的主要 I/O 约定如下:

- CPU 逻辑阶段:
  - 输入: 上一帧 `ReadbackResult`
  - 输出: 实体状态上传、placeholder 上传、世界命令、读回请求、切页元数据

- GPU 切页与占位阶段:
  - 输入: stripe 更新、placeholder、逻辑窗口起始位置
  - 输出: 更新后的 `CellCoreGrid`, `ActiveChunkMask`, `ActiveTileTTL`

- GPU 气体阶段:
  - 输入: `GasSystem`, 环境源项, active 边界种子
  - 输出: 新的 `flow_velocity_xy`, `ambient_temperature`, `gas_concentration`

- GPU 热与相变阶段:
  - 输入: `CellCoreGrid`, `cell_temperature`, `ambient_temperature`, `MaterialTable`, `GasSpeciesTable`
  - 输出: 温度更新、基础相变结果、气液源汇更新

- GPU 本地 reaction trigger 阶段:
  - 输入: `CellCoreGrid`, `timer_pack`, `integrity`, `MaterialTable`, `ReactionBuffer`
  - 输出: 被触发的材料 `reaction_slot0..7` 请求

- GPU 运动阶段:
  - 输入: `CellCoreGrid`, `velocity_xy`, 稀疏力源, 气体风场, 液体流向意图
  - 输出: DDA reservation/resolve 结果、`FallingIsland` 新位置、更新后的活跃区域

- GPU 液体阶段:
  - 输入: 液体格、固体边界、placeholder 区域、`ActiveChunkMask`, `ActiveTileTTL`
  - 输出: 新液体分布、`Powder` 浮沉偏置、seam correction 结果、液体流向意图、下一帧液体 active 标记

- GPU 光学阶段:
  - 输入: 本帧 `emit_light` 请求、`MaterialOpticsTable`, `LightTypeTable`
  - 输出: `VisibleIllumination`, `CellOpticalDose`, `GasOpticalDose`

- GPU 反应 resolve 阶段:
  - 输入: 各类 dose、反应表、`ReactionBuffer`
  - 输出: `convert_material`, `harm`, `modify_temperature`, `modify_gas` 等正式状态改动

- GPU 读回阶段:
  - 输入: 本帧最终世界状态、实体读回请求
  - 输出: 写入 PBO 的局部环境块
