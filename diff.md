# 计划与实现的差异清单

对照 `plan/engine_plan.md`、`plan/game_plan.md` 中物质与反应部分的当前实现差异。已确认一致的部分不列入;帧流程执行顺序、光反应 dose 滞后一帧的差异按会议要求不列入。

## 物质系统

### 1. 相态完整性原则未达成(game_plan §11.6)

- 计划: 每种物质族默认尽量具备完整 `liquid / gas / powder / solid` 变种; 例外时"把相变温度设到极大值"使其正常玩法中不进入对应相态。
- 现状: 只有 water / poison / oil 三族齐三相+气体; 多数物质单相(sand/gravel/soil 仅 powder, 三种石头仅 solid, root/log/vine 仅 solid, leaf/moss 仅 powder, explosive/vortex_heart/placeholder 仅 solid, phosphor×4 仅 powder)。不支持的相态直接留 `melt_point=None`(打包为 NaN, `oracle_game/gpu/packers.py:537-538`), 未按例外条款设极大值。

### 2. 六种物质的 melt_point 是死数据

- sandstone / raw_stone / obsidian / root / log / vine 填了 melt_point(260–2200°C)但 `melt_to_material=None`(`oracle_game/rules_materials.py:108,132,156` 等)。
- 热求解器要求 `melt_to_material_id > 0` 才执行相变(`oracle_game/sim/heat.py:298-303`), 因此这些熔点永远不生效, 属于"数据已填但语义悬空"。

### 3. phosphor_* 四物质参数雷同

- 四个 phosphor 粉末除颜色和发光类型外参数完全相同(density=0.62、integrity=32 等, `oracle_game/rules_materials.py:794-885`), 是全表最接近"占位"的一组数据。

## 反应系统框架

### 4. 完整度归零路径与计划不符(engine_plan §2.9)

- 计划: `integrity <= 0` 时由材料自己的 reaction slot 处理破坏后果(碎裂/销毁/转化), 不走独立机制。
- 现状: harm / consume 使 integrity ≤ 0 时**直接清格**(`oracle_game/sim/reactions_actions.py:312-313`; GPU 侧 `oracle_game/shaders/reactions/_common.comp:302-308`)。自反应规则虽有 `integrity_at_most / integrity_at_least` 条件字段(`oracle_game/types.py:201-202`), 但默认内容未使用这条路径。

### 5. trigger_kind 无显式字段, "重叠接触"未实现(engine_plan §2.7.1)

- 计划: 配对反应表条目含 `trigger_kind`(邻接接触/重叠接触/光照 dose 超阈值/气体浓度超阈值/温度区间/定时触发/完整度归零)。
- 现状: 无 `trigger_kind` 字段(`oracle_game/types.py:175-190`), 触发类型由表类别 + threshold 隐式决定; 材材反应仅 4 邻接(`oracle_game/sim/reactions_selectors.py:49`), "重叠接触"未实现; "完整度归零"作为触发类型也因上一条的原因未被默认内容使用。

### 6. 自反应匹配源缺两项(engine_plan §2.7.2)

- 计划: 自反应匹配条件来自 material_id / phase / cell_temperature / timer_pack / base_integrity / integrity。
- 现状: `SelfReactionRule` 只支持 material_id / phase / 温度 / integrity(`oracle_game/types.py:193-202`); timer_pack 与 base_integrity 不能作为匹配条件(`timer_index` 仅是校验提示, 要求与 slot 一致, `oracle_game/world_table_validation.py:448-453`)。

### 7. GasGasReactionTable 空转

- 框架、GPU pass、上传链路齐全(`oracle_game/rules.py:47`), 但默认规则为 0 条(`oracle_game/rules_reactions.py:11` `GG=[]` 从未 append)。

### 8. LightLightReactionTable 无预留结构

- 计划允许其"预留为可选表, 默认关闭"; 代码中连预留结构都不存在(全库无 `light_light` 匹配)。影响小, 仅记录。

### 9. convert_material.generation 字段未使用

- 字段在 schema 中存在(`oracle_game/types.py:88`), 但 28 个默认动作全部为 0, 无任何使用。

## 反应内容(game_plan §13.2)

### 10. oil 自燃只配了 liquid 形态

- 计划: `oil` 温度达阈值会自己燃烧。
- 现状: 只有 oil_liquid 有自燃自反应(`oracle_game/rules_reactions.py:298`, ≥120°C); oil_solid / oil_powder 无自燃 self rule, 只能被引燃。

### 11. vine 超配(方向相反的偏差)

- 计划(§13.2.2): `vine` 的反应暂时留空。
- 现状: vine_solid 实际配了酸蚀/毒/污染/可燃/光温共 7 个槽(`oracle_game/rules.py:437, 511-513, 516-518`)。

### 12. chaos_light "轻微升温"数值未体现轻微

- 计划(§13.2.8): chaos_light 轻微升温; visible_light 升温乘 dose。
- 现状: 两者升温动作同为 MODIFY_TEMPERATURE +3.0(action 19, `oracle_game/rules.py:394`), chaos 与 visible 共用 slot 6, 数据上未区分强弱。

## 光与气体

### 13. MaterialOpticsTable 只有一维差异化数据(engine_plan §2.6.2)

- 二维表结构存在(材料×光, 含 absorption/scattering/refraction), 但默认值只按 render_group 分 7 档 + 3 个特例(`oracle_game/rules.py:523-542`); 除 `gold_solid×visible_light` 外不随 light_type 变化, 即"[material][light] 双维度"数据基本未填。

### 14. 气体的两个 reaction tag mask 字段空转

- `material_reaction_tag_mask` / `light_reaction_tag_mask` 在 schema、GPU 打包、反应选择器全链路存在(`oracle_game/gpu/packers.py:644-645`, `oracle_game/sim/reactions_selectors.py:325-333`), 但 `_build_gases()` 不传值, 默认数据全 0; 当前规则均按名字引用气体。

### 15. 计划外气体物种 fire_gas

- game_plan §14.1.1 首版 species 只含 空气 + water/poison/oil/pollution 相变气体; 代码额外定义了 `fire_gas`(`oracle_game/rules.py:280` 附近), 被燃烧规则使用(如 `oracle_game/rules_reactions.py:42-60`)。属计划外扩展, 需要回写计划或确认保留。

### 16. magic_light 无任何内置行为

- 计划(§12): magic_light 用来挂载玩家自定义内容。机制(独立 dose 通道 3、render_style "magic"、运行时规则 API `oracle_game/world_table_api.py:152/198/451`)存在, 但全仓无任何引用 magic_light 的反应规则, 无默认/示例内容。与定位不冲突, 仅记录空白状态。
