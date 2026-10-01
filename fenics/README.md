# FEniCS 三维连续体示例

本目录的脚本与仓库主体是**两套独立实现**，不共享代码：

| | 实现 | 依赖 |
|---|---|---|
| 仓库主体 `src/fem_beam.py` | 一维梁理论 | NumPy、Matplotlib |
| 本目录 | 三维连续体有限元 | FEniCS / dolfin、mshr、ufl |

因此单独放置，避免与 `examples/` 下基于 `fem_beam` 的脚本混淆。
运行本目录脚本需另行安装 FEniCS。

| 文件 | 内容 |
|---|---|
| `static_cantilever_deflection.py` | 三维悬臂梁在分布面力下的静态挠度 |
| `hollow_box_beam_static.py` | 空心箱梁在自重与面载下的静态变形 |
| `transient_beam_vibration.py` | 静态预载后的悬臂梁瞬态振动 |
| `PROJECT_SUMMARY.md` | 原项目说明 |
