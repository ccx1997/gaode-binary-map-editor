# dushiyuan-1

这是 `dushiyuan-1` 小区当前人工确认、可继续编辑的高德截图道路地图。当前正式版来自 2026-09-29 的本地导出，保留完整原图尺寸 `2192 × 1642`，未裁剪，坐标偏移为 `(0, 0)`。

地图使用黑色楼栋轮廓（0）、灰色道路（192）和白色背景（255），道路在初始宽度基础上加粗 2 px。JSON 包含 `b0`–`b10` 共 11 个楼栋入口标签，保存在 `project.annotations.specialPoints` 中；点和标签可在网页中查看、修改，PNG 保持灰度地图内容。

## 文件

- [`raw_inputs/source.jpeg`](raw_inputs/source.jpeg)：原始高德地图截图。
- [`maps/final/map.png`](maps/final/map.png)：最新灰度道路地图，可直接在 GitHub 查看。
- [`maps/final/annotations.json`](maps/final/annotations.json)：可重新导入编辑器的完整标注工程。
- [`maps/final/meta.json`](maps/final/meta.json)：版本角色和尺寸说明。

`final/` 表示当前人工确认版本，不是不可修改版本。以后完成修改后，使用新的 `map.png` 和 `annotations.json` 覆盖同名文件，再人工执行 Git 提交和推送；GitHub 不会自动覆盖本地编辑工程。

## 在 macOS 上继续编辑

在终端中进入仓库根目录：

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python map_converter.py \
  --input "communities/dushiyuan-1/raw_inputs/source.jpeg"
```

网页自动打开后：

1. 点击“导入标注 JSON”。
2. 选择 `communities/dushiyuan-1/maps/final/annotations.json`。
3. 网页恢复当前道路、矩形轮廓、道路删除范围、复制楼栋、道路样式和 11 个楼栋入口标签。
4. 点击“验证全局路线”，依次选择起点和终点；出现“规划成功：两点之间道路连通”即表示该段道路验证通过。
5. 修改完成后点击“导出 PNG + JSON”。

第一次导入并导出后，本机会保存编辑工程；以后从同一仓库、使用同一张原图启动时，会自动恢复上一次导出的状态。

## 边界

这里的 `map.png` 是像素级灰度道路图，`annotations.json` 是本编辑器工程文件。当前没有 ROS/Nav2 所需的 `resolution` 和 `origin`，不能只凭这两个文件直接部署机器人导航。
