#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert a Gaode screenshot into an editable 0/127/255 grayscale map.

The image processing and editor both run locally. The web UI is embedded in this
single Python file and is served only on 127.0.0.1.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import threading
import webbrowser
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
from PIL import Image, ImageOps


@dataclass
class ConversionReport:
    source_name: str
    width: int
    height: int
    boundary_width_px: int
    green_tolerance: int
    green_fraction: float
    filled_small_holes: int
    contour_count: int
    input_was_binary: bool
    road_width_px: int = 9
    road_component_count: int = 0
    road_centerline_pixels: int = 0
    input_was_grayscale_map: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="高德截图转灰色道路中心线、黑边楼栋地图，并启动本地编辑网页。"
    )
    parser.add_argument("--input", required=True, help="输入 JPEG/PNG 图片路径")
    parser.add_argument(
        "--output-dir", default="outputs", help="初始灰度底图和报告的输出目录"
    )
    parser.add_argument(
        "--boundary-width", type=int, default=3, help="自动轮廓线宽，单位 px"
    )
    parser.add_argument(
        "--green-tolerance", type=int, default=34, help="绿色识别 Lab 色差容差"
    )
    parser.add_argument(
        "--road-width", type=int, default=9, help="道路中心线加粗宽度，单位 px（灰度 127）"
    )
    parser.add_argument("--port", type=int, default=8765, help="本地网页端口")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    parser.add_argument("--convert-only", action="store_true", help="只转换，不启动网页")
    return parser.parse_args()


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        return np.asarray(image)


def png_bytes(rgb: np.ndarray) -> bytes:
    buffer = BytesIO()
    Image.fromarray(rgb.astype(np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def looks_binary(rgb: np.ndarray) -> bool:
    channel_spread = rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2).astype(np.int16)
    gray = rgb.mean(axis=2)
    near_bw = (gray <= 20) | (gray >= 235)
    return bool(np.mean(channel_spread <= 8) > 0.985 and np.mean(near_bw) > 0.97)


def looks_grayscale_map(rgb: np.ndarray) -> bool:
    """Accept exported maps without mistaking ordinary grayscale photos for maps."""
    spread = rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2).astype(np.int16)
    gray = rgb.mean(axis=2)
    near_palette = (gray <= 20) | (np.abs(gray - 127) <= 12) | (gray >= 235)
    return bool(np.mean(spread <= 8) > 0.985 and np.mean(near_palette) > 0.97)


def thin_centerline(mask: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning keeps road junctions and loops connected."""
    pixels = np.pad((mask > 0).astype(np.uint8), 1)
    while True:
        changed = False
        for phase in (0, 1):
            p = pixels[1:-1, 1:-1]
            neighbors = [pixels[:-2, 1:-1], pixels[:-2, 2:], pixels[1:-1, 2:],
                         pixels[2:, 2:], pixels[2:, 1:-1], pixels[2:, :-2],
                         pixels[1:-1, :-2], pixels[:-2, :-2]]
            n, ne, e, se, s, sw, w, nw = neighbors
            count = sum(neighbors)
            transitions = sum(((a == 0) & (b == 1)).astype(np.uint8)
                              for a, b in zip(neighbors, neighbors[1:] + neighbors[:1]))
            if phase == 0:
                corners = (n * e * s == 0) & (e * s * w == 0)
            else:
                corners = (n * e * w == 0) & (n * s * w == 0)
            remove = (p == 1) & (count >= 2) & (count <= 6) & (transitions == 1) & corners
            if np.any(remove):
                p[remove] = 0
                changed = True
        if not changed:
            return pixels[1:-1, 1:-1] * 255


def split_map_regions(green_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Separate thin road networks from compact buildings in this map palette.

    Roads and buildings share a similar color. Area relative to the largest
    inscribed radius distinguishes a long road/network from a compact block.
    """
    foreground = cv2.bitwise_not(green_mask)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, connectivity=8)
    roads, buildings = np.zeros_like(green_mask), np.zeros_like(green_mask)
    road_count = 0
    for label in range(1, count):
        x, y, w, h, area = map(int, stats[label])
        if area < max(80, foreground.size * 0.00004):
            continue
        component = (labels[y:y+h, x:x+w] == label).astype(np.uint8)
        distance = cv2.distanceTransform(np.pad(component, 1), cv2.DIST_L2, 5)
        radius = max(1.0, float(distance.max()))
        is_road = area / (radius * radius) >= 30
        target = roads if is_road else buildings
        target[y:y+h, x:x+w][component > 0] = 255
        road_count += int(is_road)
    return roads, buildings, road_count


def remove_small_mask_components(mask: np.ndarray, minimum_area: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= minimum_area:
            cleaned[labels == label] = 255
    return cleaned


def fill_small_holes(mask: np.ndarray, max_area: int) -> tuple[np.ndarray, int]:
    inverse = cv2.bitwise_not(mask)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(inverse, connectivity=8)
    height, width = mask.shape
    result = mask.copy()
    filled = 0
    for label in range(1, count):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        touches_edge = x == 0 or y == 0 or x + w >= width or y + h >= height
        compact_enough = w < width * 0.10 and h < height * 0.10
        if not touches_edge and compact_enough and area <= max_area:
            result[labels == label] = 255
            filled += 1
    return result, filled


def remove_colored_map_annotations(rgb: np.ndarray) -> np.ndarray:
    """Inpaint saturated non-green labels/icons before semantic segmentation."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    non_green_hue = (hue < 20) | (hue > 105)
    colored = ((saturation >= 52) & (value >= 55) & non_green_hue).astype(np.uint8) * 255

    count, labels, stats, _ = cv2.connectedComponentsWithStats(colored, connectivity=8)
    annotation_mask = np.zeros_like(colored)
    image_area = colored.size
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        if 2 <= area <= image_area * 0.012 and width < rgb.shape[1] * 0.30 and height < rgb.shape[0] * 0.15:
            annotation_mask[labels == label] = 255

    annotation_mask = cv2.dilate(
        annotation_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    )
    if not np.any(annotation_mask):
        return rgb
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    repaired = cv2.inpaint(bgr, annotation_mask, 5, cv2.INPAINT_TELEA)
    return cv2.cvtColor(repaired, cv2.COLOR_BGR2RGB)


def gaode_green_mask(rgb: np.ndarray, tolerance: int) -> tuple[np.ndarray, int]:
    repaired = remove_colored_map_annotations(rgb)
    blurred = cv2.GaussianBlur(repaired, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_RGB2HSV)
    candidate = (
        (hsv[:, :, 0] >= 25)
        & (hsv[:, :, 0] <= 100)
        & (hsv[:, :, 1] >= 28)
        & (hsv[:, :, 2] >= 80)
    )
    if float(np.mean(candidate)) < 0.05:
        raise ValueError(
            "没有识别到足够的浅绿色地图区域；请确认输入为配色相近的高德小区地图截图。"
        )

    lab = cv2.cvtColor(blurred, cv2.COLOR_RGB2LAB).astype(np.float32)
    green_reference = np.median(lab[candidate], axis=0)
    distance = np.linalg.norm(lab - green_reference, axis=2)
    hue_ok = (hsv[:, :, 0] >= 22) & (hsv[:, :, 0] <= 105)
    saturation_ok = hsv[:, :, 1] >= 18
    mask = ((distance <= tolerance) & hue_ok & saturation_ok).astype(np.uint8) * 255

    short_side = min(mask.shape)
    close_size = max(5, int(round(short_side * 0.008)))
    if close_size % 2 == 0:
        close_size += 1
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size)),
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    minimum_green_area = max(80, int(mask.size * 0.00004))
    mask = remove_small_mask_components(mask, minimum_green_area)
    # Ignore thin screenshot gutters running along an entire edge. Otherwise
    # the non-green gutter is interpreted as a road joined to the entrance.
    for view in (mask, mask[:, ::-1], mask.T, mask.T[:, ::-1]):
        green_fraction = np.mean(view > 0, axis=0)
        candidates = np.flatnonzero(green_fraction >= 0.05)
        if candidates.size and 0 < candidates[0] <= view.shape[1] * 0.02:
            view[:, :candidates[0]] = 255
    maximum_hole_area = max(350, int(mask.size * 0.0009))
    mask, filled = fill_small_holes(mask, maximum_hole_area)
    mask = cv2.medianBlur(mask, 7)
    return mask, filled


def convert_map(
    rgb: np.ndarray, source_name: str, boundary_width: int, tolerance: int,
    road_width: int = 9,
) -> tuple[np.ndarray, ConversionReport]:
    height, width = rgb.shape[:2]
    boundary_width = max(1, int(boundary_width))
    tolerance = max(5, min(100, int(tolerance)))
    road_width = max(1, min(100, int(road_width)))

    if looks_grayscale_map(rgb):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gray = np.where(gray < 64, 0, np.where(gray < 192, 127, 255)).astype(np.uint8)
        output = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        report = ConversionReport(
            source_name=source_name,
            width=width,
            height=height,
            boundary_width_px=boundary_width,
            green_tolerance=tolerance,
            green_fraction=0.0,
            filled_small_holes=0,
            contour_count=0,
            input_was_binary=looks_binary(rgb) and not bool(np.any(gray == 127)),
            input_was_grayscale_map=True,
            road_width_px=road_width,
        )
        return output, report

    green_mask, filled_count = gaode_green_mask(rgb, tolerance)
    roads, buildings, road_count = split_map_regions(green_mask)
    centerline = thin_centerline(roads)
    contours, _ = cv2.findContours(
        buildings, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    min_perimeter = max(18.0, min(width, height) * 0.009)
    useful_contours: list[np.ndarray] = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        if perimeter < min_perimeter:
            continue
        epsilon = max(1.25, perimeter * 0.00085)
        useful_contours.append(cv2.approxPolyDP(contour, epsilon, True))

    binary_gray = np.full((height, width), 255, dtype=np.uint8)
    thick_roads = cv2.dilate(centerline, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (road_width, road_width)))
    binary_gray[thick_roads > 0] = 127
    cv2.drawContours(
        binary_gray, useful_contours, -1, color=0, thickness=boundary_width, lineType=cv2.LINE_8
    )
    edge_clear = boundary_width + 1
    binary_gray[:edge_clear, :] = 255
    binary_gray[-edge_clear:, :] = 255
    binary_gray[:, :edge_clear] = 255
    binary_gray[:, -edge_clear:] = 255
    output = cv2.cvtColor(binary_gray, cv2.COLOR_GRAY2RGB)
    report = ConversionReport(
        source_name=source_name,
        width=width,
        height=height,
        boundary_width_px=boundary_width,
        green_tolerance=tolerance,
        green_fraction=round(float(np.mean(green_mask > 0)), 6),
        filled_small_holes=filled_count,
        contour_count=len(useful_contours),
        input_was_binary=False,
        road_width_px=road_width,
        road_component_count=road_count,
        road_centerline_pixels=int(np.count_nonzero(centerline)),
    )
    return output, report


def estimate_outline_width(rgb: np.ndarray, fallback: int = 5) -> int:
    """Measure common short black spans, including imported map outlines.

    OpenCV thickness=3 rasterizes straight building edges as 5 pixels. Reading
    the rendered map avoids using that nominal thickness for canvas rectangles.
    """
    black = rgb[:, :, 0] < 64
    counts = np.zeros(101, dtype=np.int64)
    for plane in (black, black.T):
        for row in plane:
            changes = np.diff(np.r_[False, row, False].astype(np.int8))
            lengths = np.flatnonzero(changes == -1) - np.flatnonzero(changes == 1)
            lengths = lengths[(lengths > 0) & (lengths <= 100)]
            counts += np.bincount(lengths, minlength=101)
    return int(np.argmax(counts)) if counts.any() else max(1, min(100, fallback))


def write_initial_outputs(
    output_dir: Path, stem: str, binary_rgb: np.ndarray, report: ConversionReport
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    base_path = output_dir / f"{stem}_auto_base.png"
    report_path = output_dir / f"{stem}_conversion_report.json"
    Image.fromarray(binary_rgb).save(base_path, format="PNG")
    report_path.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return base_path, report_path


HTML = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>高德地图道路与楼栋编辑器</title>
  <link rel="icon" type="image/svg+xml" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='7' fill='%23141b2d'/%3E%3Cpath d='M6 23V9h20v14H6zm3-3h14v-8H9v8z' fill='white'/%3E%3Cpath d='M12 12v8M20 12v8' stroke='%2300c2a8' stroke-width='2'/%3E%3C/svg%3E">
  <style>
    :root{color-scheme:dark;--bg:#0d1220;--panel:#141b2d;--panel2:#1b2439;--line:#2c3854;--text:#f3f6fb;--muted:#a7b2c8;--accent:#00c2a8;--danger:#ff5e6c;--focus:#78a9ff}
    *{box-sizing:border-box}[hidden]{display:none!important}html,body{height:100%;margin:0;overflow:hidden;background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif;font-size:16px}
    button,input,select{font:inherit}button{min-height:38px;border:1px solid var(--line);border-radius:8px;background:#202b43;color:var(--text);padding:7px 10px;cursor:pointer}button:hover{border-color:#53627f}button.active,button.primary{background:var(--accent);border-color:var(--accent);color:#071411;font-weight:700}button.danger{color:#ffd7dc;border-color:#713845}.app{display:grid;grid-template-columns:310px minmax(0,1fr) 300px;height:100%}.sidebar,.layers{background:var(--panel);overflow:auto;padding:16px;border-right:1px solid var(--line)}.layers{border-right:0;border-left:1px solid var(--line)}h1{font-size:20px;margin:0 0 4px}.subtitle{color:var(--muted);font-size:13px;line-height:1.5;margin-bottom:14px}.section{border-top:1px solid var(--line);padding-top:13px;margin-top:13px}.section-title{font-size:13px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-bottom:9px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:7px}.stack{display:grid;gap:7px}.row{display:flex;gap:8px;align-items:center}.row>*{min-width:0}.grow{flex:1}label{display:block;color:var(--muted);font-size:13px;margin:8px 0 5px}input[type=number],input[type=text]{width:100%;height:38px;border:1px solid var(--line);border-radius:8px;background:#0f1627;color:var(--text);padding:7px 9px}.check{display:flex;align-items:center;gap:7px;color:var(--text);margin:8px 0}.check input{width:17px;height:17px}.workspace{min-width:0;display:grid;grid-template-rows:48px minmax(0,1fr);background:#090d16}.topbar{display:flex;align-items:center;justify-content:space-between;padding:0 14px;border-bottom:1px solid var(--line);background:#101729}.status{color:var(--muted);font-size:13px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.stage{position:relative;min-height:0;overflow:hidden;touch-action:none;cursor:crosshair}.stage.pan{cursor:grab}.stage.panning{cursor:grabbing}canvas{position:absolute;inset:0;width:100%;height:100%;display:block}.hint{font-size:13px;color:var(--muted);line-height:1.5;margin-top:7px}.gap-note{border:1px solid #ff9f1c;color:#ffd59b;background:#3a2917;padding:8px;border-radius:8px;font-size:13px;line-height:1.5;margin-top:8px}.kbd{border:1px solid var(--line);border-bottom-width:2px;border-radius:5px;padding:1px 5px;background:#0f1627;color:#d9e1ef}.layer-item{border:1px solid var(--line);border-radius:8px;padding:9px;margin-bottom:7px;background:var(--panel2);cursor:pointer}.layer-item.selected{border-color:var(--focus);box-shadow:0 0 0 1px var(--focus) inset}.layer-head{display:flex;justify-content:space-between;gap:8px;font-size:14px}.layer-meta{color:var(--muted);font-size:12px;margin-top:4px;word-break:break-all}.mini{min-height:26px;padding:2px 7px;font-size:12px}.empty{color:var(--muted);font-size:13px;padding:14px 3px}.toast{position:absolute;left:50%;bottom:24px;transform:translateX(-50%);background:#10192b;border:1px solid #52617c;border-radius:9px;padding:10px 14px;box-shadow:0 12px 35px #0008;opacity:0;pointer-events:none;transition:.18s;z-index:4}.toast.show{opacity:1}.swatch{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:6px;border:1px solid #71809a;vertical-align:-1px}.white{background:white}.black{background:black}.gray{background:#7f7f7f}.crop-note{border:1px dashed #f4c04a;color:#ffe8a7;padding:8px;border-radius:8px;font-size:13px;margin-top:8px}
    .layer-head>span{min-width:0;overflow-wrap:anywhere}.layer-head>.mini{flex-shrink:0;align-self:flex-start}
    @media(max-width:1000px){.app{grid-template-columns:260px minmax(0,1fr)}.layers{display:none}}@media(max-width:700px){.app{grid-template-columns:220px minmax(0,1fr)}.sidebar{padding:10px}.grid{grid-template-columns:1fr}}
  </style>
</head>
<body>
<div class="app">
  <aside class="sidebar">
    <h1>道路与楼栋编辑器</h1>
    <div class="subtitle" id="sourceText">正在加载地图…</div>
    <div class="section">
      <div class="section-title">视图</div>
      <div class="grid">
        <button id="fitBtn">适合窗口</button><button id="toggleBaseBtn">查看原图</button>
        <button id="zoomOutBtn">缩小</button><button id="zoomInBtn">放大</button>
        <button id="rotateViewBtn" style="grid-column:1/-1" title="只旋转编辑视图，不改变地图坐标和导出方向">顺时针旋转 90° · 当前 0°</button>
      </div>
      <div class="hint">滚轮缩放；按住空格或使用平移工具拖动画面。</div>
    </div>
    <div class="section">
      <div class="section-title">编辑工具</div>
      <div class="grid" id="toolGrid">
        <button data-tool="select" class="active">选择编辑</button><button data-tool="pan">平移</button>
        <button data-tool="free"><span class="swatch white"></span>画可行区</button><button data-tool="outline"><span class="swatch black"></span>画矩形</button>
        <button data-tool="outlineWidth" style="grid-column:1/-1">调整矩形宽度</button>
      <div id="outlineWidthPanel" class="section" style="grid-column:1/-1;margin-top:0" hidden>
        <label for="outlineWidthSlider">边框线宽 <output id="outlineWidthValue">5 px</output></label>
        <input id="outlineWidthSlider" type="range" min="1" max="100" step="1" value="5" style="width:100%;accent-color:#00c2a8" disabled>
        <button id="matchOutlineWidthBtn" style="width:100%;margin-top:7px" disabled>与楼栋线宽一致</button>
        <div class="hint" id="outlineWidthHint">点击手绘矩形，再拖动滑块调整边框线宽。</div>
      </div>
        <button data-tool="roadSelect">框选道路</button><button id="deleteRoadsBtn" disabled>删除道路</button>
        <button data-tool="fill">点击填充障碍区</button>
        <button data-tool="road"><span class="swatch gray"></span>画道路中心线</button><button data-tool="crop">裁剪保留框</button>
        <button data-tool="copyBuilding">框选复制楼栋</button><button data-tool="pasteBuilding">粘贴楼栋</button>
        <button data-tool="route" style="grid-column:1/-1">验证全局路线</button>
        <button data-tool="entrance" style="grid-column:1/-1">标注楼栋入口</button>
      </div>
      <form id="entrancePanel" class="section" hidden>
        <div class="section-title">楼栋入口标签</div>
        <div class="hint" id="entrancePosition">请先在道路线上点击入口位置。</div>
        <label for="entranceLabel">字符串标签</label>
        <input id="entranceLabel" type="text" placeholder="例如：3栋1单元入口" autocomplete="off" required>
        <div class="grid" style="margin-top:7px">
          <button id="saveEntranceBtn" type="submit" class="primary">保存入口标签</button>
          <button id="cancelEntranceBtn" type="button">取消选点</button>
          <button id="deleteEntranceBtn" type="button" class="danger" hidden>删除入口</button>
        </div>
        <div class="hint">道路附近 12 px 内可吸附。保存后用“选择编辑”拖动入口；修改标签后点击保存。导出时写入本地工程和 JSON。</div>
      </form>
      <div class="grid" style="margin-top:7px"><button id="undoBtn">撤销</button><button id="redoBtn">重做</button></div>
      <div class="hint" id="toolHint">点击标注进行选择；拖动标注可移动。</div>
      <div class="hint" id="roadSelectionStatus" hidden></div>
      <div class="gap-note" id="gapStatus" hidden></div>
    </div>
    <div class="section">
      <div class="section-title">道路中心线</div>
      <button id="thickenRoadsBtn" style="width:100%">加粗道路线</button>
      <div class="hint" id="roadThicknessHint">每次加粗 2 px，作用于全部道路，可撤销。</div>
      <label for="roadColor">道路颜色 <output id="roadColorValue">127</output></label>
      <input id="roadColor" type="range" min="0" max="255" step="1" value="127" style="width:100%;accent-color:#00c2a8">
      <div class="hint">0 为黑色，255 为白色；实时预览并随工程保存。</div>
      <label for="roadWidth">新增道路线宽（像素）</label>
      <input id="roadWidth" type="number" min="1" max="100" step="1" value="9">
      <label class="check"><input id="orthogonalRoads" type="checkbox">道路水平/垂直吸附</label>
      <div class="hint">两点画一段道路，可移动、调整端点与线宽。用“框选道路”和“删除道路”清除道路后，可重新补画。</div>
    </div>
    <div class="section">
      <div class="section-title">矩形轮廓 · 黑色 0</div>
      <label for="outlineWidth">新矩形边框线宽（像素）</label>
      <input id="outlineWidth" type="number" min="1" max="100" step="1" value="5">
      <div class="hint" id="outlineDefaultHint">新矩形默认匹配楼栋线宽；已有矩形可用“调整矩形宽度”修改。</div>
    </div>
    <div class="section">
      <div class="section-title">全局路线验证</div>
      <label for="routeClearance">障碍安全边距（像素）</label>
      <input id="routeClearance" type="number" min="0" max="50" step="1" value="0">
      <button id="clearRouteBtn" style="width:100%;margin-top:7px">清除路线验证</button>
      <div class="crop-note" id="routeStatus">点击“验证全局路线”，再依次点击起点和终点。</div>
      <div class="hint">仅沿道路规划，白色背景、已删除道路和黑色边界不可通行。起终点可在 40 px 内吸附到道路；道路颜色不影响验证。</div>
    </div>
    <div class="section">
      <div class="section-title">选中项</div>
      <div id="selectionEmpty" class="empty">尚未选择标注</div>
      <div id="selectionFields" hidden>
        <label for="selectedName">名称</label><input id="selectedName" type="text">
        <div id="rectFields" class="grid">
          <div><label>X</label><input id="rectX" type="number" step="1"></div><div><label>Y</label><input id="rectY" type="number" step="1"></div>
          <div><label>宽</label><input id="rectW" type="number" min="1" step="1"></div><div><label>高</label><input id="rectH" type="number" min="1" step="1"></div>
        </div>
        <div id="outlineFields" hidden><label for="selectedOutlineWidth">矩形线宽（像素）</label><input id="selectedOutlineWidth" type="number" min="1" max="100" step="1" value="3"></div>
        <div id="lineFields" class="grid" hidden>
          <div><label>X1</label><input id="lineX1" type="number" step="1"></div><div><label>Y1</label><input id="lineY1" type="number" step="1"></div>
          <div><label>X2</label><input id="lineX2" type="number" step="1"></div><div><label>Y2</label><input id="lineY2" type="number" step="1"></div>
          <div style="grid-column:1/-1"><label>线宽 px</label><input id="selectedLineWidth" type="number" min="1" max="100" step="1"></div>
        </div>
        <div id="buildingFields" hidden>
          <div class="grid">
            <div><label>中心 X</label><input id="buildingX" type="number" step="1"></div><div><label>中心 Y</label><input id="buildingY" type="number" step="1"></div>
            <div><label>宽</label><input id="buildingW" type="number" min="3" step="1"></div><div><label>高</label><input id="buildingH" type="number" min="3" step="1"></div>
          </div>
          <div class="hint" id="buildingChildCount"></div>
          <button id="duplicateBuildingBtn" style="width:100%;margin-top:7px">复制一个</button>
        </div>
        <div id="fillFields" hidden><div class="hint" id="fillInfo"></div></div>
        <button id="deleteBtn" class="danger" style="width:100%;margin-top:9px">删除选中项</button>
      </div>
    </div>
    <div class="section">
      <div class="section-title">裁剪</div>
      <div class="crop-note" id="cropStatus">未设置裁剪框，将导出完整尺寸。</div>
      <button id="toggleCropPreviewBtn" style="width:100%;margin-top:7px" disabled>预览裁剪结果</button>
      <button id="clearCropBtn" style="width:100%;margin-top:7px">清除裁剪框</button>
    </div>
    <div class="section stack">
      <div class="section-title">保存</div>
      <button id="exportBtn" class="primary">导出 PNG + JSON</button>
      <button id="importBtn">导入标注 JSON</button>
      <input id="importFile" type="file" accept="application/json" hidden>
      <button id="clearBtn" class="danger">清空人工标注</button>
      <div class="hint">点击导出时会同时保存本地工程；下次用同一张原图启动后自动恢复。JSON 导入保留为换电脑或恢复历史版本使用。</div>
    </div>
  </aside>
  <main class="workspace">
    <div class="topbar"><div class="status" id="modeText">选择编辑</div><div class="status" id="countText">0 个标注</div></div>
    <div class="stage" id="stage" tabindex="0"><canvas id="canvas"></canvas><div class="toast" id="toast"></div></div>
  </main>
  <aside class="layers"><div class="section-title">人工标注列表</div><div id="layerList"></div></aside>
</div>
<script>
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const canvas = $('canvas'), stage = $('stage'), ctx = canvas.getContext('2d');
  const state = {config:null, base:null, original:null, baseCanvas:null, showOriginal:false, viewRotation:0, tool:'select', zoom:1, panX:0, panY:0, space:false, lineStart:null, linePreview:null, fillPreview:null, gapHints:[], cropPreview:false, draft:null, drag:null, selected:null, entranceDraft:null, buildingTemplate:null, pastePreview:null, route:{startClick:null,start:null,goalClick:null,goal:null,path:[],status:'点击“验证全局路线”，再依次点击起点和终点。'}, annotations:{freeRects:[],obstacleRects:[],obstacleFills:[],obstacleLines:[],roadLines:[],outlineRects:[],roadErases:[],buildingCopies:[],specialPoints:[]}, cropRect:null, undo:[], redo:[], order:0};
  const labels = {entrance:'标注楼栋入口',select:'选择编辑',pan:'平移',free:'画可行区',outline:'画矩形',outlineWidth:'调整矩形宽度',roadSelect:'框选道路',fill:'点击填充障碍区',road:'画道路中心线',crop:'裁剪保留框',copyBuilding:'框选复制楼栋',pasteBuilding:'粘贴楼栋',route:'验证全局路线'};
  const hints = {entrance:'先点击道路上的入口位置，再填写字符串标签并保存；可选中已有入口修改。',select:'点击标注进行选择；拖动标注可移动。',pan:'拖动画面；滚轮缩放。',free:'在地图上拖出矩形，白色区域会覆盖底图。',outline:'拖出黑色空心矩形，默认与楼栋线宽一致；选中后可移动、调整宽高和线宽。',outlineWidth:'点击手绘矩形，拖动下方滑块调整边框线宽；可一键匹配楼栋线宽。',roadSelect:'拖出矩形，高亮框内道路，再点击“删除道路”。支持自动道路和手工道路，Esc 取消。',fill:'点击封闭白区进行预览；若有 1–8 px 小缺口，会以橙色标出。',road:'依次点击两点画加粗道路，颜色由道路颜色滑块控制；可在选择编辑中移动、调整端点和线宽。Esc 取消。',crop:'拖出要保留的矩形；导出时框外区域会被裁掉。',copyBuilding:'拖出要复制的矩形；框内当前灰度内容会按所选范围原样复制。',pasteBuilding:'移动鼠标预览，单击放置楼栋；可连续粘贴。',route:'依次点击道路上的起点和终点，验证当前道路是否连通；40 px 内自动吸附到道路。'};
  const clone = value => JSON.parse(JSON.stringify(value));
  const patchCache = new Map();
  const mapCache = {raw:null,semantic:null,display:null};
  state.roadStyle = {color:127,extraWidthPx:0};
  let roadColorEditing = false;
  let outlineWidthEditingId = null;
  function normalizedRoadStyle(style){
    const color=Number(style?.color??127),extra=Number(style?.extraWidthPx??0);
    return {color:clamp(Number.isFinite(color)?Math.round(color):127,0,255),extraWidthPx:clamp(Number.isFinite(extra)?Math.round(extra/2)*2:0,0,100)};
  }
  function invalidateMap(){mapCache.raw=null;mapCache.semantic=null;mapCache.display=null;}
  function updateRoadStyleUI(){
    updateRoadSelectionUI();$('roadColor').value=state.roadStyle.color;$('roadColorValue').textContent=state.roadStyle.color;
    $('roadColor').disabled=state.showOriginal;
    $('thickenRoadsBtn').disabled=state.showOriginal||state.roadStyle.extraWidthPx>=100;
    $('roadThicknessHint').textContent=`全部道路已加粗 ${state.roadStyle.extraWidthPx} px；每次增加 2 px，可撤销。`;
    document.querySelectorAll('.swatch.gray').forEach(el=>el.style.background=`rgb(${state.roadStyle.color},${state.roadStyle.color},${state.roadStyle.color})`);
  }
  function thickenRoads(){
    if(state.showOriginal||state.roadStyle.extraWidthPx>=100)return;
    pushUndo();state.roadStyle.extraWidthPx+=2;mapCache.semantic=null;mapCache.display=null;
    updateUI();toast(`全部道路线已加粗 ${state.roadStyle.extraWidthPx} px`);
  }
  function changeRoadColor(){
    if(state.showOriginal){updateRoadStyleUI();return;}
    const color=clamp(Math.round(Number($('roadColor').value)),0,255);
    if(color===state.roadStyle.color)return;
    if(!roadColorEditing){pushUndo(false);roadColorEditing=true;}
    state.roadStyle.color=color;mapCache.display=null;
    updateRoadStyleUI();$('undoBtn').disabled=!state.undo.length;$('redoBtn').disabled=true;render();
  }
  // Grow only the semantic road mask; black building edges always take priority.
  function thickenRoadPixels(pixels,width,height,extraWidthPx,underlay=null){
    const radius=Math.round(extraWidthPx/2);
    if(!radius)return;
    const limit=radius*3,distance=new Uint16Array(width*height);distance.fill(limit+4);
    for(let i=0;i<distance.length;i++)if(pixels[i*4]===127&&pixels[i*4+3])distance[i]=0;
    for(let y=0;y<height;y++)for(let x=0;x<width;x++){
      const i=y*width+x;
      if(x)distance[i]=Math.min(distance[i],distance[i-1]+3);
      if(y){distance[i]=Math.min(distance[i],distance[i-width]+3);if(x)distance[i]=Math.min(distance[i],distance[i-width-1]+4);if(x+1<width)distance[i]=Math.min(distance[i],distance[i-width+1]+4);}
    }
    for(let y=height-1;y>=0;y--)for(let x=width-1;x>=0;x--){
      const i=y*width+x;
      if(x+1<width)distance[i]=Math.min(distance[i],distance[i+1]+3);
      if(y+1<height){distance[i]=Math.min(distance[i],distance[i+width]+3);if(x)distance[i]=Math.min(distance[i],distance[i+width-1]+4);if(x+1<width)distance[i]=Math.min(distance[i],distance[i+width+1]+4);}
      const offset=i*4,transparent=pixels[offset+3]===0;
      if(distance[i]<=limit&&(transparent||pixels[offset]!==0)&&!(transparent&&underlay&&underlay[offset]===0&&underlay[offset+3])){
        pixels[offset]=pixels[offset+1]=pixels[offset+2]=127;pixels[offset+3]=255;
      }
    }
  }
  function recolorRoadPixels(pixels,color){
    for(let i=0;i<pixels.length;i+=4)if(pixels[i]===127)pixels[i]=pixels[i+1]=pixels[i+2]=color;
  }
  function mapSurface(){const surface=document.createElement('canvas');surface.width=state.config.width;surface.height=state.config.height;return surface;}
  function boundedRect(rect,limit={x:0,y:0,width:state.config.width,height:state.config.height}){
    const x=Math.max(limit.x,Math.floor(rect.x)),y=Math.max(limit.y,Math.floor(rect.y));
    const right=Math.min(limit.x+limit.width,Math.ceil(rect.x+rect.width)),bottom=Math.min(limit.y+limit.height,Math.ceil(rect.y+rect.height));
    return right>x&&bottom>y?{x,y,width:right-x,height:bottom-y}:null;
  }
  function eraseRoadPixels(pixels){let count=0;for(let i=0;i<pixels.length;i+=4)if(pixels[i]===127&&pixels[i+3]){pixels[i]=pixels[i+1]=pixels[i+2]=255;count++;}return count;}
  function eraseRoadRect(target,rect){const r=boundedRect(rect);if(!r)return;const data=target.getImageData(r.x,r.y,r.width,r.height);eraseRoadPixels(data.data);target.putImageData(data,r.x,r.y);}
  function paintOutline(target,item){
    // Keep the border inside the rectangle and leave its interior untouched.
    const x=Math.round(item.x),y=Math.round(item.y),w=Math.round(item.width),h=Math.round(item.height),t=Math.min(w,h,Math.max(1,normalizedOutlineWidth(item.widthPx)));
    target.fillStyle='#000';target.fillRect(x,y,w,t);target.fillRect(x,y+h-t,w,t);target.fillRect(x,y,t,h);target.fillRect(x+w-t,y,t,h);
  }
  function paintLayer(target,type,item){
    if(type==='entrance')return; // Point labels are metadata, never map pixels.
    target.save();
    if(isLineType(type))paintMapLine(target,item,type);
    else if(type==='outline')paintOutline(target,item);
    else if(type==='building')paintBuilding(target,item);
    else if(type==='fill')paintObstacleFill(target,item);
    else{target.fillStyle=type==='free'?'#fff':'#000';target.fillRect(item.x,item.y,item.width,item.height);}
    target.restore();
  }
  function paintStyledLayer(target,type,item,extra){
    if(!extra||!['road','building'].includes(type)){paintLayer(target,type,item);return;}
    const b=type==='building'?buildingBounds(item):{x:Math.min(item.x1,item.x2),y:Math.min(item.y1,item.y2),width:Math.abs(item.x2-item.x1),height:Math.abs(item.y2-item.y1)};
    const pad=Math.ceil((extra+(type==='road'?item.widthPx:0))/2)+2;
    const r=boundedRect({x:b.x-pad,y:b.y-pad,width:b.width+pad*2,height:b.height+pad*2});if(!r)return;
    const patch=document.createElement('canvas');patch.width=r.width;patch.height=r.height;
    const c=patch.getContext('2d',{willReadFrequently:true});c.translate(-r.x,-r.y);paintLayer(c,type,item);
    const data=c.getImageData(0,0,r.width,r.height),pixels=data.data;
    for(let i=0;i<pixels.length;i+=4){if(pixels[i+3]<128){pixels[i+3]=0;continue;}const v=pixels[i]<64?0:pixels[i]<192?127:255;pixels[i]=pixels[i+1]=pixels[i+2]=v;pixels[i+3]=255;}
    thickenRoadPixels(pixels,r.width,r.height,extra,target.getImageData(r.x,r.y,r.width,r.height).data);
    c.putImageData(data,0,0);target.drawImage(patch,r.x,r.y);
  }
  function composeLayers(extra){
    const {width,height}=state.config,surface=mapSurface(),target=surface.getContext('2d',{willReadFrequently:true});
    target.imageSmoothingEnabled=false;target.drawImage(state.base,0,0);forceMapPalette(target,width,height);
    if(extra){const data=target.getImageData(0,0,width,height);thickenRoadPixels(data.data,width,height,extra);target.putImageData(data,0,0);}
    // Apply erasures in edit order, after thickening. Later road strokes can
    // repaint an erased area, and changing the road width cannot resurrect it.
    for(const {type,item} of allItems()){
      if(type==='roadErase')eraseRoadRect(target,item);
      else paintStyledLayer(target,type,item,extra);
    }
    forceMapPalette(target,width,height);return surface;
  }
  function composedMap(mode='display'){
    const {width,height}=state.config;
    if(mode==='raw'){if(!mapCache.raw)mapCache.raw=composeLayers(0);return mapCache.raw;}
    if(!mapCache.semantic)mapCache.semantic=composeLayers(state.roadStyle.extraWidthPx);
    if(mode==='semantic'||state.roadStyle.color===127)return mapCache.semantic;
    if(!mapCache.display){
      const surface=mapSurface(),target=surface.getContext('2d',{willReadFrequently:true});target.drawImage(mapCache.semantic,0,0);
      const data=target.getImageData(0,0,width,height);recolorRoadPixels(data.data,state.roadStyle.color);target.putImageData(data,0,0);mapCache.display=surface;
    }
    return mapCache.display;
  }
  state.roadSelection=null;
  function roadSelectionFromPixels(pixels,rect){
    const runs=[];let area=0;
    for(let y=0;y<rect.height;y++){
      let start=-1;
      for(let x=0;x<=rect.width;x++){
        const road=x<rect.width&&pixels[(y*rect.width+x)*4]===127;
        if(road&&start<0)start=x;
        if(!road&&start>=0){runs.push([rect.y+y,rect.x+start,rect.x+x-1]);area+=x-start;start=-1;}
      }
    }
    return {...rect,runs,area};
  }
  function updateRoadSelectionUI(){
    const selection=state.roadSelection;
    $('deleteRoadsBtn').disabled=state.showOriginal||(!selection&&state.selected?.type!=='road');
    $('roadSelectionStatus').hidden=!selection;
    $('roadSelectionStatus').textContent=selection?`已框选 ${selection.area} 个道路像素；点击“删除道路”清除框内道路，Esc 取消。`:'';
  }
  function selectRoads(rect){
    state.roadSelection=null;state.selected=null;
    const r=boundedRect(rect,state.cropPreview&&state.cropRect?state.cropRect:undefined);
    if(r){const pixels=composedMap('semantic').getContext('2d').getImageData(r.x,r.y,r.width,r.height).data;const selection=roadSelectionFromPixels(pixels,r);if(selection.area)state.roadSelection=selection;}
    updateRoadSelectionUI();updateSelectionFields();render();
    toast(state.roadSelection?'道路已高亮，点击“删除道路”确认':'框内没有道路');
  }
  function deleteRoads(){
    if(state.showOriginal)return;
    const selection=state.roadSelection;
    if(!selection){if(state.selected?.type==='road')deleteSelected();return;}
    const {x,y,width,height}=selection;pushUndo();
    state.annotations.roadErases.push({id:uid('roadErase'),name:'道路删除范围 '+(state.annotations.roadErases.length+1),x,y,width,height,drawOrder:nextOrder()});
    state.selected=null;updateUI();toast('已删除框内道路；可撤销或重新补画');
  }
  const snapshot = () => ({annotations:clone(state.annotations),cropRect:clone(state.cropRect),roadStyle:clone(state.roadStyle),order:state.order});
  const restore = snap => {state.entranceDraft=null;state.roadStyle=normalizedRoadStyle(snap.roadStyle);roadColorEditing=false;outlineWidthEditingId=null;state.annotations=clone(snap.annotations);state.cropRect=clone(snap.cropRect);state.cropPreview=false;state.order=snap.order;state.selected=null;state.roadSelection=null;state.lineStart=null;state.linePreview=null;state.fillPreview=null;state.gapHints=[];invalidateRoute();updateUI();render();};
  function pushUndo(affectsRoute=true){outlineWidthEditingId=null;state.roadSelection=null;updateRoadSelectionUI();if(affectsRoute)invalidateRoute();state.undo.push(snapshot());if(state.undo.length>100)state.undo.shift();state.redo=[];}
  function toast(message){const el=$('toast');el.textContent=message;el.classList.add('show');clearTimeout(toast.timer);toast.timer=setTimeout(()=>el.classList.remove('show'),2200);}
  function clamp(v,min,max){return Math.max(min,Math.min(max,v));}
  function orthogonalEnabled(){return $('orthogonalRoads').checked;}
  function snapOrthogonal(anchor,point,force=false){if(!force&&!orthogonalEnabled())return {x:point.x,y:point.y};return Math.abs(point.x-anchor.x)>=Math.abs(point.y-anchor.y)?{x:point.x,y:anchor.y}:{x:anchor.x,y:point.y};}
  const isLineType = type => type==='line'||type==='road';
  function snapForLine(anchor,point,type=state.tool){return type==='road'?($('orthogonalRoads').checked?snapOrthogonal(anchor,point,true):{...point}):snapOrthogonal(anchor,point);}
  function nextOrder(){state.order+=1;return state.order;}
  function allItems(){return [...state.annotations.freeRects.map(item=>({type:'free',item})),...state.annotations.obstacleRects.map(item=>({type:'obstacle',item})),...state.annotations.obstacleFills.map(item=>({type:'fill',item})),...state.annotations.obstacleLines.map(item=>({type:'line',item})),...state.annotations.roadLines.map(item=>({type:'road',item})),...state.annotations.outlineRects.map(item=>({type:'outline',item})),...state.annotations.roadErases.map(item=>({type:'roadErase',item})),...state.annotations.buildingCopies.map(item=>({type:'building',item})),...state.annotations.specialPoints.map(item=>({type:'entrance',item}))].sort((a,b)=>(a.item.drawOrder||0)-(b.item.drawOrder||0));}
  function listFor(type){return type==='entrance'?state.annotations.specialPoints:type==='free'?state.annotations.freeRects:type==='obstacle'?state.annotations.obstacleRects:type==='fill'?state.annotations.obstacleFills:type==='line'?state.annotations.obstacleLines:type==='road'?state.annotations.roadLines:type==='outline'?state.annotations.outlineRects:type==='roadErase'?state.annotations.roadErases:state.annotations.buildingCopies;}
  function selectedItem(){if(!state.selected)return null;return listFor(state.selected.type).find(x=>x.id===state.selected.id)||null;}
  function uid(prefix){return prefix+'_'+Date.now().toString(36)+'_'+Math.random().toString(36).slice(2,7);}
  function rotatedSize(){const quarter=(state.viewRotation/90)%2;return quarter?{width:state.config.height,height:state.config.width}:{width:state.config.width,height:state.config.height};}
  function rotatedPoint(p){const w=state.config.width,h=state.config.height,r=state.viewRotation;if(r===90)return {x:h-p.y,y:p.x};if(r===180)return {x:w-p.x,y:h-p.y};if(r===270)return {x:p.y,y:w-p.x};return {x:p.x,y:p.y};}
  function imagePoint(event){const box=canvas.getBoundingClientRect(),rx=(event.clientX-box.left-state.panX)/state.zoom,ry=(event.clientY-box.top-state.panY)/state.zoom,w=state.config.width,h=state.config.height,r=state.viewRotation;if(r===90)return {x:ry,y:h-rx};if(r===180)return {x:w-rx,y:h-ry};if(r===270)return {x:w-ry,y:rx};return {x:rx,y:ry};}
  function screenPoint(p){const rotated=rotatedPoint(p);return {x:state.panX+rotated.x*state.zoom,y:state.panY+rotated.y*state.zoom};}
  function fit(){if(!state.config)return;const pad=24,size=rotatedSize();state.zoom=Math.min((stage.clientWidth-pad*2)/size.width,(stage.clientHeight-pad*2)/size.height);state.zoom=clamp(state.zoom,.05,12);state.panX=(stage.clientWidth-size.width*state.zoom)/2;state.panY=(stage.clientHeight-size.height*state.zoom)/2;render();}
  function fitCrop(){if(!state.cropRect){state.cropPreview=false;fit();return;}const c=state.cropRect,corners=[{x:c.x,y:c.y},{x:c.x+c.width,y:c.y},{x:c.x+c.width,y:c.y+c.height},{x:c.x,y:c.y+c.height}].map(rotatedPoint),xs=corners.map(p=>p.x),ys=corners.map(p=>p.y),minX=Math.min(...xs),maxX=Math.max(...xs),minY=Math.min(...ys),maxY=Math.max(...ys),pad=24,width=maxX-minX,height=maxY-minY;state.zoom=clamp(Math.min((stage.clientWidth-pad*2)/width,(stage.clientHeight-pad*2)/height),.05,12);state.panX=(stage.clientWidth-width*state.zoom)/2-minX*state.zoom;state.panY=(stage.clientHeight-height*state.zoom)/2-minY*state.zoom;render();}
  function resize(){const dpr=window.devicePixelRatio||1;canvas.width=Math.round(stage.clientWidth*dpr);canvas.height=Math.round(stage.clientHeight*dpr);render();}
  function buildingLocalToWorld(item,p){const angle=(Number(item.rotation)||0)*Math.PI/180,c=Math.cos(angle),s=Math.sin(angle),sx=Number(item.scaleX)||1,sy=Number(item.scaleY)||1;return {x:item.x+p.x*sx*c-p.y*sy*s,y:item.y+p.x*sx*s+p.y*sy*c};}
  function buildingWorldPoints(item){return item.points.map(p=>buildingLocalToWorld(item,p));}
  function buildingBounds(item){const pts=buildingWorldPoints(item),xs=pts.map(p=>p.x),ys=pts.map(p=>p.y);return {x:Math.min(...xs),y:Math.min(...ys),width:Math.max(...xs)-Math.min(...xs),height:Math.max(...ys)-Math.min(...ys)};}
  function polygonPath(target,points){target.beginPath();target.moveTo(points[0].x,points[0].y);for(let i=1;i<points.length;i++)target.lineTo(points[i].x,points[i].y);target.closePath();}
  function patchEntry(data){if(!data)return null;if(patchCache.has(data))return patchCache.get(data);const image=new Image(),entry={source:image,ready:false,promise:null};entry.promise=new Promise((resolve,reject)=>{image.onload=()=>{entry.ready=true;invalidateMap();render();resolve(image);};image.onerror=()=>reject(new Error('楼栋复制图块加载失败'));});image.src=data;patchCache.set(data,entry);return entry;}
  function cachePatch(data,source){patchCache.set(data,{source,ready:true,promise:Promise.resolve(source)});}
  async function preloadBuildingPatches(){const entries=state.annotations.buildingCopies.filter(item=>item.patchData).map(item=>patchEntry(item.patchData));await Promise.all(entries.map(entry=>entry.promise));}
  function paintBuilding(target,item,preview=false){const points=buildingWorldPoints(item);if(points.length<3)return;target.save();target.globalAlpha=preview?.55:1;if(item.patchData){const entry=patchEntry(item.patchData),bounds=buildingBounds(item);target.imageSmoothingEnabled=false;if(entry?.ready)target.drawImage(entry.source,bounds.x,bounds.y,bounds.width,bounds.height);if(preview){target.strokeStyle='#2f80ff';target.lineWidth=2/state.zoom;target.setLineDash([8/state.zoom,5/state.zoom]);target.strokeRect(bounds.x,bounds.y,bounds.width,bounds.height);}target.restore();return;}target.fillStyle='#fff';target.strokeStyle=preview?'#2f80ff':'#000';target.lineWidth=Math.max(1,Number(item.lineWidth)||3);target.lineJoin='round';polygonPath(target,points);target.fill();target.stroke();const children=Array.isArray(item.children)?[...item.children].sort((a,b)=>(a.drawOrder||0)-(b.drawOrder||0)):[];for(const child of children){if(isLineType(child.type)){const a=buildingLocalToWorld(item,{x:child.x1,y:child.y1}),b=buildingLocalToWorld(item,{x:child.x2,y:child.y2});paintMapLine(target,{x1:a.x,y1:a.y,x2:b.x,y2:b.y,widthPx:Math.max(1,(Number(child.widthPx)||3)*((Math.abs(item.scaleX||1)+Math.abs(item.scaleY||1))/2))},child.type);}else{const corners=[{x:child.x,y:child.y},{x:child.x+child.width,y:child.y},{x:child.x+child.width,y:child.y+child.height},{x:child.x,y:child.y+child.height}].map(p=>buildingLocalToWorld(item,p));target.fillStyle=child.type==='free'?'#fff':'#000';polygonPath(target,corners);target.fill();}}target.restore();}
  const lineRasterCache = new Map();
  let lineRasterPixels = 0;
  function paintMapLine(target,item,type='line'){
    const width=Math.max(1,Number(item.widthPx)||1),pad=Math.ceil(width/2)+1;
    const x=Math.floor(Math.min(item.x1,item.x2))-pad,y=Math.floor(Math.min(item.y1,item.y2))-pad;
    const w=Math.ceil(Math.max(item.x1,item.x2))-x+pad+1,h=Math.ceil(Math.max(item.y1,item.y2))-y+pad+1;
    const key=[type,item.x1,item.y1,item.x2,item.y2,width].join(',');let patch=lineRasterCache.get(key);
    if(!patch){
      patch=document.createElement('canvas');patch.width=w;patch.height=h;
      const c=patch.getContext('2d'),value=type==='road'?127:0;
      c.strokeStyle='#000';c.lineWidth=width;c.lineCap='round';c.beginPath();c.moveTo(item.x1-x,item.y1-y);c.lineTo(item.x2-x,item.y2-y);c.stroke();
      const image=c.getImageData(0,0,w,h),pixels=image.data;
      for(let i=0;i<pixels.length;i+=4){pixels[i]=pixels[i+1]=pixels[i+2]=value;pixels[i+3]=pixels[i+3]>=128?255:0;}
      c.putImageData(image,0,0);
      if(lineRasterPixels+w*h>8000000){lineRasterCache.clear();lineRasterPixels=0;}
      if(w*h<=8000000){lineRasterCache.set(key,patch);lineRasterPixels+=w*h;}
    }
    target.imageSmoothingEnabled=false;target.drawImage(patch,x,y);
  }
  function paintObstacleFill(target,item,color='#000',alpha=1){target.save();target.fillStyle=color;target.globalAlpha=alpha;for(const run of item.runs||[]){const [y,x1,x2]=run;target.fillRect(x1,y,x2-x1+1,1);}target.restore();}
  function pointInObstacleFill(point,item){const x=Math.floor(point.x),y=Math.floor(point.y);if(x<item.x||x>=item.x+item.width||y<item.y||y>=item.y+item.height)return false;for(const run of item.runs||[]){if(run[0]>y)break;if(run[0]===y&&x>=run[1]&&x<=run[2])return true;}return false;}
  function detectGapHints(pixels,width,height,start,visited,originalArea,queue){
    const total=width*height,maxRadius=8,distance=new Uint8Array(total);distance.fill(maxRadius+1);
    for(let index=0;index<total;index++)if(pixels[index*4]<64)distance[index]=0;
    for(let y=0;y<height;y++){const row=y*width;for(let x=0;x<width;x++){const index=row+x;if(!distance[index])continue;let best=distance[index];if(x>0)best=Math.min(best,distance[index-1]+1);if(y>0)best=Math.min(best,distance[index-width]+1);distance[index]=Math.min(maxRadius+1,best);}}
    for(let y=height-1;y>=0;y--){const row=y*width;for(let x=width-1;x>=0;x--){const index=row+x;if(!distance[index])continue;let best=distance[index];if(x<width-1)best=Math.min(best,distance[index+1]+1);if(y<height-1)best=Math.min(best,distance[index+width]+1);distance[index]=Math.min(maxRadius+1,best);}}
    const sealed=new Uint8Array(total);let chosenRadius=0;
    for(let radius=1;radius<=maxRadius;radius++){
      if(distance[start]<=radius)continue;
      let head=0,tail=0,area=0,touchesEdge=false;queue[tail++]=start;sealed[start]=radius;
      while(head<tail){const index=queue[head++],x=index%width,y=Math.floor(index/width);area++;if(x===0||y===0||x===width-1||y===height-1)touchesEdge=true;let next;if(x>0){next=index-1;if(visited[next]&&distance[next]>radius&&sealed[next]!==radius){sealed[next]=radius;queue[tail++]=next;}}if(x<width-1){next=index+1;if(visited[next]&&distance[next]>radius&&sealed[next]!==radius){sealed[next]=radius;queue[tail++]=next;}}if(y>0){next=index-width;if(visited[next]&&distance[next]>radius&&sealed[next]!==radius){sealed[next]=radius;queue[tail++]=next;}}if(y<height-1){next=index+width;if(visited[next]&&distance[next]>radius&&sealed[next]!==radius){sealed[next]=radius;queue[tail++]=next;}}}
      if(!touchesEdge&&area>=20&&area<=originalArea*.8){chosenRadius=radius;break;}
    }
    if(!chosenRadius)return [];
    const candidate=new Uint8Array(total),limit=chosenRadius*2+2;
    const classifyRay=(x,y,dx,dy)=>{for(let step=1;step<=limit;step++){const nx=x+dx*step,ny=y+dy*step;if(nx<0||ny<0||nx>=width||ny>=height)return 2;const index=ny*width+nx;if(pixels[index*4]<64)return 0;if(distance[index]<=chosenRadius)continue;if(!visited[index])return 0;return sealed[index]===chosenRadius?1:2;}return 0;};
    for(let index=0;index<total;index++){
      if(!visited[index]||pixels[index*4]<64||distance[index]>chosenRadius)continue;
      const x=index%width,y=Math.floor(index/width),left=classifyRay(x,y,-1,0),right=classifyRay(x,y,1,0),up=classifyRay(x,y,0,-1),down=classifyRay(x,y,0,1);
      if((left&&right&&left!==right)||(up&&down&&up!==down))candidate[index]=1;
    }
    const seen=new Uint8Array(total),hints=[];
    for(let seed=0;seed<total;seed++){
      if(!candidate[seed]||seen[seed])continue;
      let head=0,tail=0,sumX=0,sumY=0,count=0,minX=width,minY=height,maxX=0,maxY=0;queue[tail++]=seed;seen[seed]=1;
      while(head<tail){const index=queue[head++],x=index%width,y=Math.floor(index/width);sumX+=x;sumY+=y;count++;minX=Math.min(minX,x);maxX=Math.max(maxX,x);minY=Math.min(minY,y);maxY=Math.max(maxY,y);for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){if(!dx&&!dy)continue;const nx=x+dx,ny=y+dy;if(nx<0||ny<0||nx>=width||ny>=height)continue;const next=ny*width+nx;if(candidate[next]&&!seen[next]){seen[next]=1;queue[tail++]=next;}}}
      if(count)hints.push({x:Math.round(sumX/count),y:Math.round(sumY/count),radius:chosenRadius,width:maxX-minX+1,height:maxY-minY+1,count});
    }
    return hints.sort((a,b)=>b.count-a.count).slice(0,12);
  }
  function buildObstacleFillCandidate(point){
    const width=state.config.width,height=state.config.height,startX=clamp(Math.floor(point.x),0,width-1),startY=clamp(Math.floor(point.y),0,height-1),surface=document.createElement('canvas');
    surface.width=width;surface.height=height;
    const surfaceContext=surface.getContext('2d',{willReadFrequently:true}),old=state.showOriginal;
    try{state.showOriginal=false;drawScene(surfaceContext,0,0,false,'semantic');}finally{state.showOriginal=old;}
    forceMapPalette(surfaceContext,width,height);
    const pixels=surfaceContext.getImageData(0,0,width,height).data,start=startY*width+startX;
    if(pixels[start*4]<64)return {error:'这里已经是黑色障碍，请点击封闭白色区域内部。'};
    const total=width*height,visited=new Uint8Array(total),queue=new Int32Array(total);let head=0,tail=0,area=0,minX=startX,maxX=startX,minY=startY,maxY=startY,touchesEdge=false;
    visited[start]=1;queue[tail++]=start;
    while(head<tail){const index=queue[head++],x=index%width,y=Math.floor(index/width);area++;if(x<minX)minX=x;if(x>maxX)maxX=x;if(y<minY)minY=y;if(y>maxY)maxY=y;if(x===0||y===0||x===width-1||y===height-1)touchesEdge=true;let next;if(x>0){next=index-1;if(!visited[next]&&pixels[next*4]>=64){visited[next]=1;queue[tail++]=next;}}if(x<width-1){next=index+1;if(!visited[next]&&pixels[next*4]>=64){visited[next]=1;queue[tail++]=next;}}if(y>0){next=index-width;if(!visited[next]&&pixels[next*4]>=64){visited[next]=1;queue[tail++]=next;}}if(y<height-1){next=index+width;if(!visited[next]&&pixels[next*4]>=64){visited[next]=1;queue[tail++]=next;}}}
    const ratio=area/total,gapHints=detectGapHints(pixels,width,height,start,visited,area,queue);
    if(gapHints.length)return {error:`检测到 ${gapHints.length} 处疑似缺口，已用橙色圆圈标出。请放大检查并调整现有边界后再填充。`,gapHints};
    if(touchesEdge)return {error:'填充范围连接到地图边缘，但没有可靠定位到 1–8 px 小缺口。请放大后人工检查边界。'};
    if(ratio>.35)return {error:`填充范围达到整张地图的 ${Math.round(ratio*100)}%，但没有可靠定位到小缺口。请放大后人工检查边界。`};
    const runs=[];for(let y=minY;y<=maxY;y++){let x=minX;while(x<=maxX){while(x<=maxX&&!visited[y*width+x])x++;if(x>maxX)break;const x1=x;while(x<=maxX&&visited[y*width+x])x++;runs.push([y,x1,x-1]);}}
    return {x:minX,y:minY,width:maxX-minX+1,height:maxY-minY+1,area,runs};
  }
  function updateGapStatus(){const el=$('gapStatus'),hints=state.gapHints||[];el.hidden=!hints.length;if(!hints.length){el.textContent='';return;}const coordinates=hints.slice(0,6).map((hint,index)=>`${index+1}. (${hint.x}, ${hint.y})`).join('　');el.textContent=`橙色为疑似缺口，共 ${hints.length} 处：${coordinates}${hints.length>6?' …':''}。请放大检查并调整现有边界后，再点击填充；Esc 清除提示。`;}
  function handleObstacleFillClick(point){if(state.fillPreview&&pointInObstacleFill(point,state.fillPreview)){pushUndo();const list=state.annotations.obstacleFills,item={...state.fillPreview,id:uid('fill'),name:'封闭障碍区 '+(list.length+1),drawOrder:nextOrder()};list.push(item);state.fillPreview=null;state.gapHints=[];state.selected={type:'fill',id:item.id};updateUI();toast(`已填充 ${item.area} 个像素`);return;}const candidate=buildObstacleFillCandidate(point);state.gapHints=candidate.gapHints||[];updateGapStatus();if(candidate.error){state.fillPreview=null;render();toast(candidate.error);return;}state.fillPreview=candidate;state.gapHints=[];state.selected=null;updateGapStatus();render();toast(`预览 ${candidate.area} 个像素；再次点击红色区域确认，Esc 取消`);}
  function updatePlannerUI(){const status=$('routeStatus'),clear=$('clearRouteBtn');if(status)status.textContent=state.route.status;if(clear)clear.disabled=!state.route.startClick&&!state.route.goalClick&&!state.route.path.length;}
  function invalidateRoute(){if(!state.route.startClick&&!state.route.goalClick&&!state.route.path.length)return;state.route={startClick:null,start:null,goalClick:null,goal:null,path:[],status:'地图已修改，请重新选择起点和终点。'};updatePlannerUI();}
  function clearRoute(showMessage=true){state.route={startClick:null,start:null,goalClick:null,goal:null,path:[],status:'点击“验证全局路线”，再依次点击起点和终点。'};updatePlannerUI();render();if(showMessage)toast('已清除路线验证');}
  function planningGridFromPixels(pixels,width,height,clearance=0,crop=null){
    // One cell per pixel preserves narrow roads and one-pixel breaks. Display colors
    // never enter this mask: semantic road pixels are always 127.
    const free=new Uint8Array(width*height),stride=width+1;
    clearance=clamp(Math.round(Number(clearance)||0),0,50);
    const prefix=clearance?new Uint32Array((height+1)*stride):null;
    if(prefix)for(let y=0;y<height;y++){
      let black=0;
      for(let x=0;x<width;x++){
        const i=(y*width+x)*4;
        if(pixels[i]===0&&pixels[i+3])black++;
        prefix[(y+1)*stride+x+1]=prefix[y*stride+x+1]+black;
      }
    }
    let roadPixels=0,usableRoadPixels=0;
    for(let y=0;y<height;y++)for(let x=0;x<width;x++){
      const i=y*width+x;
      if(pixels[i*4]!==127||!pixels[i*4+3])continue;
      if(crop&&(x+.5<crop.x||x+.5>=crop.x+crop.width||y+.5<crop.y||y+.5>=crop.y+crop.height))continue;
      roadPixels++;
      if(prefix){
        const x0=Math.max(0,x-clearance),y0=Math.max(0,y-clearance),x1=Math.min(width,x+clearance+1),y1=Math.min(height,y+clearance+1);
        if(prefix[y1*stride+x1]-prefix[y0*stride+x1]-prefix[y1*stride+x0]+prefix[y0*stride+x0])continue;
      }
      free[i]=1;usableRoadPixels++;
    }
    return {width,height,cellSize:1,cols:width,rows:height,free,clearance,roadPixels,usableRoadPixels};
  }
  function buildPlanningGrid(){
    const {width,height}=state.config;
    const pixels=composedMap('semantic').getContext('2d',{willReadFrequently:true}).getImageData(0,0,width,height).data;
    return planningGridFromPixels(pixels,width,height,$('routeClearance').value,state.cropRect);
  }
  function nearestPlanningPoint(point,grid,maxDistancePx=40){
    const {cellSize,cols,rows,free}=grid,limit=maxDistancePx*maxDistancePx;
    const col0=Math.max(0,Math.ceil((point.x-maxDistancePx)/cellSize-.5)),col1=Math.min(cols-1,Math.floor((point.x+maxDistancePx)/cellSize-.5));
    const row0=Math.max(0,Math.ceil((point.y-maxDistancePx)/cellSize-.5)),row1=Math.min(rows-1,Math.floor((point.y+maxDistancePx)/cellSize-.5));
    const clickedCol=Math.floor(point.x/cellSize),clickedRow=Math.floor(point.y/cellSize);
    const clickedFree=clickedCol>=0&&clickedCol<cols&&clickedRow>=0&&clickedRow<rows&&free[clickedRow*cols+clickedCol];
    let best=null,bestDistance=Infinity;
    for(let row=row0;row<=row1;row++)for(let col=col0;col<=col1;col++){
      if(!free[row*cols+col])continue;
      const x=col*cellSize+cellSize/2,y=row*cellSize+cellSize/2,distance=(x-point.x)**2+(y-point.y)**2;
      if(distance<=limit&&distance<bestDistance){best={row,col,x,y,snapped:!clickedFree};bestDistance=distance;}
    }
    return best;
  }
  function normalizedSpecialPoints(points){
    if(points===undefined)return [];
    if(!Array.isArray(points))throw new Error('特殊点标签必须是列表');
    const ids=new Set();
    return points.map(point=>{
      if(!point||point.type!=='buildingEntrance'||typeof point.label!=='string'||!point.label.trim()||!Number.isFinite(point.x)||!Number.isFinite(point.y)||!inBounds(point))throw new Error('楼栋入口标签或坐标无效');
      let id=typeof point.id==='string'&&/^[\w-]+$/.test(point.id)?point.id:uid('entrance');
      if(ids.has(id))id=uid('entrance');ids.add(id);
      return {id,type:'buildingEntrance',label:point.label.trim(),x:point.x,y:point.y,drawOrder:Number(point.drawOrder)||0};
    });
  }
  function entranceRoadPoint(point,maxDistance=12){
    if(!Number.isFinite(point.x)||!Number.isFinite(point.y)||!inBounds(point))return null;
    const limit=state.cropPreview&&state.cropRect?state.cropRect:{x:0,y:0,width:state.config.width,height:state.config.height};
    if(point.x<limit.x||point.y<limit.y||point.x>=limit.x+limit.width||point.y>=limit.y+limit.height)return null;
    const rect=boundedRect({x:point.x-maxDistance-1,y:point.y-maxDistance-1,width:maxDistance*2+3,height:maxDistance*2+3},limit);
    const pixels=composedMap('semantic').getContext('2d',{willReadFrequently:true}).getImageData(rect.x,rect.y,rect.width,rect.height).data;
    let best=null,distance=maxDistance*maxDistance;
    for(let y=0;y<rect.height;y++)for(let x=0;x<rect.width;x++){
      const i=(y*rect.width+x)*4;if(pixels[i]!==127||!pixels[i+3])continue;
      const px=rect.x+x,py=rect.y+y;
      if(px===Math.floor(point.x)&&py===Math.floor(point.y))return {x:px,y:py};
      const d=(px+.5-point.x)**2+(py+.5-point.y)**2;
      if(d<=distance){best={x:px,y:py};distance=d;}
    }
    return best;
  }
  function hitEntrance(point){
    for(const item of [...state.annotations.specialPoints].reverse()){
      if(state.cropPreview&&state.cropRect){const c=state.cropRect;if(item.x<c.x||item.y<c.y||item.x>=c.x+c.width||item.y>=c.y+c.height)continue;}
      if(Math.hypot(point.x-item.x,point.y-item.y)<=11/state.zoom)return {type:'entrance',id:item.id};
    }
    return null;
  }
  function updateEntranceUI(){
    const item=state.selected?.type==='entrance'?selectedItem():null,point=state.entranceDraft||item,input=$('entranceLabel');
    $('entrancePanel').hidden=state.showOriginal||(!point&&state.tool!=='entrance');
    const binding=state.entranceDraft?'draft':item?.id||'';
    if(input.dataset.binding!==binding){input.value=item?.label||'';input.dataset.binding=binding;}
    input.disabled=$('saveEntranceBtn').disabled=state.showOriginal||!point;
    $('saveEntranceBtn').textContent=item?'保存修改':'保存入口标签';
    $('cancelEntranceBtn').hidden=!state.entranceDraft;
    $('deleteEntranceBtn').hidden=!item;
    $('entrancePosition').textContent=point?`${item?'已保存入口':'待保存入口'} · X=${point.x}，Y=${point.y}`:'请先在道路线上点击入口位置。';
  }
  function handleEntranceClick(point){
    if(state.showOriginal)return;
    const hit=hitEntrance(point);
    if(hit){state.entranceDraft=null;state.selected=hit;updateUI();$('entranceLabel').focus();return;}
    const position=entranceRoadPoint(point);
    if(!position){toast('附近 12 px 内没有道路，请在道路线上选择入口');return;}
    state.entranceDraft=position;state.selected=null;updateUI();$('entranceLabel').focus();
  }
  function saveEntrance(){
    if(state.showOriginal)return;
    const item=state.selected?.type==='entrance'?selectedItem():null,label=$('entranceLabel').value.trim();
    if(!state.entranceDraft&&!item)return;
    if(!label){toast('请输入楼栋入口标签');$('entranceLabel').focus();return;}
    if(item){
      if(item.label!==label){pushUndo(false);item.label=label;}
    }else{
      const point=entranceRoadPoint(state.entranceDraft,0);
      if(!point){toast('该位置已不在道路上，请重新选择入口');return;}
      pushUndo(false);
      const entry={id:uid('entrance'),type:'buildingEntrance',label,...point,drawOrder:nextOrder()};
      state.annotations.specialPoints.push(entry);state.selected={type:'entrance',id:entry.id};state.entranceDraft=null;
    }
    $('entranceLabel').value=label;updateUI();toast('入口标签已保存；导出时会保存到本地工程和 JSON');
  }
  function moveEntrance(point){
    const item=selectedItem();if(state.showOriginal||state.selected?.type!=='entrance'||!item||state.drag?.kind!=='entrance')return;
    const position=entranceRoadPoint(point);state.drag.rejected=!position;
    if(!position||position.x===item.x&&position.y===item.y)return;
    if(!state.drag.changed){pushUndo(false);state.drag.changed=true;}
    item.x=position.x;item.y=position.y;updateEntranceUI();render();
  }
  function drawEntrances(target){
    const points=state.annotations.specialPoints.map(item=>({item,draft:false}));
    if(state.entranceDraft)points.push({item:{...state.entranceDraft,label:'待保存'},draft:true});
    for(const {item,draft} of points){
      if(state.cropPreview&&state.cropRect){const c=state.cropRect;if(item.x<c.x||item.y<c.y||item.x>=c.x+c.width||item.y>=c.y+c.height)continue;}
      const selected=state.selected?.type==='entrance'&&state.selected.id===item.id;
      target.save();target.translate(item.x,item.y);target.rotate(-state.viewRotation*Math.PI/180);target.scale(1/state.zoom,1/state.zoom);
      target.fillStyle=draft?'#b77900':'#7c3aed';target.strokeStyle=selected?'#00d9ff':'#fff';target.lineWidth=selected?3:2;
      target.beginPath();target.arc(0,0,selected?9:7,0,Math.PI*2);target.fill();target.stroke();
      target.font='13px sans-serif';target.textAlign='left';target.textBaseline='middle';
      const chars=Array.from(item.label),label=chars.length>24?chars.slice(0,24).join('')+'…':item.label,width=target.measureText(label).width;
      target.fillStyle='#241139';target.fillRect(13,-12,width+12,24);target.fillStyle='#fff';target.fillText(label,19,0);
      target.restore();
    }
  }
  function planAStar(grid,start,goal){if(!start||!goal||![start,goal].every(p=>Number.isInteger(p.row)&&Number.isInteger(p.col)&&p.row>=0&&p.row<grid.rows&&p.col>=0&&p.col<grid.cols&&grid.free[p.row*grid.cols+p.col]))return [];const total=grid.rows*grid.cols,startId=start.row*grid.cols+start.col,goalId=goal.row*grid.cols+goal.col,gScore=new Float64Array(total),cameFrom=new Int32Array(total),closed=new Uint8Array(total);gScore.fill(Infinity);cameFrom.fill(-1);gScore[startId]=0;const heapIds=[],heapScores=[];const heapPush=(id,score)=>{let index=heapIds.length;heapIds.push(id);heapScores.push(score);while(index>0){const parent=(index-1)>>1;if(heapScores[parent]<=score)break;heapIds[index]=heapIds[parent];heapScores[index]=heapScores[parent];index=parent;}heapIds[index]=id;heapScores[index]=score;};const heapPop=()=>{if(!heapIds.length)return -1;const root=heapIds[0],lastId=heapIds.pop(),lastScore=heapScores.pop();if(heapIds.length){let index=0;while(true){const left=index*2+1,right=left+1;if(left>=heapIds.length)break;const child=right<heapIds.length&&heapScores[right]<heapScores[left]?right:left;if(heapScores[child]>=lastScore)break;heapIds[index]=heapIds[child];heapScores[index]=heapScores[child];index=child;}heapIds[index]=lastId;heapScores[index]=lastScore;}return root;};const heuristic=(row,col)=>{const dx=Math.abs(col-goal.col),dy=Math.abs(row-goal.row);return Math.max(dx,dy)+(Math.SQRT2-1)*Math.min(dx,dy);};heapPush(startId,heuristic(start.row,start.col));const directions=[[-1,0,1],[1,0,1],[0,-1,1],[0,1,1],[-1,-1,Math.SQRT2],[-1,1,Math.SQRT2],[1,-1,Math.SQRT2],[1,1,Math.SQRT2]];while(heapIds.length){const current=heapPop();if(current<0||closed[current])continue;if(current===goalId){const ids=[];let cursor=current;while(cursor>=0){ids.push(cursor);if(cursor===startId)break;cursor=cameFrom[cursor];}if(ids.at(-1)!==startId)return [];return ids.reverse().map(id=>{const row=Math.floor(id/grid.cols),col=id%grid.cols;return {x:Math.min(grid.width-1,col*grid.cellSize+grid.cellSize/2),y:Math.min(grid.height-1,row*grid.cellSize+grid.cellSize/2)};});}closed[current]=1;const row=Math.floor(current/grid.cols),col=current%grid.cols;for(const [dr,dc,cost] of directions){const nextRow=row+dr,nextCol=col+dc;if(nextRow<0||nextRow>=grid.rows||nextCol<0||nextCol>=grid.cols)continue;const next=nextRow*grid.cols+nextCol;if(!grid.free[next]||closed[next])continue;if(dr&&dc&&(!grid.free[row*grid.cols+nextCol]||!grid.free[nextRow*grid.cols+col]))continue;const tentative=gScore[current]+cost;if(tentative>=gScore[next])continue;cameFrom[next]=current;gScore[next]=tentative;heapPush(next,tentative+heuristic(nextRow,nextCol));}}return [];}
  function planningCell(grid,point){return {row:clamp(Math.floor(point.y/grid.cellSize),0,grid.rows-1),col:clamp(Math.floor(point.x/grid.cellSize),0,grid.cols-1)};}
  function routeSegmentClear(grid,a,b){
    let {row,col}=planningCell(grid,a);const end=planningCell(grid,b),isFree=(checkRow,checkCol)=>checkRow>=0&&checkRow<grid.rows&&checkCol>=0&&checkCol<grid.cols&&Boolean(grid.free[checkRow*grid.cols+checkCol]);
    if(!isFree(row,col)||!isFree(end.row,end.col))return false;
    const deltaCol=end.col-col,deltaRow=end.row-row,stepCol=Math.sign(deltaCol),stepRow=Math.sign(deltaRow),absCol=Math.abs(deltaCol),absRow=Math.abs(deltaRow);
    let maxCol=absCol?.5/absCol:Infinity,maxRow=absRow?.5/absRow:Infinity;const stepMaxCol=absCol?1/absCol:Infinity,stepMaxRow=absRow?1/absRow:Infinity,epsilon=1e-12;
    while(row!==end.row||col!==end.col){
      if(maxCol+epsilon<maxRow){col+=stepCol;maxCol+=stepMaxCol;}
      else if(maxRow+epsilon<maxCol){row+=stepRow;maxRow+=stepMaxRow;}
      else{
        const nextCol=col+stepCol,nextRow=row+stepRow;
        if(!isFree(row,nextCol)||!isFree(nextRow,col))return false;
        col=nextCol;row=nextRow;maxCol+=stepMaxCol;maxRow+=stepMaxRow;
      }
      if(!isFree(row,col))return false;
    }
    return true;
  }
  function simplifyRoute(grid,path){
    if(path.length<3)return path.slice();
    const simplified=[path[0]];let anchor=0;
    while(anchor<path.length-1){
      let low=anchor+1,high=path.length-1,best=anchor+1;
      while(low<=high){
        const candidate=(low+high)>>1;
        if(routeSegmentClear(grid,path[anchor],path[candidate])){best=candidate;low=candidate+1;}
        else high=candidate-1;
      }
      simplified.push(path[best]);anchor=best;
    }
    return simplified;
  }
  const planGridRoute=planAStar;
  planAStar=function(grid,start,goal){
    const rawPath=planGridRoute(grid,start,goal),displayPath=simplifyRoute(grid,rawPath),status=$('routeStatus');
    if(status){status.dataset.rawPoints=String(rawPath.length);status.dataset.displayPoints=String(displayPath.length);}
    return displayPath;
  };
  function routeDistance(path){let total=0;for(let i=1;i<path.length;i++)total+=Math.hypot(path[i].x-path[i-1].x,path[i].y-path[i-1].y);return total;}
  function handleRouteClick(point){
    const grid=buildPlanningGrid();
    if(!grid.usableRoadPixels){
      clearRoute(false);
      state.route.status=grid.roadPixels?'安全边距内没有可通行道路，请减小障碍安全边距。':'当前地图或裁剪范围内没有道路，请先画道路中心线。';
      updatePlannerUI();toast(state.route.status);return;
    }
    if(!state.route.startClick||state.route.goalClick){
      const start=nearestPlanningPoint(point,grid);
      if(!start){state.route.status='起点附近 40 px 内没有可通行道路，请靠近道路重新选择。';updatePlannerUI();toast('起点附近没有道路');return;}
      state.route={startClick:{x:point.x,y:point.y},start,goalClick:null,goal:null,path:[],status:`起点已设置 (${Math.round(start.x)}, ${Math.round(start.y)})${start.snapped?'，已吸附到道路':''}，请点击终点。`};
      updatePlannerUI();render();return;
    }
    const start=nearestPlanningPoint(state.route.startClick,grid),goal=nearestPlanningPoint(point,grid);
    if(!start||!goal){
      state.route.goalClick=null;state.route.goal=null;state.route.path=[];
      state.route.status='终点附近 40 px 内没有可通行道路，请靠近道路重新选择终点。';
      updatePlannerUI();toast('终点附近没有道路');render();return;
    }
    state.route.start=start;state.route.goalClick={x:point.x,y:point.y};state.route.goal=goal;
    const path=planAStar(grid,start,goal);state.route.path=path;
    if(path.length){
      const snapped=start.snapped||goal.snapped?'；起点或终点已吸附到最近道路':'';
      state.route.status=`规划成功：两点之间道路连通；沿道路约 ${Math.round(routeDistance(path))} px，障碍安全边距 ${grid.clearance} px${snapped}`;
      toast('规划成功：两点之间道路连通');
    }else{
      state.route.status=`规划失败：两点之间道路不连通；请检查道路断口、删除范围或黑色边界，障碍安全边距 ${grid.clearance} px。`;
      toast('规划失败：道路不连通');
    }
    updatePlannerUI();render();
  }
  function drawRoute(target){const route=state.route,path=route.path;if(path.length){target.save();target.lineCap='round';target.lineJoin='round';target.strokeStyle='#00131a';target.lineWidth=10/state.zoom;target.beginPath();target.moveTo(path[0].x,path[0].y);for(let i=1;i<path.length;i++)target.lineTo(path[i].x,path[i].y);target.stroke();target.strokeStyle='#00d9ff';target.lineWidth=5/state.zoom;target.stroke();let travelled=0,nextArrow=90/state.zoom;for(let i=1;i<path.length;i++){const a=path[i-1],b=path[i],length=Math.hypot(b.x-a.x,b.y-a.y);while(length&&travelled+length>=nextArrow){const ratio=(nextArrow-travelled)/length,x=a.x+(b.x-a.x)*ratio,y=a.y+(b.y-a.y)*ratio,angle=Math.atan2(b.y-a.y,b.x-a.x),size=9/state.zoom;target.fillStyle='#00d9ff';target.beginPath();target.moveTo(x+Math.cos(angle)*size,y+Math.sin(angle)*size);target.lineTo(x+Math.cos(angle+2.55)*size,y+Math.sin(angle+2.55)*size);target.lineTo(x+Math.cos(angle-2.55)*size,y+Math.sin(angle-2.55)*size);target.closePath();target.fill();nextArrow+=90/state.zoom;}travelled+=length;}target.restore();}const marker=(point,color,label)=>{if(!point)return;target.save();target.fillStyle=color;target.strokeStyle='#fff';target.lineWidth=3/state.zoom;target.beginPath();target.arc(point.x,point.y,10/state.zoom,0,Math.PI*2);target.fill();target.stroke();target.fillStyle='#fff';target.font=`bold ${12/state.zoom}px sans-serif`;target.textAlign='center';target.textBaseline='middle';target.fillText(label,point.x,point.y);target.restore();};marker(route.start,'#14a85b','起');marker(route.goal,'#e5484d','终');}
  function drawGapHints(target){target.save();target.strokeStyle='#ff9f1c';target.fillStyle='#ff9f1c';target.lineWidth=3/state.zoom;target.font=`bold ${12/state.zoom}px sans-serif`;target.textAlign='center';target.textBaseline='middle';for(let index=0;index<state.gapHints.length;index++){const hint=state.gapHints[index],radius=15/state.zoom,cross=8/state.zoom;target.beginPath();target.arc(hint.x,hint.y,radius,0,Math.PI*2);target.stroke();target.beginPath();target.moveTo(hint.x-cross,hint.y);target.lineTo(hint.x+cross,hint.y);target.moveTo(hint.x,hint.y-cross);target.lineTo(hint.x,hint.y+cross);target.stroke();target.fillText(String(index+1),hint.x,hint.y-radius-8/state.zoom);}target.restore();}
  function drawScene(target, offsetX=0, offsetY=0, includeGuides=true, mapMode='display'){
    target.save();target.translate(-offsetX,-offsetY);target.imageSmoothingEnabled=false;
    if(includeGuides&&state.cropPreview&&state.cropRect){const c=state.cropRect;target.beginPath();target.rect(c.x,c.y,c.width,c.height);target.clip();}
    target.drawImage(state.showOriginal?state.original:composedMap(mapMode),0,0);
    if(state.showOriginal){target.restore();return;}
    if(includeGuides){
      if(state.fillPreview)paintObstacleFill(target,state.fillPreview,'#ef4444',.58);
      if(state.roadSelection){const r=state.roadSelection;paintObstacleFill(target,r,'#00a8ff',.8);target.save();target.strokeStyle='#00a8ff';target.lineWidth=2/state.zoom;target.setLineDash([8/state.zoom,5/state.zoom]);target.strokeRect(r.x,r.y,r.width,r.height);target.restore();}
      if(state.gapHints.length)drawGapHints(target);
      if(state.cropRect&&!state.cropPreview){const c=state.cropRect;target.save();target.strokeStyle='#f4c04a';target.lineWidth=2/state.zoom;target.setLineDash([10/state.zoom,7/state.zoom]);target.strokeRect(c.x,c.y,c.width,c.height);target.restore();}
      if(state.draft&&state.draft.kind==='rect'){const d=normalizedRect(state.draft.start,state.draft.end);target.save();target.fillStyle=state.draft.type==='free'?'#ffffff99':state.draft.type==='outline'?'#00000000':state.draft.type==='roadSelect'?'#00a8ff22':'#f4c04a22';target.strokeStyle=state.draft.type==='crop'?'#f4c04a':'#78a9ff';target.lineWidth=2/state.zoom;target.setLineDash([8/state.zoom,5/state.zoom]);target.fillRect(d.x,d.y,d.width,d.height);target.strokeRect(d.x,d.y,d.width,d.height);target.restore();}
      if(state.lineStart){target.save();if(state.linePreview){target.strokeStyle='#78a9ff';target.lineWidth=Math.max(1,Number($('roadWidth').value)||5);target.lineCap='round';target.setLineDash([8/state.zoom,5/state.zoom]);target.beginPath();target.moveTo(state.lineStart.x,state.lineStart.y);target.lineTo(state.linePreview.x,state.linePreview.y);target.stroke();}target.fillStyle='#78a9ff';target.setLineDash([]);target.beginPath();target.arc(state.lineStart.x,state.lineStart.y,6/state.zoom,0,Math.PI*2);target.fill();if(state.linePreview){target.beginPath();target.arc(state.linePreview.x,state.linePreview.y,4/state.zoom,0,Math.PI*2);target.fill();}target.restore();}
      if(state.tool==='pasteBuilding'&&state.pastePreview&&state.buildingTemplate)paintBuilding(target,{...state.buildingTemplate,x:state.pastePreview.x,y:state.pastePreview.y},true);
      drawRoute(target);drawEntrances(target);drawSelection(target);
    }
    target.restore();
  }
  function drawSelection(target){const item=selectedItem();if(!item||state.selected.type==='entrance')return;target.save();target.strokeStyle='#2f80ff';target.fillStyle='#fff';target.lineWidth=2/state.zoom;if(isLineType(state.selected.type)){target.beginPath();target.moveTo(item.x1,item.y1);target.lineTo(item.x2,item.y2);target.stroke();for(const p of [{x:item.x1,y:item.y1},{x:item.x2,y:item.y2}]){target.beginPath();target.arc(p.x,p.y,6/state.zoom,0,Math.PI*2);target.fill();target.stroke();}}else if(state.selected.type==='building'){const points=buildingWorldPoints(item);target.beginPath();target.moveTo(points[0].x,points[0].y);for(let i=1;i<points.length;i++)target.lineTo(points[i].x,points[i].y);target.closePath();target.stroke();const b=buildingBounds(item);target.beginPath();target.arc(b.x+b.width/2,b.y-14/state.zoom,6/state.zoom,0,Math.PI*2);target.fill();target.stroke();}else{target.strokeRect(item.x,item.y,item.width,item.height);}target.restore();}
  function applyViewTransform(target,dpr){const z=dpr*state.zoom,px=dpr*state.panX,py=dpr*state.panY,w=state.config.width,h=state.config.height,r=state.viewRotation;if(r===90)target.setTransform(0,z,-z,0,px+z*h,py);else if(r===180)target.setTransform(-z,0,0,-z,px+z*w,py+z*h);else if(r===270)target.setTransform(0,-z,z,0,px,py+z*w);else target.setTransform(z,0,0,z,px,py);}
  function render(){if(!state.config||!state.base)return;const dpr=window.devicePixelRatio||1;ctx.setTransform(1,0,0,1,0,0);ctx.clearRect(0,0,canvas.width,canvas.height);ctx.fillStyle='#090d16';ctx.fillRect(0,0,canvas.width,canvas.height);applyViewTransform(ctx,dpr);drawScene(ctx);ctx.setTransform(1,0,0,1,0,0);}
  function normalizedRect(a,b){const x=Math.floor(Math.min(a.x,b.x)),y=Math.floor(Math.min(a.y,b.y)),right=Math.ceil(Math.max(a.x,b.x)),bottom=Math.ceil(Math.max(a.y,b.y));return {x,y,width:Math.max(1,right-x),height:Math.max(1,bottom-y)};}
  function inBounds(p){return p.x>=0&&p.y>=0&&p.x<state.config.width&&p.y<state.config.height;}
  function pointLineDistance(p,a,b){const dx=b.x-a.x,dy=b.y-a.y;if(!dx&&!dy)return Math.hypot(p.x-a.x,p.y-a.y);const t=clamp(((p.x-a.x)*dx+(p.y-a.y)*dy)/(dx*dx+dy*dy),0,1);return Math.hypot(p.x-(a.x+t*dx),p.y-(a.y+t*dy));}
  function pointInPolygon(point,points){let inside=false;for(let i=0,j=points.length-1;i<points.length;j=i++){const a=points[i],b=points[j];if(((a.y>point.y)!==(b.y>point.y))&&(point.x<(b.x-a.x)*(point.y-a.y)/(b.y-a.y)+a.x))inside=!inside;}return inside;}
  function hitTest(p){const entrance=hitEntrance(p);if(entrance)return entrance;const layers=allItems().filter(layer=>layer.type!=='entrance').reverse();const tol=9/state.zoom;for(const layer of layers){const i=layer.item;if(isLineType(layer.type)){if(pointLineDistance(p,{x:i.x1,y:i.y1},{x:i.x2,y:i.y2})<=Math.max(tol,i.widthPx/2+tol/2))return {type:layer.type,id:i.id};}else if(layer.type==='building'){if(pointInPolygon(p,buildingWorldPoints(i)))return {type:'building',id:i.id};}else if(layer.type==='fill'){if(pointInObstacleFill(p,i))return {type:'fill',id:i.id};}else if(p.x>=i.x&&p.x<=i.x+i.width&&p.y>=i.y&&p.y<=i.y+i.height)return {type:layer.type,id:i.id};}return null;}
  function hitLineHandle(p){const item=selectedItem();if(!item||!isLineType(state.selected.type))return null;const tol=11/state.zoom;if(Math.hypot(p.x-item.x1,p.y-item.y1)<=tol)return 'start';if(Math.hypot(p.x-item.x2,p.y-item.y2)<=tol)return 'end';return null;}
  function convexHull(points){const unique=[...new Map(points.map(p=>[p.x+','+p.y,p])).values()].sort((a,b)=>a.x-b.x||a.y-b.y);if(unique.length<=3)return unique;const cross=(o,a,b)=>(a.x-o.x)*(b.y-o.y)-(a.y-o.y)*(b.x-o.x),lower=[],upper=[];for(const p of unique){while(lower.length>=2&&cross(lower.at(-2),lower.at(-1),p)<=0)lower.pop();lower.push(p);}for(let i=unique.length-1;i>=0;i--){const p=unique[i];while(upper.length>=2&&cross(upper.at(-2),upper.at(-1),p)<=0)upper.pop();upper.push(p);}lower.pop();upper.pop();return lower.concat(upper);}
  function captureBuildingChildren(hull,cx,cy){const children=[];for(const layer of allItems()){if(!['free','obstacle','line','road'].includes(layer.type))continue;const item=layer.item;if(isLineType(layer.type)){const a={x:item.x1,y:item.y1},b={x:item.x2,y:item.y2};if(pointInPolygon(a,hull)&&pointInPolygon(b,hull))children.push({type:layer.type,name:item.name,x1:item.x1-cx,y1:item.y1-cy,x2:item.x2-cx,y2:item.y2-cy,widthPx:item.widthPx,drawOrder:item.drawOrder||0});}else{const center={x:item.x+item.width/2,y:item.y+item.height/2};if(pointInPolygon(center,hull))children.push({type:layer.type,name:item.name,x:item.x-cx,y:item.y-cy,width:item.width,height:item.height,drawOrder:item.drawOrder||0});}}return children.sort((a,b)=>(a.drawOrder||0)-(b.drawOrder||0));}
  function extractBuildingTemplate(rect){const x=clamp(Math.floor(rect.x),0,state.config.width-1),y=clamp(Math.floor(rect.y),0,state.config.height-1),width=clamp(Math.ceil(rect.width),1,state.config.width-x),height=clamp(Math.ceil(rect.height),1,state.config.height-y),cx=x+width/2,cy=y+height/2,hull=[{x,y},{x:x+width,y},{x:x+width,y:y+height},{x,y:y+height}],children=captureBuildingChildren(hull,cx,cy),patch=document.createElement('canvas');patch.width=width;patch.height=height;const patchContext=patch.getContext('2d'),old=state.showOriginal;try{state.showOriginal=false;drawScene(patchContext,x,y,false,'raw');}finally{state.showOriginal=old;}forceMapPalette(patchContext,width,height);const patchData=patch.toDataURL('image/png');cachePatch(patchData,patch);return {points:[{x:-width/2,y:-height/2},{x:width/2,y:-height/2},{x:width/2,y:height/2},{x:-width/2,y:height/2}],templateWidth:width,templateHeight:height,scaleX:1,scaleY:1,rotation:0,lineWidth:0,children,patchData,x:cx,y:cy};}
  function defaultOutlineWidth(){return clamp(Math.round(Number(state.config?.defaultOutlineWidth)||5),1,100);}
  function normalizedOutlineWidth(value){return clamp(Math.round(Number(value)||defaultOutlineWidth()),1,100);}
  function updateOutlineWidthUI(){
    const item=state.selected?.type==='outline'?selectedItem():null;
    $('outlineWidthPanel').hidden=state.showOriginal||(!item&&state.tool!=='outlineWidth');
    $('outlineWidthSlider').disabled=$('matchOutlineWidthBtn').disabled=state.showOriginal||!item;
    const width=item?normalizedOutlineWidth(item.widthPx):defaultOutlineWidth();
    $('outlineWidthSlider').value=width;$('outlineWidthValue').textContent=width+' px';
    $('outlineDefaultHint').textContent=`楼栋参考线宽 ${defaultOutlineWidth()} px；新矩形默认匹配此线宽，已有矩形可用“调整矩形宽度”修改。`;
    $('outlineWidthHint').textContent=item?`${item.name}：当前边框 ${width} px，楼栋参考 ${defaultOutlineWidth()} px。支持撤销。`:'点击用“画矩形”创建的矩形，再拖动滑块调整边框线宽。';
  }
  function selectOutline(point){
    outlineWidthEditingId=null;
    const item=state.annotations.outlineRects.slice().sort((a,b)=>(b.drawOrder||0)-(a.drawOrder||0)).find(r=>point.x>=r.x&&point.x<=r.x+r.width&&point.y>=r.y&&point.y<=r.y+r.height);
    state.selected=item?{type:'outline',id:item.id}:null;updateUI();
    if(!item)toast('请点击用“画矩形”创建的矩形');
  }
  function changeOutlineWidth(value=$('outlineWidthSlider').value){
    const item=selectedItem();if(state.showOriginal||!item||state.selected.type!=='outline')return;
    const width=normalizedOutlineWidth(value);if(width===normalizedOutlineWidth(item.widthPx))return;
    if(outlineWidthEditingId!==item.id){pushUndo();outlineWidthEditingId=item.id;}
    item.widthPx=width;updateUI();
  }
  function matchOutlineWidth(){
    outlineWidthEditingId=null;changeOutlineWidth(defaultOutlineWidth());outlineWidthEditingId=null;
  }
  function setTool(tool){state.entranceDraft=null;outlineWidthEditingId=null;state.roadSelection=null;if(tool==='pasteBuilding'&&!state.buildingTemplate){toast('请先使用“框选复制楼栋”提取一个模板');return;}if(tool==='crop'&&state.cropPreview){state.cropPreview=false;fit();toast('已返回完整地图，可重新绘制裁剪框');}if(state.showOriginal&&tool!=='pan'){state.showOriginal=false;$('toggleBaseBtn').textContent='查看原图';toast('已切回灰度图以继续编辑');}state.tool=tool;state.lineStart=null;state.linePreview=null;state.draft=null;if(tool!=='fill')state.fillPreview=null;if(tool!=='pasteBuilding')state.pastePreview=null;document.querySelectorAll('[data-tool]').forEach(b=>b.classList.toggle('active',b.dataset.tool===tool));$('modeText').textContent=labels[tool];$('toolHint').textContent=hints[tool];stage.classList.toggle('pan',tool==='pan');updateOutlineWidthUI();updateEntranceUI();updateRoadStyleUI();updateCropPreviewUI();render();}
  function addRect(type,rect){pushUndo();const list=listFor(type);list.push({id:uid(type),name:(type==='free'?'可行区 ':type==='outline'?'矩形轮廓 ':'障碍区 ')+(list.length+1),...rect,...(type==='outline'?{widthPx:normalizedOutlineWidth($('outlineWidth').value)}:{}),drawOrder:nextOrder()});state.selected={type,id:list.at(-1).id};updateUI();}
  function addLine(a,b){const type='road',end=snapForLine(a,b,type);pushUndo();const list=listFor(type);list.push({id:uid(type),name:(type==='road'?'道路中心线 ':'障碍线 ')+(list.length+1),x1:Math.round(a.x),y1:Math.round(a.y),x2:Math.round(end.x),y2:Math.round(end.y),widthPx:clamp(Number($('roadWidth').value)||5,1,100),drawOrder:nextOrder()});state.selected={type,id:list.at(-1).id};state.lineStart=null;state.linePreview=null;updateUI();}
  function placeBuilding(point){if(!state.buildingTemplate)return;pushUndo();const list=state.annotations.buildingCopies,item={...clone(state.buildingTemplate),id:uid('building'),name:'楼栋副本 '+(list.length+1),x:Math.round(point.x),y:Math.round(point.y),drawOrder:nextOrder()};list.push(item);state.selected={type:'building',id:item.id};updateUI();}
  function pointerDown(e){stage.focus();const p=imagePoint(e);const panGesture=state.tool==='pan'||state.space||e.button===1;if(panGesture){state.drag={kind:'pan',sx:e.clientX,sy:e.clientY,panX:state.panX,panY:state.panY};stage.classList.add('panning');canvas.setPointerCapture(e.pointerId);return;}if(state.showOriginal){toast('当前为原图预览，请切回灰度图后编辑');return;}if(!inBounds(p))return;if(state.tool==='entrance'){handleEntranceClick(p);return;}if(state.tool==='outlineWidth'){selectOutline(p);return;}if(state.tool==='route'){handleRouteClick(p);return;}if(state.tool==='fill'){handleObstacleFillClick(p);return;}if(['free','outline','roadSelect','crop','copyBuilding'].includes(state.tool)){state.draft={kind:'rect',type:state.tool,start:p,end:p};canvas.setPointerCapture(e.pointerId);render();return;}if(state.tool==='pasteBuilding'){placeBuilding(p);state.pastePreview=p;render();return;}if(state.tool==='road'){if(!state.lineStart){state.lineStart=p;state.linePreview=p;toast('已设置起点，请点击终点');}else if(Math.hypot(p.x-state.lineStart.x,p.y-state.lineStart.y)>1)addLine(state.lineStart,p);render();return;}if(state.tool==='select'){const entrance=hitEntrance(p),handle=entrance?null:hitLineHandle(p);const hit=entrance||(handle?state.selected:hitTest(p));if(!hit){state.selected=null;updateUI();return;}state.selected=hit;if(hit.type==='entrance'){state.drag={kind:'entrance',start:p,original:clone(selectedItem()),changed:false};canvas.setPointerCapture(e.pointerId);updateUI();return;}if(hit.type==='fill'){updateUI();return;}const item=selectedItem();pushUndo();state.drag={kind:handle?'line-handle':'move',handle,start:p,original:clone(item)};canvas.setPointerCapture(e.pointerId);updateUI();}}
  function pointerMove(e){const p=imagePoint(e);if(state.tool==='pasteBuilding'&&!state.drag&&!state.draft){state.pastePreview=inBounds(p)?p:null;render();return;}if(state.tool==='road'&&state.lineStart&&!state.drag&&!state.draft){const bounded={x:clamp(p.x,0,state.config.width-1),y:clamp(p.y,0,state.config.height-1)};state.linePreview=snapForLine(state.lineStart,bounded);render();return;}if(!state.drag&&!state.draft)return;if(state.drag?.kind==='pan'){state.panX=state.drag.panX+(e.clientX-state.drag.sx);state.panY=state.drag.panY+(e.clientY-state.drag.sy);render();return;}if(state.draft){state.draft.end={x:clamp(p.x,0,state.config.width),y:clamp(p.y,0,state.config.height)};render();return;}const item=selectedItem();if(!item)return;const dx=p.x-state.drag.start.x,dy=p.y-state.drag.start.y;if(state.drag.kind==='entrance'){moveEntrance({x:state.drag.original.x+dx,y:state.drag.original.y+dy});return;}if(state.drag.kind==='line-handle'){if(state.drag.handle==='start'){const fixed={x:state.drag.original.x2,y:state.drag.original.y2},moving=snapForLine(fixed,{x:state.drag.original.x1+dx,y:state.drag.original.y1+dy},state.selected.type);item.x1=clamp(Math.round(moving.x),0,state.config.width-1);item.y1=clamp(Math.round(moving.y),0,state.config.height-1);}else{const fixed={x:state.drag.original.x1,y:state.drag.original.y1},moving=snapForLine(fixed,{x:state.drag.original.x2+dx,y:state.drag.original.y2+dy},state.selected.type);item.x2=clamp(Math.round(moving.x),0,state.config.width-1);item.y2=clamp(Math.round(moving.y),0,state.config.height-1);}}else if(isLineType(state.selected.type)){item.x1=clamp(Math.round(state.drag.original.x1+dx),0,state.config.width-1);item.y1=clamp(Math.round(state.drag.original.y1+dy),0,state.config.height-1);item.x2=clamp(Math.round(state.drag.original.x2+dx),0,state.config.width-1);item.y2=clamp(Math.round(state.drag.original.y2+dy),0,state.config.height-1);}else if(state.selected.type==='building'){item.x=clamp(Math.round(state.drag.original.x+dx),0,state.config.width-1);item.y=clamp(Math.round(state.drag.original.y+dy),0,state.config.height-1);}else{item.x=clamp(Math.round(state.drag.original.x+dx),0,state.config.width-item.width);item.y=clamp(Math.round(state.drag.original.y+dy),0,state.config.height-item.height);}invalidateMap();updateSelectionFields();render();}
  function pointerUp(e){if(state.drag?.kind==='pan'){state.drag=null;stage.classList.remove('panning');return;}if(state.draft){const d=normalizedRect(state.draft.start,state.draft.end),type=state.draft.type;state.draft=null;if(d.width<3||d.height<3){toast('区域太小，未保存');render();return;}if(type==='crop'){pushUndo();state.cropRect=d;state.cropPreview=false;state.selected=null;toast('已设置裁剪保留框；导出后将自动显示裁剪结果');updateUI();}else if(type==='roadSelect'){selectRoads(d);}else if(type==='copyBuilding'){try{state.buildingTemplate=extractBuildingTemplate(d);state.pastePreview={x:state.buildingTemplate.x,y:state.buildingTemplate.y};setTool('pasteBuilding');toast(`已原样复制所选 ${Math.round(d.width)}×${Math.round(d.height)}px 区域`);}catch(err){toast(err.message);render();}}else addRect(type,d);return;}if(state.drag?.kind==='entrance'&&state.drag.rejected)toast('入口必须位于道路上，已保留最后有效位置');state.drag=null;updateUI();}
  function deleteSelected(){if(state.roadSelection){deleteRoads();return;}if(state.showOriginal){toast('当前为原图预览，请切回灰度图后编辑');return;}const item=selectedItem();if(!item)return;pushUndo(state.selected.type!=='entrance');const list=listFor(state.selected.type),index=list.findIndex(x=>x.id===item.id);if(index>=0)list.splice(index,1);state.selected=null;updateUI();render();}
  function updateSelectionFields(){updateOutlineWidthUI();updateEntranceUI();const item=selectedItem(),isEntrance=state.selected?.type==='entrance',has=!!item&&!isEntrance;$('selectionEmpty').textContent=isEntrance?'请在上方“楼栋入口标签”中编辑。':'尚未选择标注';$('selectionEmpty').hidden=has;$('selectionFields').hidden=!has;if(!has)return;$('selectedName').value=item.name||'';$('outlineFields').hidden=state.selected.type!=='outline';if(state.selected.type==='outline')$('selectedOutlineWidth').value=normalizedOutlineWidth(item.widthPx);const isLine=isLineType(state.selected.type),isBuilding=state.selected.type==='building',isFill=state.selected.type==='fill';$('rectFields').hidden=isLine||isBuilding||isFill;$('lineFields').hidden=!isLine;$('buildingFields').hidden=!isBuilding;$('fillFields').hidden=!isFill;if(isLine){$('lineX1').value=Math.round(item.x1);$('lineY1').value=Math.round(item.y1);$('lineX2').value=Math.round(item.x2);$('lineY2').value=Math.round(item.y2);$('selectedLineWidth').value=item.widthPx;}else if(isBuilding){$('buildingX').value=Math.round(item.x);$('buildingY').value=Math.round(item.y);$('buildingW').value=Math.round(item.templateWidth*(item.scaleX||1));$('buildingH').value=Math.round(item.templateHeight*(item.scaleY||1));$('buildingChildCount').textContent=`复制范围 ${Math.round(item.templateWidth)}×${Math.round(item.templateHeight)}px，框内内容会整体移动和缩放。`;}else if(isFill){$('fillInfo').textContent=`封闭区域 ${item.area} px；范围 ${item.width}×${item.height}px。填充区不可移动，可删除后重新填充。`;}else{$('rectX').value=Math.round(item.x);$('rectY').value=Math.round(item.y);$('rectW').value=Math.round(item.width);$('rectH').value=Math.round(item.height);}}
  function renderList(){const items=allItems(),el=$('layerList');if(!items.length){el.innerHTML='<div class="empty">还没有人工标注</div>';return;}el.innerHTML=items.map(({type,item})=>{const selected=state.selected?.type===type&&state.selected?.id===item.id;const meta=type==='entrance'?`楼栋入口 · (${item.x}, ${item.y})`:isLineType(type)?`(${Math.round(item.x1)}, ${Math.round(item.y1)}) → (${Math.round(item.x2)}, ${Math.round(item.y2)}) · ${item.widthPx}px`:type==='building'?`中心 (${Math.round(item.x)}, ${Math.round(item.y)}) · ${Math.round(item.templateWidth*(item.scaleX||1))}×${Math.round(item.templateHeight*(item.scaleY||1))} · ${(item.children||[]).length} 个内部标注`:type==='fill'?`封闭填充 · ${item.area} px · ${item.width}×${item.height}`:`x=${Math.round(item.x)}, y=${Math.round(item.y)} · ${Math.round(item.width)}×${Math.round(item.height)}${type==='outline'?` · 边框 ${normalizedOutlineWidth(item.widthPx)}px`:''}`;return `<div class="layer-item ${selected?'selected':''}" data-type="${type}" data-id="${item.id}"><div class="layer-head"><span>${escapeHtml(type==='entrance'?item.label:item.name)}</span><button class="mini" data-delete="1">删除</button></div><div class="layer-meta">${meta}</div></div>`;}).join('');}
  function escapeHtml(value){return String(value).replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));}
  function updateCropPreviewUI(){const button=$('toggleCropPreviewBtn');button.disabled=!state.cropRect;button.textContent=state.cropPreview?'显示完整地图':'预览裁剪结果';}
  function updateUI(){invalidateMap();updateRoadStyleUI();const count=allItems().length;$('countText').textContent=`${count} 个人工标注 · ${Math.round(state.zoom*100)}% · 视图 ${state.viewRotation}°`;$('cropStatus').textContent=state.cropRect?(state.cropPreview?'当前正在预览裁剪结果':'保留 x='+state.cropRect.x+', y='+state.cropRect.y+', '+state.cropRect.width+'×'+state.cropRect.height+'px'):'未设置裁剪框，将导出完整尺寸。';$('undoBtn').disabled=!state.undo.length;$('redoBtn').disabled=!state.redo.length;updateSelectionFields();updatePlannerUI();updateGapStatus();updateCropPreviewUI();renderList();render();}
  function applyFields(){if(state.selected?.type==='entrance')return;if(state.showOriginal){toast('当前为原图预览，请切回灰度图后编辑');return;}const item=selectedItem();if(!item)return;pushUndo();item.name=$('selectedName').value.trim()||item.name;if(state.selected.type==='outline')item.widthPx=normalizedOutlineWidth($('selectedOutlineWidth').value);if(state.selected.type==='fill'){updateUI();return;}if(isLineType(state.selected.type)){const x1=clamp(Number($('lineX1').value)||0,0,state.config.width-1),y1=clamp(Number($('lineY1').value)||0,0,state.config.height-1),x2=clamp(Number($('lineX2').value)||0,0,state.config.width-1),y2=clamp(Number($('lineY2').value)||0,0,state.config.height-1),coordsChanged=x1!==item.x1||y1!==item.y1||x2!==item.x2||y2!==item.y2,end=coordsChanged?snapForLine({x:x1,y:y1},{x:x2,y:y2},state.selected.type):{x:x2,y:y2};item.x1=x1;item.y1=y1;item.x2=end.x;item.y2=end.y;item.widthPx=clamp(Number($('selectedLineWidth').value)||1,1,100);}else if(state.selected.type==='building'){item.x=clamp(Number($('buildingX').value)||0,0,state.config.width-1);item.y=clamp(Number($('buildingY').value)||0,0,state.config.height-1);item.scaleX=clamp((Number($('buildingW').value)||item.templateWidth)/item.templateWidth,.05,20);item.scaleY=clamp((Number($('buildingH').value)||item.templateHeight)/item.templateHeight,.05,20);}else{item.width=clamp(Number($('rectW').value)||1,1,state.config.width);item.height=clamp(Number($('rectH').value)||1,1,state.config.height);item.x=clamp(Number($('rectX').value)||0,0,state.config.width-item.width);item.y=clamp(Number($('rectY').value)||0,0,state.config.height-item.height);}updateUI();}
  function duplicateBuilding(){const item=selectedItem();if(!item||state.selected.type!=='building')return;pushUndo();const copy={...clone(item),id:uid('building'),name:'楼栋副本 '+(state.annotations.buildingCopies.length+1),x:clamp(item.x+20,0,state.config.width-1),y:clamp(item.y+20,0,state.config.height-1),drawOrder:nextOrder()};state.annotations.buildingCopies.push(copy);state.selected={type:'building',id:copy.id};state.buildingTemplate={points:clone(copy.points),templateWidth:copy.templateWidth,templateHeight:copy.templateHeight,scaleX:copy.scaleX,scaleY:copy.scaleY,rotation:copy.rotation,lineWidth:copy.lineWidth,children:clone(copy.children||[]),patchData:copy.patchData};updateUI();}
  function clipRect(r,c){const x=Math.max(r.x,c.x),y=Math.max(r.y,c.y),right=Math.min(r.x+r.width,c.x+c.width),bottom=Math.min(r.y+r.height,c.y+c.height);if(right<=x||bottom<=y)return null;return {...r,x:x-c.x,y:y-c.y,width:right-x,height:bottom-y};}
  function clipLine(line,c){let x1=line.x1,y1=line.y1,x2=line.x2,y2=line.y2;const xmin=c.x,ymin=c.y,xmax=c.x+c.width-1,ymax=c.y+c.height-1;const code=(x,y)=>(x<xmin?1:0)|(x>xmax?2:0)|(y<ymin?4:0)|(y>ymax?8:0);let a=code(x1,y1),b=code(x2,y2);while(true){if(!(a|b))break;if(a&b)return null;const out=a||b;let x,y;if(out&8){x=x1+(x2-x1)*(ymax-y1)/(y2-y1);y=ymax;}else if(out&4){x=x1+(x2-x1)*(ymin-y1)/(y2-y1);y=ymin;}else if(out&2){y=y1+(y2-y1)*(xmax-x1)/(x2-x1);x=xmax;}else{y=y1+(y2-y1)*(xmin-x1)/(x2-x1);x=xmin;}if(out===a){x1=x;y1=y;a=code(x1,y1);}else{x2=x;y2=y;b=code(x2,y2);}}return {...line,x1:x1-c.x,y1:y1-c.y,x2:x2-c.x,y2:y2-c.y};}
  function clipFill(item,c){const runs=[];let area=0,minX=Infinity,maxX=-Infinity,minY=Infinity,maxY=-Infinity;for(const [y,x1,x2] of item.runs||[]){if(y<c.y||y>=c.y+c.height)continue;const left=Math.max(x1,c.x),right=Math.min(x2,c.x+c.width-1);if(right<left)continue;const adjustedY=y-c.y,adjustedX1=left-c.x,adjustedX2=right-c.x;runs.push([adjustedY,adjustedX1,adjustedX2]);area+=adjustedX2-adjustedX1+1;if(adjustedX1<minX)minX=adjustedX1;if(adjustedX2>maxX)maxX=adjustedX2;if(adjustedY<minY)minY=adjustedY;if(adjustedY>maxY)maxY=adjustedY;}if(!runs.length)return null;return {...clone(item),x:minX,y:minY,width:maxX-minX+1,height:maxY-minY+1,area,runs};}
  function buildingIntersectsCrop(item,c){const b=buildingBounds(item);return b.x+b.width>=c.x&&b.x<=c.x+c.width&&b.y+b.height>=c.y&&b.y<=c.y+c.height;}
  function exportedAnnotations(c){if(!c)return clone(state.annotations);return {specialPoints:state.annotations.specialPoints.filter(p=>p.x>=c.x&&p.x<c.x+c.width&&p.y>=c.y&&p.y<c.y+c.height).map(p=>({...p,x:p.x-c.x,y:p.y-c.y})),freeRects:state.annotations.freeRects.map(r=>clipRect(r,c)).filter(Boolean),obstacleRects:state.annotations.obstacleRects.map(r=>clipRect(r,c)).filter(Boolean),obstacleFills:state.annotations.obstacleFills.map(item=>clipFill(item,c)).filter(Boolean),obstacleLines:state.annotations.obstacleLines.map(r=>clipLine(r,c)).filter(Boolean),roadLines:state.annotations.roadLines.map(r=>clipLine(r,c)).filter(Boolean),roadErases:state.annotations.roadErases.map(r=>clipRect(r,c)).filter(Boolean),outlineRects:state.annotations.outlineRects.filter(r=>clipRect(r,c)).map(r=>({...r,x:r.x-c.x,y:r.y-c.y})),buildingCopies:state.annotations.buildingCopies.filter(item=>buildingIntersectsCrop(item,c)).map(item=>({...clone(item),x:item.x-c.x,y:item.y-c.y}))};}
  function download(blob,name){const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=name;document.body.appendChild(a);a.click();setTimeout(()=>{URL.revokeObjectURL(a.href);a.remove();},1000);}
  function forceMapPalette(context,width,height){const image=context.getImageData(0,0,width,height),pixels=image.data;for(let i=0;i<pixels.length;i+=4){const value=pixels[i]+pixels[i+1]+pixels[i+2]<192?0:pixels[i]+pixels[i+1]+pixels[i+2]<576?127:255;pixels[i]=value;pixels[i+1]=value;pixels[i+2]=value;pixels[i+3]=255;}context.putImageData(image,0,0);}
  async function saveLocalProject(data){const response=await fetch('/project.json',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});if(!response.ok){let message='HTTP '+response.status;try{message=(await response.json()).error||message;}catch{}throw new Error(message);}}
  async function exportAll(){try{await preloadBuildingPatches();}catch(err){alert('导出失败：'+err.message);return;}const c=state.cropRect||{x:0,y:0,width:state.config.width,height:state.config.height};const full=document.createElement('canvas');full.width=state.config.width;full.height=state.config.height;const fctx=full.getContext('2d');const old=state.showOriginal;state.showOriginal=false;drawScene(fctx,0,0,false);state.showOriginal=old;const out=document.createElement('canvas');out.width=c.width;out.height=c.height;const octx=out.getContext('2d');octx.imageSmoothingEnabled=false;octx.drawImage(full,c.x,c.y,c.width,c.height,0,0,c.width,c.height);const data={version:'gaode-binary-map-editor-v6',palette:{obstacle:0,road:state.roadStyle.color,background:255},exportedAt:new Date().toISOString(),source:{name:state.config.sourceName,width:state.config.width,height:state.config.height},preprocessing:state.config.report,project:{cropRect:clone(state.cropRect),roadStyle:clone(state.roadStyle),annotations:clone(state.annotations)},exported:{width:c.width,height:c.height,offsetX:c.x,offsetY:c.y,annotations:exportedAnnotations(c)}};try{await saveLocalProject(data);}catch(err){alert('导出前保存本地工程失败：'+err.message);return;}if(state.cropRect){state.cropPreview=true;fitCrop();updateUI();}const stem=state.config.stem;out.toBlob(blob=>{download(blob,stem+'_map.png');setTimeout(()=>download(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}),stem+'_annotations.json'),250);toast(state.cropRect?'已导出；当前显示裁剪结果':'已导出并保存本地工程');},'image/png');}
  async function importJson(file){if(state.showOriginal){toast('当前为原图预览，请切回灰度图后导入');return;}try{const data=JSON.parse(await file.text());if(data.source&&((Number(data.source.width)!==state.config.width)||(Number(data.source.height)!==state.config.height)))throw new Error('JSON 的原图尺寸与当前图片不一致');const annotations=data.project?.annotations||data.annotations;if(!annotations)throw new Error('JSON 中没有可编辑标注');const specialPoints=normalizedSpecialPoints(annotations.specialPoints);pushUndo();state.annotations={freeRects:Array.isArray(annotations.freeRects)?annotations.freeRects:[],obstacleRects:Array.isArray(annotations.obstacleRects)?annotations.obstacleRects:[],obstacleFills:Array.isArray(annotations.obstacleFills)?annotations.obstacleFills:[],obstacleLines:Array.isArray(annotations.obstacleLines)?annotations.obstacleLines:[],roadLines:Array.isArray(annotations.roadLines)?annotations.roadLines:[],outlineRects:Array.isArray(annotations.outlineRects)?annotations.outlineRects:[],roadErases:Array.isArray(annotations.roadErases)?annotations.roadErases:[],buildingCopies:Array.isArray(annotations.buildingCopies)?annotations.buildingCopies:[],specialPoints};state.roadStyle=normalizedRoadStyle(data.project?.roadStyle||data.roadStyle);state.cropRect=data.project?.cropRect||data.cropRect||null;state.cropPreview=false;state.order=Math.max(0,...allItems().map(x=>Number(x.item.drawOrder)||0));state.selected=null;state.entranceDraft=null;state.fillPreview=null;state.gapHints=[];await preloadBuildingPatches();updateUI();toast('标注 JSON 已导入');}catch(err){alert('导入失败：'+err.message);}}
  function loadImage(src){return new Promise((resolve,reject)=>{const image=new Image();image.onload=()=>resolve(image);image.onerror=()=>reject(new Error('图片加载失败'));image.src=src+'?v='+Date.now();});}
  async function init(){state.config=await fetch('/config.json').then(r=>r.json());[state.base,state.original]=await Promise.all([loadImage('/base.png'),loadImage('/original.png')]);state.baseCanvas=document.createElement('canvas');state.baseCanvas.width=state.config.width;state.baseCanvas.height=state.config.height;const bctx=state.baseCanvas.getContext('2d',{willReadFrequently:true});bctx.imageSmoothingEnabled=false;bctx.drawImage(state.base,0,0);$('sourceText').textContent=`${state.config.sourceName} · ${state.config.width}×${state.config.height}px · 楼栋轮廓 ${state.config.report.contour_count} 条 · 道路初始宽度 ${state.config.report.road_width_px}px`;$('outlineWidth').value=defaultOutlineWidth();$('roadWidth').value=state.config.defaultRoadWidth;const saved=await fetch('/project.json',{cache:'no-store'}).then(r=>r.ok?r.json():null);if(saved?.project?.annotations&&Number(saved.source?.width)===state.config.width&&Number(saved.source?.height)===state.config.height){const annotations=saved.project.annotations;state.annotations={freeRects:Array.isArray(annotations.freeRects)?annotations.freeRects:[],obstacleRects:Array.isArray(annotations.obstacleRects)?annotations.obstacleRects:[],obstacleFills:Array.isArray(annotations.obstacleFills)?annotations.obstacleFills:[],obstacleLines:Array.isArray(annotations.obstacleLines)?annotations.obstacleLines:[],roadLines:Array.isArray(annotations.roadLines)?annotations.roadLines:[],outlineRects:Array.isArray(annotations.outlineRects)?annotations.outlineRects:[],roadErases:Array.isArray(annotations.roadErases)?annotations.roadErases:[],buildingCopies:Array.isArray(annotations.buildingCopies)?annotations.buildingCopies:[],specialPoints:normalizedSpecialPoints(annotations.specialPoints)};state.roadStyle=normalizedRoadStyle(saved.project.roadStyle);state.cropRect=saved.project.cropRect||null;state.cropPreview=Boolean(state.cropRect);state.order=Math.max(0,...allItems().map(x=>Number(x.item.drawOrder)||0));await preloadBuildingPatches();toast(state.cropPreview?'已恢复上次导出的裁剪结果':'已自动恢复上次导出的工程');}resize();state.cropPreview?fitCrop():fit();updateUI();}
  document.querySelectorAll('[data-tool]').forEach(b=>b.addEventListener('click',()=>setTool(b.dataset.tool)));
  canvas.addEventListener('pointerdown',pointerDown);canvas.addEventListener('pointermove',pointerMove);canvas.addEventListener('pointerup',pointerUp);canvas.addEventListener('pointercancel',pointerUp);
  canvas.addEventListener('wheel',e=>{e.preventDefault();const box=canvas.getBoundingClientRect(),sx=e.clientX-box.left,sy=e.clientY-box.top,anchor=imagePoint(e),rotated=rotatedPoint(anchor),factor=Math.exp(-e.deltaY*.0012),next=clamp(state.zoom*factor,.05,12);state.panX=sx-rotated.x*next;state.panY=sy-rotated.y*next;state.zoom=next;updateUI();},{passive:false});
  window.addEventListener('resize',resize);window.addEventListener('keydown',e=>{const editing=['INPUT','TEXTAREA'].includes(document.activeElement.tagName);if(e.code==='Space'&&!editing){state.space=true;e.preventDefault();}if(e.key==='Escape'){state.entranceDraft=null;updateEntranceUI();state.roadSelection=null;updateRoadSelectionUI();state.lineStart=null;state.linePreview=null;state.fillPreview=null;state.gapHints=[];state.draft=null;state.pastePreview=null;updateGapStatus();if(state.tool==='pasteBuilding')setTool('select');if(state.tool==='route')clearRoute(false);render();}if((e.key==='Delete'||e.key==='Backspace')&&!editing){e.preventDefault();deleteSelected();}if((e.metaKey||e.ctrlKey)&&e.key.toLowerCase()==='z'&&document.activeElement.id!=='entranceLabel'){e.preventDefault();$('undoBtn').click();}if((e.metaKey||e.ctrlKey)&&e.key.toLowerCase()==='c'&&!editing&&state.selected?.type==='building'){const item=selectedItem();state.buildingTemplate={points:clone(item.points),templateWidth:item.templateWidth,templateHeight:item.templateHeight,scaleX:item.scaleX,scaleY:item.scaleY,rotation:item.rotation,lineWidth:item.lineWidth,children:clone(item.children||[]),patchData:item.patchData};toast('所选复制区域已复制，按 Command+V 粘贴');e.preventDefault();}if((e.metaKey||e.ctrlKey)&&e.key.toLowerCase()==='v'&&!editing&&state.buildingTemplate){setTool('pasteBuilding');toast('移动鼠标并单击放置复制区域');e.preventDefault();}});window.addEventListener('keyup',e=>{if(e.code==='Space')state.space=false;});
  $('fitBtn').onclick=()=>state.cropPreview?fitCrop():fit();$('zoomInBtn').onclick=()=>{state.zoom=clamp(state.zoom*1.25,.05,12);updateUI();};$('zoomOutBtn').onclick=()=>{state.zoom=clamp(state.zoom/1.25,.05,12);updateUI();};$('toggleBaseBtn').onclick=()=>{state.entranceDraft=null;state.roadSelection=null;state.showOriginal=!state.showOriginal;$('toggleBaseBtn').textContent=state.showOriginal?'查看灰度图':'查看原图';state.lineStart=null;state.linePreview=null;state.fillPreview=null;state.gapHints=[];state.draft=null;state.pastePreview=null;updateGapStatus();if(state.showOriginal){state.selected=null;$('modeText').textContent='原图预览';$('toolHint').textContent='仅显示原始截图；可缩放、平移，切回灰度图后继续编辑。';toast('已进入纯原图预览，人工标注已暂时隐藏');}else{$('modeText').textContent=labels[state.tool];$('toolHint').textContent=hints[state.tool];toast('已返回灰度编辑视图');}updateRoadStyleUI();updateSelectionFields();render();};$('rotateViewBtn').onclick=()=>{state.viewRotation=(state.viewRotation+90)%360;$('rotateViewBtn').textContent=`顺时针旋转 90° · 当前 ${state.viewRotation}°`;state.cropPreview?fitCrop():fit();updateUI();};
  $('undoBtn').onclick=()=>{if(state.showOriginal){toast('当前为原图预览，请切回灰度图后编辑');return;}if(!state.undo.length)return;state.redo.push(snapshot());restore(state.undo.pop());};$('redoBtn').onclick=()=>{if(state.showOriginal){toast('当前为原图预览，请切回灰度图后编辑');return;}if(!state.redo.length)return;state.undo.push(snapshot());restore(state.redo.pop());};$('deleteBtn').onclick=deleteSelected;
  $('toggleCropPreviewBtn').onclick=()=>{if(!state.cropRect)return;state.cropPreview=!state.cropPreview;if(state.cropPreview)fitCrop();else fit();updateUI();toast(state.cropPreview?'正在显示裁剪结果':'已返回完整地图');};
  $('clearCropBtn').onclick=()=>{if(state.showOriginal){toast('当前为原图预览，请切回灰度图后编辑');return;}if(!state.cropRect)return;pushUndo();const wasPreview=state.cropPreview;state.cropRect=null;state.cropPreview=false;if(wasPreview)fit();updateUI();toast('已清除裁剪框');};$('clearBtn').onclick=()=>{if(state.showOriginal){toast('当前为原图预览，请切回灰度图后编辑');return;}if(!allItems().length||!confirm('确认清空全部人工标注？自动底图不会被删除。'))return;pushUndo();state.annotations={freeRects:[],obstacleRects:[],obstacleFills:[],obstacleLines:[],roadLines:[],outlineRects:[],roadErases:[],buildingCopies:[],specialPoints:[]};state.entranceDraft=null;state.fillPreview=null;state.gapHints=[];state.selected=null;updateUI();};
  $('outlineWidthSlider').oninput=()=>changeOutlineWidth();
  $('outlineWidthSlider').onchange=()=>{changeOutlineWidth();outlineWidthEditingId=null;};
  $('outlineWidthSlider').onblur=()=>{outlineWidthEditingId=null;};
  $('matchOutlineWidthBtn').onclick=matchOutlineWidth;
  $('outlineWidth').onchange=()=>{$('outlineWidth').value=normalizedOutlineWidth($('outlineWidth').value);};
  $('entrancePanel').onsubmit=e=>{e.preventDefault();saveEntrance();};
  $('cancelEntranceBtn').onclick=()=>{state.entranceDraft=null;updateUI();};
  $('deleteEntranceBtn').onclick=deleteSelected;
  $('deleteRoadsBtn').onclick=deleteRoads;
  $('thickenRoadsBtn').onclick=thickenRoads;
  $('roadColor').oninput=changeRoadColor;
  $('roadColor').onchange=()=>{changeRoadColor();roadColorEditing=false;};
  $('roadColor').onblur=()=>{roadColorEditing=false;};
  $('exportBtn').onclick=exportAll;$('importBtn').onclick=()=>$('importFile').click();$('importFile').onchange=e=>{if(e.target.files[0])importJson(e.target.files[0]);e.target.value='';};
  $('layerList').onclick=e=>{if(state.showOriginal){toast('当前为原图预览，请切回灰度图后编辑');return;}const row=e.target.closest('.layer-item');if(!row)return;state.entranceDraft=null;state.selected={type:row.dataset.type,id:row.dataset.id};if(e.target.dataset.delete){deleteSelected();}else updateUI();};
  ['selectedOutlineWidth','selectedName','rectX','rectY','rectW','rectH','lineX1','lineY1','lineX2','lineY2','selectedLineWidth','buildingX','buildingY','buildingW','buildingH'].forEach(id=>$(id).addEventListener('change',applyFields));
  ['orthogonalRoads'].forEach(id=>$(id).onchange=()=>{if(state.lineStart&&state.linePreview)state.linePreview=snapForLine(state.lineStart,state.linePreview);render();});
  $('duplicateBuildingBtn').onclick=duplicateBuilding;
  $('clearRouteBtn').onclick=()=>clearRoute(true);
  $('routeClearance').onchange=()=>{const value=clamp(Math.round(Number($('routeClearance').value)||0),0,50);$('routeClearance').value=value;invalidateRoute();render();};
  init().catch(err=>{console.error(err);alert('启动失败：'+err.message);});
})();
</script>
</body>
</html>'''


class EditorHandler(BaseHTTPRequestHandler):
    html = HTML.encode("utf-8")
    base_png = b""
    original_png = b""
    config_json = b"{}"
    config: dict[str, object] = {}
    project_path: Path | None = None

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self.send_content(self.html, "text/html; charset=utf-8", cache=False)
        elif path == "/base.png":
            self.send_content(self.base_png, "image/png", cache=False)
        elif path == "/original.png":
            self.send_content(self.original_png, "image/png", cache=False)
        elif path == "/config.json":
            self.send_content(self.config_json, "application/json; charset=utf-8", cache=False)
        elif path == "/project.json":
            content = b"null"
            if self.project_path is not None and self.project_path.is_file():
                content = self.project_path.read_bytes()
            self.send_content(content, "application/json; charset=utf-8", cache=False)
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path != "/project.json" or self.project_path is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > 50 * 1024 * 1024:
            self.send_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
            return
        try:
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError("工程文件格式无效")
            source = data.get("source", {})
            if (
                int(source.get("width", -1)) != int(self.config["width"])
                or int(source.get("height", -1)) != int(self.config["height"])
                or not isinstance(data.get("project", {}).get("annotations"), dict)
            ):
                raise ValueError("工程文件与当前原图不匹配")
            self.project_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.project_path.with_suffix(self.project_path.suffix + ".tmp")
            temporary.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            os.replace(temporary, self.project_path)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            response = json.dumps(
                {"ok": False, "error": str(error)}, ensure_ascii=False
            ).encode("utf-8")
            self.send_response(HTTPStatus.BAD_REQUEST)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(response)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(response)
            return
        self.send_content(b'{"ok":true}', "application/json; charset=utf-8", cache=False)

    def send_content(self, content: bytes, mime: str, cache: bool = True) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "public, max-age=3600" if cache else "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self' data:; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; connect-src 'self'")
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, fmt: str, *args: object) -> None:
        return


def available_port(preferred: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])


def serve_editor(
    rgb: np.ndarray,
    binary_rgb: np.ndarray,
    report: ConversionReport,
    source: Path,
    project_path: Path,
    port: int,
    open_browser: bool,
) -> None:
    EditorHandler.base_png = png_bytes(binary_rgb)
    EditorHandler.original_png = png_bytes(rgb)
    config = {
        "sourceName": source.name,
        "stem": source.stem,
        "width": report.width,
        "height": report.height,
        "defaultLineWidth": 5,
        "defaultRoadWidth": report.road_width_px,
        "defaultOutlineWidth": estimate_outline_width(binary_rgb),
        "report": asdict(report),
    }
    EditorHandler.config = config
    EditorHandler.config_json = json.dumps(config, ensure_ascii=False).encode("utf-8")
    EditorHandler.project_path = project_path
    selected_port = available_port(port)
    server = ThreadingHTTPServer(("127.0.0.1", selected_port), EditorHandler)
    url = f"http://127.0.0.1:{selected_port}/"
    print(f"本地编辑器：{url}")
    print("按 Control+C 停止。")
    if open_browser:
        threading.Timer(0.35, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n编辑器已停止。")
    finally:
        server.server_close()


def main() -> int:
    args = parse_args()
    source = Path(args.input).expanduser().resolve()
    if not source.is_file():
        raise SystemExit(f"输入图片不存在：{source}")
    if args.boundary_width < 1:
        raise SystemExit("--boundary-width 必须大于或等于 1")
    if not 1 <= args.road_width <= 100:
        raise SystemExit("--road-width 必须在 1–100 之间")

    rgb = read_rgb(source)
    binary_rgb, report = convert_map(
        rgb, source.name, args.boundary_width, args.green_tolerance, args.road_width
    )
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    resolved_output_dir = output_dir.resolve()
    base_path, report_path = write_initial_outputs(
        resolved_output_dir, source.stem, binary_rgb, report
    )
    print(f"初始灰度底图：{base_path}")
    print(f"转换报告：{report_path}")
    if args.convert_only:
        return 0
    serve_editor(
        rgb,
        binary_rgb,
        report,
        source,
        resolved_output_dir / f"{source.stem}_editor_project.json",
        args.port,
        open_browser=not args.no_browser,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
