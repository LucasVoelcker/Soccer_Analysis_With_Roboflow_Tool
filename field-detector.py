from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np

RuleValue = Union[float, str, tuple[str, float]]


@dataclass(frozen=True)
class Rule:
    name: str
    length_min: RuleValue
    length_max: RuleValue
    angle_min: RuleValue
    angle_max: RuleValue
    x_min: RuleValue = float("-inf")
    x_max: RuleValue = float("inf")
    y_min: RuleValue = float("-inf")
    y_max: RuleValue = float("inf")


@dataclass
class DetectedLine:
    image: str
    grouped_lines_final: int
    line_id: int
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    left_x: int
    left_y: int
    right_x: int
    right_y: int
    angle_deg: float
    length: float


class FieldDetector:
    def __init__(self, image_path: str, output_dir: Optional[str] = None, save_steps: bool = False):
        self.image_path = Path(image_path)
        self.save_steps = save_steps

        if output_dir is None:
            img_prefix = self.image_path.stem[-4:]
            self.output_dir = Path(
                rf"C:\Users\lucas\OneDrive\Documentos\Lucas\roboflow\test\out-field-detector-{img_prefix}"
            )
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.rules = self._build_default_rules()
        self.selected_by_rule: Dict[str, Optional[DetectedLine]] = {}
        self.detected_lines_txt: Optional[Path] = None
        self.filtered_lines_txt: Optional[Path] = None

        self.tophat_kernel = (13, 13)
        self.close_kernel = (7, 7)
        self.open_kernel = (1, 1)
        self.min_component_area_pre = 200
        self.adaptive_blocksize = 11
        self.adaptive_c = -3
        self.line_threshold = 180
        self.min_component_area_lines = 50

        self.max_gap_fill = 12
        self.min_run_fill = 3
        self.max_iters_fill = 3
        self.min_line_length = 20
        self.max_line_gap_hough = 10
        self.hough_threshold = 35
        self.orientation_tol_deg = 5.0
        self.line_distance_tol_px = 5.0
        self.endpoint_gap_tol_px = 600.0 # antes estava em 300
        self.connect_thickness = 1
        self.auto_polarity = True

    def run_processing(self) -> Dict[str, Optional[Dict[str, object]]]:
        out_detect_final = self._run_detect_lines_complete()
        self.detected_lines_txt = self._run_complete_discontinuous_lines_v2(out_detect_final)
        all_lines = self._parse_blocks(self.detected_lines_txt)
        self.selected_by_rule = self._apply_rules(all_lines, self.rules)
        self.filtered_lines_txt = self._save_filtered_result(self.selected_by_rule)
        return self.get_final_lines_values()

    def get_final_lines_values(self) -> Dict[str, Optional[Dict[str, object]]]:
        output: Dict[str, Optional[Dict[str, object]]] = {}
        for rule in self.rules:
            line = self.selected_by_rule.get(rule.name)
            if line is None:
                output[rule.name] = None
            else:
                output[rule.name] = {
                    "left_point": {"x": line.left_x, "y": line.left_y},
                    "right_point": {"x": line.right_x, "y": line.right_y},
                }
        return output

    @staticmethod
    def _save_path(path: Path, img: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), img)

    def _save_step(self, name: str, img: np.ndarray, always: bool = False) -> None:
        if self.save_steps or always:
            self._save_path(self.output_dir / name, img)

    @staticmethod
    def _remove_small_components(bin_img: np.ndarray, min_area: int) -> np.ndarray:
        num, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img, connectivity=8)
        output = np.zeros_like(bin_img)
        for i in range(1, num):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                output[labels == i] = 255
        return output

    @staticmethod
    def _save_resp_vis(resp: np.ndarray) -> np.ndarray:
        vis = cv2.normalize(resp, None, 0, 255, cv2.NORM_MINMAX)
        return vis.astype(np.uint8)

    def _run_detect_lines_complete(self) -> Path:
        img = cv2.imread(str(self.image_path))
        if img is None:
            raise ValueError(f"Image not found: {self.image_path}")

        blur = cv2.GaussianBlur(img, (5, 5), 0)
        self._save_step("01_blur.jpg", blur)

        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
        self._save_step("02_V.png", v_channel)

        kernel_tophat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.tophat_kernel)
        tophat = cv2.morphologyEx(v_channel, cv2.MORPH_TOPHAT, kernel_tophat)
        self._save_step("03_tophat.png", tophat)

        mask = cv2.adaptiveThreshold(
            tophat,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self.adaptive_blocksize,
            self.adaptive_c,
        )
        self._save_step("04_mask_adaptive.png", mask)

        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.close_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        self._save_step("05_mask_close.png", mask)

        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.open_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        self._save_step("06_mask_open.png", mask)

        #mask = self._remove_small_components(mask, self.min_component_area_pre)
        #self._save_step("07_mask_clean.png", mask)

        img_bin = (mask > 0).astype(np.float32)

        k_h = np.array([
            [-1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1],
            [1, 1, 1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2, 2, 2],
            [1, 1, 1, 1, 1, 1, 1],
            [-1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1],
        ], dtype=np.float32)
        k_v = k_h.T.copy()
        k_d1 = np.array([
            [2, 1, -1, -1, -1, -1, -1],
            [1, 2, 1, -1, -1, -1, -1],
            [-1, 1, 2, 1, -1, -1, -1],
            [-1, -1, 1, 2, 1, -1, -1],
            [-1, -1, -1, 1, 2, 1, -1],
            [-1, -1, -1, -1, 1, 2, 1],
            [-1, -1, -1, -1, -1, 1, 2],
        ], dtype=np.float32)
        k_d2 = np.fliplr(k_d1).copy()

        resp_h = cv2.filter2D(img_bin, cv2.CV_32F, k_h)
        resp_v = cv2.filter2D(img_bin, cv2.CV_32F, k_v)
        resp_d1 = cv2.filter2D(img_bin, cv2.CV_32F, k_d1)
        resp_d2 = cv2.filter2D(img_bin, cv2.CV_32F, k_d2)

        vis_h = self._save_resp_vis(resp_h)
        vis_v = self._save_resp_vis(resp_v)
        vis_d1 = self._save_resp_vis(resp_d1)
        vis_d2 = self._save_resp_vis(resp_d2)

        self._save_step("08_resp_h.png", vis_h)
        self._save_step("09_resp_v.png", vis_v)
        self._save_step("10_resp_d1.png", vis_d1)
        self._save_step("11_resp_d2.png", vis_d2)

        _, binary_h = cv2.threshold(vis_h, self.line_threshold, 255, cv2.THRESH_BINARY)
        _, binary_v = cv2.threshold(vis_v, self.line_threshold, 255, cv2.THRESH_BINARY)
        _, binary_d1 = cv2.threshold(vis_d1, self.line_threshold, 255, cv2.THRESH_BINARY)
        _, binary_d2 = cv2.threshold(vis_d2, self.line_threshold, 255, cv2.THRESH_BINARY)

        self._save_step(f"12_out-{self.line_threshold}--resp_h.png", binary_h)
        self._save_step(f"13_out-{self.line_threshold}--resp_v.png", binary_v)
        self._save_step(f"14_out-{self.line_threshold}--resp_d1.png", binary_d1)
        self._save_step(f"15_out-{self.line_threshold}--resp_d2.png", binary_d2)

        mask_or = cv2.bitwise_or(binary_h, binary_v)
        mask_or = cv2.bitwise_or(mask_or, binary_d1)
        mask_or = cv2.bitwise_or(mask_or, binary_d2)
        self._save_step(f"16_out-{self.line_threshold}-combined.png", mask_or)

        bin_img = (mask_or > 0).astype(np.uint8) * 255
        out_img = self._remove_small_components(bin_img, self.min_component_area_lines)

        out_detect_final = self.output_dir / f"17_out-{self.line_threshold}-combined-clean.png"
        self._save_path(out_detect_final, out_img)
        return out_detect_final

    @staticmethod
    def _to_binary(img_gray: np.ndarray) -> np.ndarray:
        _, bin_img = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
        return bin_img

    def _detect_line_polarity(self, bin_img: np.ndarray) -> int:
        if not self.auto_polarity:
            return 255
        white_ratio = np.mean(bin_img == 255)
        return 0 if white_ratio > 0.5 else 255

    @staticmethod
    def _fill_gaps_1d(vec: np.ndarray, max_gap: int, min_run: int) -> tuple[np.ndarray, int]:
        out = vec.copy()
        padded = np.r_[False, vec, False]
        diff = np.diff(padded.astype(np.int8))
        starts = np.flatnonzero(diff == 1)
        ends = np.flatnonzero(diff == -1) - 1

        if starts.size < 2:
            return out, 0

        added = 0
        for i in range(starts.size - 1):
            left_start, left_end = starts[i], ends[i]
            right_start, right_end = starts[i + 1], ends[i + 1]
            left_len = left_end - left_start + 1
            right_len = right_end - right_start + 1
            gap_start = left_end + 1
            gap_end = right_start - 1
            gap_len = gap_end - gap_start + 1

            if gap_len > 0 and gap_len <= max_gap and left_len >= min_run and right_len >= min_run:
                add_now = np.count_nonzero(~out[gap_start:right_start])
                out[gap_start:right_start] = True
                added += int(add_now)

        return out, added

    @staticmethod
    def _diag_indices(shape: tuple[int, int], offset: int) -> tuple[np.ndarray, np.ndarray]:
        h, w = shape
        if offset >= 0:
            n = min(h, w - offset)
            rows = np.arange(n)
            cols = rows + offset
        else:
            n = min(w, h + offset)
            cols = np.arange(n)
            rows = cols - offset
        return rows, cols

    def _process_rows(self, mat: np.ndarray, max_gap: int, min_run: int) -> int:
        added_total = 0
        for r in range(mat.shape[0]):
            filled, added = self._fill_gaps_1d(mat[r, :], max_gap, min_run)
            mat[r, :] = filled
            added_total += added
        return added_total

    def _process_cols(self, mat: np.ndarray, max_gap: int, min_run: int) -> int:
        added_total = 0
        for c in range(mat.shape[1]):
            filled, added = self._fill_gaps_1d(mat[:, c], max_gap, min_run)
            mat[:, c] = filled
            added_total += added
        return added_total

    def _process_diags(self, mat: np.ndarray, max_gap: int, min_run: int) -> int:
        added_total = 0
        h, w = mat.shape
        for off in range(-h + 1, w):
            rr, cc = self._diag_indices(mat.shape, off)
            if rr.size < 2:
                continue
            filled, added = self._fill_gaps_1d(mat[rr, cc], max_gap, min_run)
            mat[rr, cc] = filled
            added_total += added
        return added_total

    def _complete_discontinuous_lines(self, line_mask: np.ndarray, max_gap: int, min_run: int, max_iters: int) -> np.ndarray:
        work = line_mask.copy()
        for _ in range(max_iters):
            added = 0
            added += self._process_rows(work, max_gap, min_run)
            added += self._process_cols(work, max_gap, min_run)
            added += self._process_diags(work, max_gap, min_run)
            flipped = np.fliplr(work)
            added += self._process_diags(flipped, max_gap, min_run)
            if added == 0:
                break
        return work

    @staticmethod
    def _compute_orientation_deg(x1: int, y1: int, x2: int, y2: int) -> float:
        angle = np.degrees(np.arctan2((y2 - y1), (x2 - x1)))
        if angle < 0:
            angle += 180.0
        return float(angle)

    def _detect_and_annotate_lines(self, line_mask: np.ndarray, min_len: int, max_gap: int, hough_thr: int) -> tuple[np.ndarray, List[Dict[str, object]]]:
        hough_input = line_mask.astype(np.uint8) * 255
        lines = cv2.HoughLinesP(hough_input, rho=1, theta=np.pi / 180, threshold=hough_thr, minLineLength=min_len, maxLineGap=max_gap)

        vis = cv2.cvtColor(hough_input, cv2.COLOR_GRAY2BGR)
        line_data: List[Dict[str, object]] = []
        if lines is None:
            return vis, line_data

        for i, line in enumerate(lines[:, 0, :], start=1):
            x1, y1, x2, y2 = map(int, line)
            angle = self._compute_orientation_deg(x1, y1, x2, y2)
            length = float(np.hypot(x2 - x1, y2 - y1))
            cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            line_data.append({"id": i, "p1": (x1, y1), "p2": (x2, y2), "length": length, "angle_deg": angle})

        return vis, line_data

    @staticmethod
    def _angle_diff_deg(a: float, b: float) -> float:
        d = abs(a - b) % 180.0
        return float(min(d, 180.0 - d))

    @staticmethod
    def _point_line_distance(px: int, py: int, x1: int, y1: int, x2: int, y2: int) -> float:
        dx = x2 - x1
        dy = y2 - y1
        den = np.hypot(dx, dy)
        if den == 0:
            return float(np.hypot(px - x1, py - y1))
        return float(abs(dy * px - dx * py + x2 * y1 - y2 * x1) / den)

    @staticmethod
    def _line_direction(x1: int, y1: int, x2: int, y2: int) -> tuple[float, float]:
        vx = float(x2 - x1)
        vy = float(y2 - y1)
        n = np.hypot(vx, vy)
        if n == 0:
            return 1.0, 0.0
        return float(vx / n), float(vy / n)

    def _endpoint_gap_along_direction(self, li: Dict[str, object], lj: Dict[str, object]) -> float:
        x1a, y1a = li["p1"]
        x2a, y2a = li["p2"]
        ux, uy = self._line_direction(x1a, y1a, x2a, y2a)

        ai = np.array([[x1a, y1a], [x2a, y2a]], dtype=np.float32)
        bj = np.array([list(lj["p1"]), list(lj["p2"])], dtype=np.float32)
        proj_i = ai @ np.array([ux, uy], dtype=np.float32)
        proj_j = bj @ np.array([ux, uy], dtype=np.float32)

        i_min, i_max = float(np.min(proj_i)), float(np.max(proj_i))
        j_min, j_max = float(np.min(proj_j)), float(np.max(proj_j))
        if i_max < j_min:
            return j_min - i_max
        if j_max < i_min:
            return i_min - j_max
        return 0.0

    @staticmethod
    def _closest_endpoints(li: Dict[str, object], lj: Dict[str, object]) -> tuple[tuple[int, int], tuple[int, int], float]:
        a_pts = [li["p1"], li["p2"]]
        b_pts = [lj["p1"], lj["p2"]]
        best_a = a_pts[0]
        best_b = b_pts[0]
        best_d = float("inf")
        for pa in a_pts:
            for pb in b_pts:
                d = float(np.hypot(pa[0] - pb[0], pa[1] - pb[1]))
                if d < best_d:
                    best_d = d
                    best_a, best_b = pa, pb
        return best_a, best_b, best_d

    def _are_lines_connectable(self, li: Dict[str, object], lj: Dict[str, object], orient_tol: float, line_dist_tol: float, endpoint_gap_tol: float) -> bool:
        if self._angle_diff_deg(li["angle_deg"], lj["angle_deg"]) > orient_tol:
            return False

        x1a, y1a = li["p1"]
        x2a, y2a = li["p2"]
        x1b, y1b = lj["p1"]
        x2b, y2b = lj["p2"]

        d1 = self._point_line_distance(x1b, y1b, x1a, y1a, x2a, y2a)
        d2 = self._point_line_distance(x2b, y2b, x1a, y1a, x2a, y2a)
        d3 = self._point_line_distance(x1a, y1a, x1b, y1b, x2b, y2b)
        d4 = self._point_line_distance(x2a, y2a, x1b, y1b, x2b, y2b)
        if min(d1, d2, d3, d4) > line_dist_tol:
            return False

        return self._endpoint_gap_along_direction(li, lj) <= endpoint_gap_tol

    def _connect_near_collinear_lines(self, line_mask: np.ndarray, lines_info: List[Dict[str, object]], orient_tol: float, line_dist_tol: float, endpoint_gap_tol: float, thickness: int) -> tuple[np.ndarray, int]:
        out = (line_mask.astype(np.uint8) * 255).copy()
        pairs_connected = 0
        for i in range(len(lines_info)):
            for j in range(i + 1, len(lines_info)):
                li, lj = lines_info[i], lines_info[j]
                if not self._are_lines_connectable(li, lj, orient_tol, line_dist_tol, endpoint_gap_tol):
                    continue
                p1, p2, d = self._closest_endpoints(li, lj)
                if d > endpoint_gap_tol:
                    continue
                cv2.line(out, p1, p2, 255, thickness, cv2.LINE_AA)
                pairs_connected += 1
        return out > 0, pairs_connected

    def _group_connectable_lines(self, lines_info: List[Dict[str, object]], orient_tol: float, line_dist_tol: float, endpoint_gap_tol: float) -> List[List[Dict[str, object]]]:
        n = len(lines_info)
        if n == 0:
            return []

        parent = list(range(n))
        rank = [0] * n

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            if rank[ra] < rank[rb]:
                parent[ra] = rb
            elif rank[ra] > rank[rb]:
                parent[rb] = ra
            else:
                parent[rb] = ra
                rank[ra] += 1

        for i in range(n):
            for j in range(i + 1, n):
                if self._are_lines_connectable(lines_info[i], lines_info[j], orient_tol, line_dist_tol, endpoint_gap_tol):
                    _, _, d = self._closest_endpoints(lines_info[i], lines_info[j])
                    if d <= endpoint_gap_tol:
                        union(i, j)

        groups: Dict[int, List[Dict[str, object]]] = {}
        for idx in range(n):
            root = find(idx)
            groups.setdefault(root, []).append(lines_info[idx])
        return list(groups.values())

    def _summarize_line_group(self, group: List[Dict[str, object]], group_id: int) -> Dict[str, object]:
        points = [p for item in group for p in (item["p1"], item["p2"])]
        x_min = int(min(p[0] for p in points))
        x_max = int(max(p[0] for p in points))
        y_min = int(min(p[1] for p in points))
        y_max = int(max(p[1] for p in points))

        p_min_x = min(points, key=lambda p: (p[0], p[1]))
        p_max_x = max(points, key=lambda p: (p[0], p[1]))

        return {
            "id": group_id,
            "p_min_x": p_min_x,
            "p_max_x": p_max_x,
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max,
        }

    @staticmethod
    def _annotate_grouped_lines(base_img_gray: np.ndarray, grouped_lines: List[Dict[str, object]]) -> np.ndarray:
        vis = cv2.cvtColor(base_img_gray, cv2.COLOR_GRAY2BGR)
        for item in grouped_lines:
            cv2.line(vis, item["p_min_x"], item["p_max_x"], (0, 255, 255), 2, cv2.LINE_AA)
        return vis

    def _run_complete_discontinuous_lines_v2(self, input_path: Path) -> Path:
        img = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Image not found: {input_path}")

        bin_img = self._to_binary(img)
        line_value = self._detect_line_polarity(bin_img)
        line_mask = bin_img == line_value

        completed_mask = self._complete_discontinuous_lines(line_mask, self.max_gap_fill, self.min_run_fill, self.max_iters_fill)
        annotated, lines_info = self._detect_and_annotate_lines(completed_mask, self.min_line_length, self.max_line_gap_hough, self.hough_threshold)

        result_completed = np.where(completed_mask, line_value, 255 - line_value).astype(np.uint8)

        out_completed = self.output_dir / "completed-lines-v2.png"
        out_annotated = self.output_dir / "detected-lines-v2-annotated.png"
        out_connected = self.output_dir / "completed-lines-v2-connected.png"
        out_connected_annotated = self.output_dir / "detected-lines-v2-connected-annotated.png"
        out_final_annotated = self.output_dir / "detected-lines-v2-final-image-annotated.png"
        out_final_txt = self.output_dir / "detected-lines-v2-final-image.txt"

        self._save_path(out_completed, result_completed)
        self._save_path(out_annotated, annotated)

        connected_mask, _ = self._connect_near_collinear_lines(completed_mask, lines_info, self.orientation_tol_deg, self.line_distance_tol_px, self.endpoint_gap_tol_px, self.connect_thickness)
        connected_result = np.where(connected_mask, line_value, 255 - line_value).astype(np.uint8)
        self._save_path(out_connected, connected_result)

        annotated_connected, _ = self._detect_and_annotate_lines(connected_mask, self.min_line_length, self.max_line_gap_hough, self.hough_threshold)
        self._save_path(out_connected_annotated, annotated_connected)

        line_groups = self._group_connectable_lines(lines_info, self.orientation_tol_deg, self.line_distance_tol_px, self.endpoint_gap_tol_px)
        grouped_lines_final = [self._summarize_line_group(group, i + 1) for i, group in enumerate(line_groups)]
        annotated_final = self._annotate_grouped_lines(connected_result, grouped_lines_final)
        self._save_path(out_final_annotated, annotated_final)

        with out_final_txt.open("w", encoding="utf-8") as f:
            f.write(f"image: {out_connected}\n")
            f.write(f"grouped_lines_final: {len(grouped_lines_final)}\n")
            f.write("id,x_min,y_min,x_max,y_max,x1,y1,x2,y2\n")
            for item in grouped_lines_final:
                x1_bbox, y1_bbox = item["x_min"], item["y_min"]
                x2_bbox, y2_bbox = item["x_max"], item["y_max"]
                x1, y1 = item["p_min_x"]
                x2, y2 = item["p_max_x"]
                f.write(f"{item['id']},{x1_bbox},{y1_bbox},{x2_bbox},{y2_bbox},{x1},{y1},{x2},{y2}\n")

        return out_final_txt

    @staticmethod
    def _compute_length_and_angle(
        x_min: int,
        y_min: int,
        x_max: int,
        y_max: int,
        left_y: Optional[int] = None,
        right_y: Optional[int] = None,
    ) -> tuple[float, float]:
        width = abs(x_max - x_min)
        height = abs(y_max - y_min)
        length = math.hypot(width, height)
        angle_base = math.degrees(math.atan2(height, width))

        if left_y is None or right_y is None:
            left_y = y_min
            right_y = y_max

        if left_y < right_y:
            angle_deg = 180.0 - angle_base
        else:
            angle_deg = angle_base

        return length, angle_deg

    @staticmethod
    def _angle_in_range(angle: float, angle_min: float, angle_max: float) -> bool:
        if angle_min <= angle_max:
            return angle_min <= angle <= angle_max
        return angle >= angle_min or angle <= angle_max

    def _parse_blocks(self, txt_path: Path) -> List[DetectedLine]:
        raw_lines = [line.strip().lstrip("\ufeff") for line in txt_path.read_text(encoding="utf-8").splitlines()]
        results: List[DetectedLine] = []
        i = 0

        while i < len(raw_lines):
            line = raw_lines[i]
            if not line:
                i += 1
                continue
            if not line.startswith("image:"):
                i += 1
                continue

            image_path = line.split("image:", 1)[1].strip()
            i += 1

            while i < len(raw_lines) and not raw_lines[i]:
                i += 1
            if i >= len(raw_lines) or not raw_lines[i].startswith("grouped_lines_final:"):
                raise ValueError(f"Invalid block for image '{image_path}': missing grouped_lines_final")

            grouped_lines_final = int(raw_lines[i].split(":", 1)[1].strip())
            i += 1

            while i < len(raw_lines) and not raw_lines[i]:
                i += 1

            expected_header = "id,x_min,y_min,x_max,y_max,x1,y1,x2,y2"
            if i >= len(raw_lines) or raw_lines[i] != expected_header:
                raise ValueError(f"Invalid header in block '{image_path}'. Expected: {expected_header}")
            i += 1

            while i < len(raw_lines):
                row = raw_lines[i]
                if not row:
                    i += 1
                    break
                if row.startswith("image:"):
                    break

                parsed = next(csv.reader([row]))
                if len(parsed) != 9:
                    raise ValueError(f"Invalid line in block '{image_path}': {row}")

                x_min = int(parsed[1])
                y_min = int(parsed[2])
                x_max = int(parsed[3])
                y_max = int(parsed[4])
                x1 = int(parsed[5])
                y1 = int(parsed[6])
                x2 = int(parsed[7])
                y2 = int(parsed[8])

                if x1 <= x2:
                    left_x, left_y, right_x, right_y = x1, y1, x2, y2
                else:
                    left_x, left_y, right_x, right_y = x2, y2, x1, y1

                length, angle_deg = self._compute_length_and_angle(x_min, y_min, x_max, y_max, left_y=left_y, right_y=right_y)

                results.append(
                    DetectedLine(
                        image=image_path,
                        grouped_lines_final=grouped_lines_final,
                        line_id=int(parsed[0]),
                        x_min=x_min,
                        y_min=y_min,
                        x_max=x_max,
                        y_max=y_max,
                        left_x=left_x,
                        left_y=left_y,
                        right_x=right_x,
                        right_y=right_y,
                        angle_deg=angle_deg,
                        length=length,
                    )
                )
                i += 1

        return results

    @staticmethod
    def _resolve_rule_value(value: RuleValue, selected_by_rule: Dict[str, Optional[DetectedLine]]) -> float:
        if isinstance(value, (int, float)):
            return float(value)

        fallback: Optional[float] = None
        if isinstance(value, tuple):
            ref = str(value[0]).strip()
            fallback = float(value[1])
        else:
            raw = str(value).strip()
            if "??" in raw:
                ref_part, fallback_part = raw.split("??", 1)
                ref = ref_part.strip()
                fallback = float(fallback_part.strip())
            else:
                ref = raw

        match = re.fullmatch(
            r"(?P<ref>[\w-]+\.(?:x_min|x_max|y_min|y_max|angle_deg|length))"
            r"(?:\s*(?P<op>[+-])\s*(?P<offset>\d+(?:\.\d+)?))?",
            ref,
        )
        if not match:
            raise ValueError(f"Invalid rule reference: '{value}'")

        base_ref = match.group("ref")
        rule_name, field_name = base_ref.rsplit(".", 1)
        ref_line = selected_by_rule.get(rule_name)

        if ref_line is None:
            if fallback is not None:
                return fallback
            raise ValueError(f"Reference '{value}' has no resolved line and no fallback")

        resolved = float(getattr(ref_line, field_name))
        op = match.group("op")
        offset_text = match.group("offset")
        if op and offset_text:
            offset = float(offset_text)
            resolved = resolved + offset if op == "+" else resolved - offset

        return resolved

    def _resolve_rule(self, rule: Rule, selected_by_rule: Dict[str, Optional[DetectedLine]]) -> Rule:
        return Rule(
            name=rule.name,
            length_min=self._resolve_rule_value(rule.length_min, selected_by_rule),
            length_max=self._resolve_rule_value(rule.length_max, selected_by_rule),
            angle_min=self._resolve_rule_value(rule.angle_min, selected_by_rule),
            angle_max=self._resolve_rule_value(rule.angle_max, selected_by_rule),
            x_min=self._resolve_rule_value(rule.x_min, selected_by_rule),
            x_max=self._resolve_rule_value(rule.x_max, selected_by_rule),
            y_min=self._resolve_rule_value(rule.y_min, selected_by_rule),
            y_max=self._resolve_rule_value(rule.y_max, selected_by_rule),
        )

    def _matches_resolved_rule(self, line: DetectedLine, rule: Rule) -> bool:
        return (
            rule.length_min <= line.length <= rule.length_max
            and self._angle_in_range(line.angle_deg, rule.angle_min, rule.angle_max)
            and rule.x_min <= line.x_min <= rule.x_max
            and rule.x_min <= line.x_max <= rule.x_max
            and rule.y_min <= line.y_min <= rule.y_max
            and rule.y_min <= line.y_max <= rule.y_max
        )

    def _apply_rules(self, lines: List[DetectedLine], rules: List[Rule]) -> Dict[str, Optional[DetectedLine]]:
        selected_by_rule: Dict[str, Optional[DetectedLine]] = {}
        available_lines = list(lines)

        for rule in rules:
            resolved_rule = self._resolve_rule(rule, selected_by_rule)
            candidates = [ln for ln in available_lines if self._matches_resolved_rule(ln, resolved_rule)]

            if not candidates:
                selected_by_rule[rule.name] = None
                continue

            selected = min(candidates, key=lambda ln: ln.y_min)
            selected_by_rule[rule.name] = selected
            available_lines.remove(selected)

        return selected_by_rule

    def _save_filtered_result(self, selected_by_rule: Dict[str, Optional[DetectedLine]]) -> Path:
        out_path = self.output_dir / "filtered-lines-final.txt"
        with out_path.open("w", encoding="utf-8") as f:
            f.write("rule_name,left_x,left_y,right_x,right_y\n")
            for rule in self.rules:
                line = selected_by_rule.get(rule.name)
                if line is None:
                    f.write(f"{rule.name},,,,\n")
                else:
                    f.write(f"{rule.name},{line.left_x},{line.left_y},{line.right_x},{line.right_y}\n")
        return out_path

    @staticmethod
    def _build_default_rules() -> List[Rule]:
        return [
            Rule(name="limite-campo-cima", length_min=500, length_max=5000, angle_min=175, angle_max=5),
            Rule(name="linha-meio-campo", length_min=200, length_max=5000, angle_min=85, angle_max=95, y_min=("limite-campo-cima.y_min", 0)),
            Rule(name="limite-area-direita-cima", length_min=100, length_max=800, angle_min=0, angle_max=5, y_min=("limite-campo-cima.y_min - 10", 0), x_min=("linha-meio-campo.x_max + 500", 0)),
            Rule(name="limite-area-esquerda-cima", length_min=100, length_max=800, angle_min=175, angle_max=0, y_min=("limite-campo-cima.y_min - 10", 0), x_max=("linha-meio-campo.x_min - 500", 5000)),
            Rule(name="limite-campo-direita", length_min=100, length_max=5000, angle_min=105, angle_max=165, y_min=("limite-campo-cima.y_min - 10", 0), x_min=("limite-campo-cima.x_max - 10", 0)),
            Rule(name="limite-campo-esquerda", length_min=100, length_max=5000, angle_min=15, angle_max=75, y_min=("limite-campo-cima.y_min - 10", 0), x_max=("limite-campo-cima.x_min + 10", 5000)),
            Rule(name="limite-area-direita-em-campo", length_min=100, length_max=5000, angle_min=110, angle_max=160, y_min=("limite-area-direita-cima.y_max - 10", 5000), x_min=("limite-area-direita-cima.x_min - 10", 5000), x_max=("limite-area-direita-cima.x_max + 100", 0)),
            Rule(name="limite-area-esquerda-em-campo", length_min=100, length_max=5000, angle_min=20, angle_max=70, y_min=("limite-area-esquerda-cima.y_max - 10", 5000), x_min=("limite-area-esquerda-cima.x_min - 100", 5000), x_max=("limite-area-esquerda-cima.x_max + 10", 0)),
            Rule(name="limite-area-direita-baixo", length_min=100, length_max=5000, angle_min=0, angle_max=5, y_min=("limite-area-direita-em-campo.y_max - 100", 5000), x_min=("limite-area-direita-em-campo.x_max - 10", 5000)),
            Rule(name="limite-area-esquerda-baixo", length_min=100, length_max=5000, angle_min=175, angle_max=0, y_min=("limite-area-esquerda-em-campo.y_max - 100", 5000), x_max=("limite-area-esquerda-em-campo.x_min + 10", 5000)),
            Rule(name="limite-campo-baixo", length_min=200, length_max=5000, angle_min=175, angle_max=5, y_min=500),
        ]


def main() -> None:
    image_path = r"C:\Users\lucas\OneDrive\Documentos\Lucas\roboflow\test\160338_mp4-0013_jpg.rf.Fu0OM7uwgRAn6YSC9X0j.jpg"
    detector = FieldDetector(image_path=image_path)
    detector.run_processing()
    print(detector.get_final_lines_values())


if __name__ == "__main__":
    main()
