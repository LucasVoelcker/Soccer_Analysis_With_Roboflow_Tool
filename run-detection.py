from __future__ import annotations

import importlib.util
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


MODEL_PATH = r"C:\Users\lucas\OneDrive\Documentos\Lucas\roboflow\treinamentos\best.pt"
IMAGE_PATH = r"C:\Users\lucas\OneDrive\Documentos\Lucas\roboflow\test-full-process\160338_mp4-0008_jpg.rf.PycGYxW94XpQmfF9EGor.jpg"
IMAGE_PATH = r"C:\Users\lucas\OneDrive\Documentos\Lucas\roboflow\codes\final\processed-video\frame-000011\input-frame.jpg"
SPLIT_IN_FOUR = False

IMAGE_SUFFIX = Path(IMAGE_PATH).stem[-4:]
INFERENCE_OUTPUT_DIR = rf"C:\Users\lucas\OneDrive\Documentos\Lucas\roboflow\codes\final\inference_splits--{IMAGE_SUFFIX}"
LINES_OUTPUT_DIR = rf"C:\Users\lucas\OneDrive\Documentos\Lucas\roboflow\codes\final\out-field-detector-{IMAGE_SUFFIX}"
HOMOGRAPHY_PLOT_PATH = rf"C:\Users\lucas\OneDrive\Documentos\Lucas\roboflow\codes\final\homography-point-{IMAGE_SUFFIX}.png"
CLASS3_BOXES_OUTPUT_PATH = str(Path(INFERENCE_OUTPUT_DIR) / "class3-boxes-with-id.jpg")

PROXIMITY_THRESHOLD_PX = 0
BOUNDS_X_MIN = 0.0
BOUNDS_X_MAX = 1314.0
BOUNDS_Y_MIN = 0.0
BOUNDS_Y_MAX = 732.0
SHOW_POINT_COORDS = False


dict_pts = {
    "limite-campo-cima x limite-campo-esquerda": (0, 68),
    "limite-campo-cima x limite-area-esquerda-em-campo": (16.5, 68),
    "limite-campo-cima x linha-meio-campo": (52.5, 68),
    "limite-campo-cima x limite-area-direita-em-campo": (88.5, 68),
    "limite-campo-cima x limite-campo-direita": (105, 68),
    "limite-area-esquerda-cima x limite-campo-esquerda": (0, 53.25),
    "limite-area-esquerda-cima x limite-area-esquerda-em-campo": (16.5, 53.25),
    "limite-area-esquerda-cima x linha-meio-campo": (52.5, 53.25),
    "limite-area-esquerda-cima x limite-area-direita-em-campo": (88.5, 53.25),
    "limite-area-esquerda-cima x limite-campo-direita": (105, 53.25),
    "limite-area-direita-cima x limite-campo-esquerda": (0, 53.25),
    "limite-area-direita-cima x limite-area-esquerda-em-campo": (16.5, 53.25),
    "limite-area-direita-cima x linha-meio-campo": (52.5, 53.25),
    "limite-area-direita-cima x limite-area-direita-em-campo": (88.5, 53.25),
    "limite-area-direita-cima x limite-campo-direita": (105, 53.25),
    "limite-area-esquerda-baixo x limite-campo-esquerda": (0, 14.75),
    "limite-area-esquerda-baixo x limite-area-esquerda-em-campo": (16.5, 14.75),
    "limite-area-esquerda-baixo x linha-meio-campo": (52.5, 14.75),
    "limite-area-esquerda-baixo x limite-area-direita-em-campo": (88.5, 14.75),
    "limite-area-esquerda-baixo x limite-campo-direita": (105, 14.75),
    "limite-area-direita-baixo x limite-campo-esquerda": (0, 14.75),
    "limite-area-direita-baixo x limite-area-esquerda-em-campo": (16.5, 14.75),
    "limite-area-direita-baixo x linha-meio-campo": (52.5, 14.75),
    "limite-area-direita-baixo x limite-area-direita-em-campo": (88.5, 14.75),
    "limite-area-direita-baixo x limite-campo-direita": (105, 14.75),
    "limite-campo-baixo x limite-campo-esquerda": (0, 0),
    "limite-campo-baixo x limite-area-esquerda-em-campo": (16.5, 0),
    "limite-campo-baixo x linha-meio-campo": (52.5, 0),
    "limite-campo-baixo x limite-area-direita-em-campo": (88.5, 0),
    "limite-campo-baixo x limite-campo-direita": (105, 0)
}


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from: {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _run_inference(
    image_path: str,
    model_path: str,
    output_dir: str,
    split_in_four: bool,
) -> Tuple[Optional[str], List[Dict[str, int]]]:
    base_dir = Path(__file__).resolve().parent
    module = _load_module("inference_yolov8", base_dir / "inference-yolov8.py")

    detector = module.SplitYOLOv8Inference(model_path)
    detector.run_inference(
        image_path=image_path,
        split_in_four=split_in_four,
        save_images=True,
        output_dir=output_dir,
    )

    class_name = detector.class_names.get(3)
    if class_name is None:
        return None, []

    return class_name, detector.get_boxes_by_class(class_name)


def _run_line_detection(image_path: str, output_dir: str) -> Dict[str, Optional[Dict[str, Dict[str, int]]]]:
    base_dir = Path(__file__).resolve().parent
    module = _load_module("field_detector", base_dir / "field-detector.py")

    detector = module.FieldDetector(image_path=image_path, output_dir=output_dir, save_steps=False)
    return detector.run_processing()


def _compute_box_mean_bgr(
    image_path: str,
    boxes: List[Dict[str, int]],
) -> List[Tuple[float, float, float]]:
    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(f"Nao foi possivel carregar a imagem para cor media: {image_path}")

    img_h, img_w = image.shape[:2]
    means: List[Tuple[float, float, float]] = []

    for box in boxes:
        x_min = max(0, min(int(box["x_min"]), img_w))
        y_min = max(0, min(int(box["y_min"]), img_h))
        x_max = max(0, min(int(box["x_max"]), img_w))
        y_max = max(0, min(int(box["y_max"]), img_h))

        if x_max <= x_min or y_max <= y_min:
            means.append((0.0, 0.0, 0.0))
            continue

        roi = image[y_min:y_max, x_min:x_max]
        bgr = roi.reshape(-1, 3).mean(axis=0)
        means.append((float(bgr[0]), float(bgr[1]), float(bgr[2])))

    return means


def _save_class3_boxes_image(
    image_path: str,
    boxes: List[Dict[str, int]],
    output_path: str,
) -> None:
    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(f"Nao foi possivel carregar a imagem para desenhar boxes: {image_path}")

    for i, box in enumerate(boxes, start=1):
        x_min = int(box["x_min"])
        y_min = int(box["y_min"])
        x_max = int(box["x_max"])
        y_max = int(box["y_max"])

        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)

        label = f"ID {i}"
        text_x = x_min
        text_y = max(20, y_min - 8)
        cv2.putText(
            image,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, image)


def _run_homography(
    src_pts: List[Tuple[float, float]],
    dst_pts: List[Tuple[float, float]],
    query_pts: List[Tuple[float, float]],
) -> Tuple[object, List[Tuple[float, float]]]:
    base_dir = Path(__file__).resolve().parent
    module = _load_module("apply_homography", base_dir / "apply-homography.py")

    H = module.homography_from_points(src_pts, dst_pts)
    mapped = module.apply_homography_to_points(query_pts, H)
    mapped_points = [(float(p[0]), float(p[1])) for p in mapped]
    return H, mapped_points


def _render_mapped_point_image(
    points: List[Tuple[float, float]],
    output_path: str,
    field_width: float = 105.0,
    field_height: float = 68.0,
    scale: int = 12,
    outer_margin: int = 10,
    show_point_coords: bool = SHOW_POINT_COORDS,
) -> int:
    width_px = int(field_width * scale) + 2 * outer_margin
    height_px = int(field_height * scale) + 2 * outer_margin
    image = np.full((height_px, width_px, 3), (40, 90, 40), dtype=np.uint8)

    x0, y0 = outer_margin, outer_margin
    x1, y1 = outer_margin + int(field_width * scale), outer_margin + int(field_height * scale)

    def to_px(field_x: float, field_y: float) -> Tuple[int, int]:
        px = int(round(x0 + field_x * scale))
        py = int(round(y0 + (field_height - field_y) * scale))
        return px, py

    # Contorno externo do campo (105x68), com margem de 10px ao redor.
    cv2.rectangle(image, (x0, y0), (x1, y1), (255, 255, 255), 2)

    # Linhas internas solicitadas.
    internal_segments = [
        ((0.0, 53.25), (16.5, 53.25)),
        ((0.0, 14.75), (16.5, 14.75)),
        ((16.5, 53.25), (16.5, 14.75)),
        ((52.5, 0.0), (52.5, 68.0)),
        ((88.5, 53.25), (105.0, 53.25)),
        ((88.5, 14.75), (105.0, 14.75)),
        ((88.5, 53.25), (88.5, 14.75)),
    ]
    for (sx, sy), (ex, ey) in internal_segments:
        cv2.line(image, to_px(sx, sy), to_px(ex, ey), (255, 255, 255), 2)

    radius = 9.15

    # Circulo central completo.
    cv2.circle(
        image,
        to_px(52.5, 34.0),
        int(round(radius * scale)),
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Circulos das areas com recorte: mostrar apenas onde 16.5 <= x <= 88.5.
    def draw_clipped_circle(center_x: float, center_y: float) -> None:
        points: List[Tuple[int, int]] = []
        for deg in range(0, 361):
            theta = math.radians(float(deg))
            fx = center_x + radius * math.cos(theta)
            fy = center_y + radius * math.sin(theta)

            if 16.5 <= fx <= 88.5:
                points.append(to_px(fx, fy))
            elif len(points) >= 2:
                cv2.polylines(
                    image,
                    [np.array(points, dtype=np.int32)],
                    False,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                points = []
            else:
                points = []

        if len(points) >= 2:
            cv2.polylines(
                image,
                [np.array(points, dtype=np.int32)],
                False,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    draw_clipped_circle(11.0, 34.0)
    draw_clipped_circle(94.0, 34.0)

    inside_count = 0
    for x, y in points:
        x = float(x)
        y = float(y)
        inside = (0.0 <= x <= field_width) and (0.0 <= y <= field_height)
        if not inside:
            continue

        inside_count += 1
        px, py = to_px(x, y)
        cv2.circle(image, (px, py), 6, (0, 0, 255), -1)
        if show_point_coords:
            cv2.putText(
                image,
                f"({x:.2f}, {y:.2f})",
                (min(px + 10, width_px - 260), max(py - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

    cv2.imwrite(output_path, image)
    return inside_count


def _cross(ax: float, ay: float, bx: float, by: float) -> float:
    return ax * by - ay * bx


def _line_intersection_point(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    q1: Tuple[float, float],
    q2: Tuple[float, float],
    eps: float = 1e-9,
) -> Optional[Tuple[float, float]]:
    px, py = p1
    rx, ry = p2[0] - p1[0], p2[1] - p1[1]
    qx, qy = q1
    sx, sy = q2[0] - q1[0], q2[1] - q1[1]

    rxs = _cross(rx, ry, sx, sy)
    qmpx, qmpy = qx - px, qy - py
    qmpxr = _cross(qmpx, qmpy, rx, ry)

    if abs(rxs) <= eps and abs(qmpxr) <= eps:
        return ((p1[0] + q1[0]) / 2.0, (p1[1] + q1[1]) / 2.0)

    if abs(rxs) <= eps:
        return None

    t = _cross(qmpx, qmpy, sx, sy) / rxs
    return (px + t * rx, py + t * ry)


def _closest_point_on_line(
    p: Tuple[float, float],
    a: Tuple[float, float],
    b: Tuple[float, float],
) -> Tuple[Tuple[float, float], float]:
    ax, ay = a
    bx, by = b
    px, py = p

    abx, aby = bx - ax, by - ay
    ab2 = abx * abx + aby * aby
    if ab2 == 0:
        d = math.hypot(px - ax, py - ay)
        return (ax, ay), d

    t = ((px - ax) * abx + (py - ay) * aby) / ab2
    cx, cy = ax + t * abx, ay + t * aby
    d = math.hypot(px - cx, py - cy)
    return (cx, cy), d


def _closest_points_between_lines(
    a1: Tuple[float, float],
    a2: Tuple[float, float],
    b1: Tuple[float, float],
    b2: Tuple[float, float],
) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
    candidates = []

    cp, d = _closest_point_on_line(a1, b1, b2)
    candidates.append((a1, cp, d))

    cp, d = _closest_point_on_line(a2, b1, b2)
    candidates.append((a2, cp, d))

    cp, d = _closest_point_on_line(b1, a1, a2)
    candidates.append((cp, b1, d))

    cp, d = _closest_point_on_line(b2, a1, a2)
    candidates.append((cp, b2, d))

    return min(candidates, key=lambda item: item[2])


def _extract_segment(coords: Dict[str, Dict[str, int]]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    left = coords["left_point"]
    right = coords["right_point"]
    return (float(left["x"]), float(left["y"])), (float(right["x"]), float(right["y"]))


def _is_point_within_bounds(
    point: Tuple[float, float],
    x_min: float = BOUNDS_X_MIN,
    x_max: float = BOUNDS_X_MAX,
    y_min: float = BOUNDS_Y_MIN,
    y_max: float = BOUNDS_Y_MAX,
) -> bool:
    x, y = point
    return x_min <= x <= x_max and y_min <= y <= y_max


def _find_line_meetings(
    lines_result: Dict[str, Optional[Dict[str, Dict[str, int]]]],
    proximity_threshold_px: float = PROXIMITY_THRESHOLD_PX,
) -> List[Dict[str, object]]:
    valid_lines = [(name, coords) for name, coords in lines_result.items() if coords is not None]
    meetings: List[Dict[str, object]] = []

    for i in range(len(valid_lines)):
        line_a_name, line_a_coords = valid_lines[i]
        seg_a = _extract_segment(line_a_coords)
        for j in range(i + 1, len(valid_lines)):
            line_b_name, line_b_coords = valid_lines[j]
            seg_b = _extract_segment(line_b_coords)

            intersection = _line_intersection_point(seg_a[0], seg_a[1], seg_b[0], seg_b[1])
            if intersection is not None:
                if not _is_point_within_bounds(intersection):
                    continue
                meetings.append(
                    {
                        "line_a": line_a_name,
                        "line_b": line_b_name,
                        "point": (intersection[0], intersection[1]),
                        "type": "crossing",
                    }
                )
                continue

            p_a, p_b, distance = _closest_points_between_lines(seg_a[0], seg_a[1], seg_b[0], seg_b[1])
            if distance < proximity_threshold_px:
                midpoint = ((p_a[0] + p_b[0]) / 2.0, (p_a[1] + p_b[1]) / 2.0)
                meetings.append(
                    {
                        "line_a": line_a_name,
                        "line_b": line_b_name,
                        "point": (midpoint[0], midpoint[1]),
                        "type": "proximity",
                    }
                )

    return meetings


def main() -> None:
    with ThreadPoolExecutor(max_workers=2) as executor:
        fut_inference = executor.submit(
            _run_inference,
            IMAGE_PATH,
            MODEL_PATH,
            INFERENCE_OUTPUT_DIR,
            SPLIT_IN_FOUR,
        )
        fut_lines = executor.submit(_run_line_detection, IMAGE_PATH, LINES_OUTPUT_DIR)

        class_name, class_3_boxes = fut_inference.result()
        lines_result = fut_lines.result()

    print("Objetos da classe=3")
    if class_name is None:
        print("- Classe de id 3 nao existe no modelo.")
    elif not class_3_boxes:
        print(f"- Nenhum objeto detectado para a classe id=3 ({class_name}).")
    else:
        mean_bgr_per_box = _compute_box_mean_bgr(IMAGE_PATH, class_3_boxes)
        _save_class3_boxes_image(IMAGE_PATH, class_3_boxes, CLASS3_BOXES_OUTPUT_PATH)
        print(f"- Classe id=3 -> nome: {class_name}")
        print(f"- Imagem com boxes da classe 3 salva em: {CLASS3_BOXES_OUTPUT_PATH}")
        for i, (box, mean_bgr) in enumerate(zip(class_3_boxes, mean_bgr_per_box), start=1):
            print(
                f"  {i}. x_min={box['x_min']}, y_min={box['y_min']}, "
                f"x_max={box['x_max']}, y_max={box['y_max']}, "
                f"mean_bgr=({mean_bgr[0]:.1f}, {mean_bgr[1]:.1f}, {mean_bgr[2]:.1f})"
            )

    print("\nLinhas finais")
    for line_name, coords in lines_result.items():
        if coords is None:
            print(f"- {line_name}: nao encontrada")
            continue

        left = coords["left_point"]
        right = coords["right_point"]
        print(
            f"- {line_name}: "
            f"left=({left['x']}, {left['y']}), "
            f"right=({right['x']}, {right['y']})"
        )

    meetings = _find_line_meetings(lines_result, proximity_threshold_px=PROXIMITY_THRESHOLD_PX)
    print(f"\nPontos de encontro de linhas (limiar < {PROXIMITY_THRESHOLD_PX}px)")
    if not meetings:
        print("- Nenhum encontro encontrado.")
    else:
        for i, meeting in enumerate(meetings, start=1):
            x, y = meeting["point"]
            print(
                f"- {i}. {meeting['line_a']} x {meeting['line_b']}: "
                f"ponto=({x:.2f}, {y:.2f}) [{meeting['type']}]"
            )

    correspondences: List[Tuple[Tuple[float, float], Tuple[float, float], str]] = []
    crossings = [meeting for meeting in meetings if meeting["type"] == "crossing"]

    for meeting in crossings:
        line_a = str(meeting["line_a"])
        line_b = str(meeting["line_b"])
        key = f"{line_a} x {line_b}"
        reverse_key = f"{line_b} x {line_a}"

        dst = dict_pts.get(key)
        match_key = key
        if dst is None:
            dst = dict_pts.get(reverse_key)
            match_key = reverse_key
        if dst is None:
            continue

        src_pt = meeting["point"]
        correspondences.append((src_pt, dst, match_key))

    print("\nPares usados para homografia (cruzamentos -> dict_pts)")
    if not correspondences:
        print("- Nenhum par correspondente encontrado no dict_pts.")
        return

    for i, (src_pt, dst_pt, match_key) in enumerate(correspondences, start=1):
        print(
            f"- {i}. {match_key}: "
            f"src=({src_pt[0]:.2f}, {src_pt[1]:.2f}) -> dst=({dst_pt[0]:.2f}, {dst_pt[1]:.2f})"
        )

    if len(correspondences) < 4:
        print(f"- Homografia nao calculada: apenas {len(correspondences)} pares (minimo: 4).")
        return

    src_pts = [item[0] for item in correspondences]
    dst_pts = [item[1] for item in correspondences]
    box_bottom_center_pts: List[Tuple[float, float]] = []
    for box in class_3_boxes:
        x = (float(box["x_min"]) + float(box["x_max"])) / 2.0
        y = float(box["y_max"])
        box_bottom_center_pts.append((x, y))

    if not box_bottom_center_pts:
        print("- Nao ha bounding boxes da classe 3 para converter.")
        return

    try:
        H, mapped_points = _run_homography(src_pts, dst_pts, box_bottom_center_pts)
    except ValueError as exc:
        print(f"- Falha ao calcular homografia: {exc}")
        return

    print("\nHomografia estimada (imagem -> plano do campo)")
    print(H)
    print("\nPontos convertidos (bottom-center das boxes da classe 3)")
    for i, (src_pt, mapped_pt) in enumerate(zip(box_bottom_center_pts, mapped_points), start=1):
        print(
            f"- {i}. src=({src_pt[0]:.2f}, {src_pt[1]:.2f}) -> "
            f"dst=({mapped_pt[0]:.4f}, {mapped_pt[1]:.4f})"
        )

    inside_count = _render_mapped_point_image(mapped_points, HOMOGRAPHY_PLOT_PATH)
    print(
        "Imagem gerada com "
        f"{inside_count}/{len(mapped_points)} pontos dentro do retangulo 105x68: "
        f"{HOMOGRAPHY_PLOT_PATH}"
    )


if __name__ == "__main__":
    main()
