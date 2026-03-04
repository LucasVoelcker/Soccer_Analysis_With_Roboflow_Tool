from __future__ import annotations

import argparse
import importlib.util
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


MODEL_PATH = r"C:\Users\lucas\OneDrive\Documentos\Lucas\roboflow\treinamentos\best.pt"
VIDEO_PATH = r"C:\Users\lucas\Videos\Gravações de Tela\Gravação de Tela 2026-02-17 160338.mp4"
SPLIT_IN_FOUR = False
DETECTION_UPDATES_PER_SECOND = 2.0
PROXIMITY_THRESHOLD_PX = 0
SHOW_POINT_COORDS = False
HOMOGRAPHY_PANEL_HEIGHT = 320

OUTPUT_ROOT = Path(__file__).resolve().parent / "processed-video-output"
OUTPUT_VIDEO_PATH = OUTPUT_ROOT / "video-with-class3-boxes-and-homography.mp4"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Processa video e gera saida com boxes e homografia.")
    parser.add_argument(
        "--video-path",
        default=VIDEO_PATH,
        help="Caminho do video de entrada.",
    )
    parser.add_argument(
        "--output-video-path",
        default=str(OUTPUT_VIDEO_PATH),
        help="Caminho do video de saida.",
    )
    return parser.parse_args()


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _create_video_writer(
    output_path: Path,
    fps: float,
    width: int,
    height: int,
) -> tuple[cv2.VideoWriter, str]:
    codec_candidates = ["avc1", "H264", "mp4v"]
    for codec in codec_candidates:
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            (width, height),
        )
        if writer.isOpened():
            return writer, codec
        writer.release()

    raise RuntimeError(f"Nao foi possivel criar o video de saida: {output_path}")


def _build_correspondences(
    meetings: List[Dict[str, object]],
    dict_pts: Dict[str, Tuple[float, float]],
) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    correspondences: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    crossings = [meeting for meeting in meetings if meeting["type"] == "crossing"]

    for meeting in crossings:
        line_a = str(meeting["line_a"])
        line_b = str(meeting["line_b"])
        key = f"{line_a} x {line_b}"
        reverse_key = f"{line_b} x {line_a}"

        dst = dict_pts.get(key)
        if dst is None:
            dst = dict_pts.get(reverse_key)
        if dst is None:
            continue

        src = meeting["point"]
        correspondences.append((src, dst))

    return correspondences


def _run_detection_for_frame(
    run_detection_module,
    frame_path: str,
    inference_dir: Path,
    lines_dir: Path,
    homography_image_path: Path,
) -> tuple[Optional[str], List[Dict[str, int]], np.ndarray]:
    with ThreadPoolExecutor(max_workers=2) as executor:
        fut_inference = executor.submit(
            run_detection_module._run_inference,
            frame_path,
            MODEL_PATH,
            str(inference_dir),
            SPLIT_IN_FOUR,
        )
        fut_lines = executor.submit(
            run_detection_module._run_line_detection,
            frame_path,
            str(lines_dir),
        )
        class_name, class_3_boxes = fut_inference.result()
        lines_result = fut_lines.result()

    meetings = run_detection_module._find_line_meetings(
        lines_result,
        proximity_threshold_px=PROXIMITY_THRESHOLD_PX,
    )
    correspondences = _build_correspondences(meetings, run_detection_module.dict_pts)

    mapped_points: List[Tuple[float, float]] = []
    if len(correspondences) >= 4 and class_3_boxes:
        src_pts = [item[0] for item in correspondences]
        dst_pts = [item[1] for item in correspondences]
        query_pts = []
        for box in class_3_boxes:
            cx = (float(box["x_min"]) + float(box["x_max"])) / 2.0
            cy = float(box["y_max"])
            query_pts.append((cx, cy))

        if query_pts:
            try:
                _, mapped_points = run_detection_module._run_homography(
                    src_pts,
                    dst_pts,
                    query_pts,
                )
            except Exception:
                mapped_points = []

    run_detection_module._render_mapped_point_image(
        mapped_points,
        str(homography_image_path),
        show_point_coords=SHOW_POINT_COORDS,
    )

    homography_img = cv2.imread(str(homography_image_path))
    if homography_img is None:
        raise RuntimeError(
            f"Nao foi possivel carregar imagem de homografia: {homography_image_path}"
        )

    return class_name, class_3_boxes, homography_img


def _draw_class3_boxes(
    frame: np.ndarray,
    class_name: Optional[str],
    boxes: List[Dict[str, int]],
) -> None:
    if class_name is None or not boxes:
        return

    for i, box in enumerate(boxes, start=1):
        x_min = int(box["x_min"])
        y_min = int(box["y_min"])
        x_max = int(box["x_max"])
        y_max = int(box["y_max"])

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
        label = f"ID {i}"
        text_x = x_min
        text_y = max(20, y_min - 8)
        cv2.putText(
            frame,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )


def _compose_output_frame(
    top_frame: np.ndarray,
    homography_img: Optional[np.ndarray],
    panel_height: int,
) -> np.ndarray:
    h, w = top_frame.shape[:2]
    out = np.zeros((h + panel_height, w, 3), dtype=np.uint8)

    out[:h, :w] = top_frame
    out[h:, :] = (18, 18, 18)
    cv2.line(out, (0, h), (w - 1, h), (255, 255, 255), 1)
    cv2.putText(
        out,
        "Homography Point",
        (10, h + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if homography_img is None:
        return out

    panel_top = h + 40
    available_h = max(1, panel_height - 50)
    available_w = w
    src_h, src_w = homography_img.shape[:2]

    scale = min(available_w / src_w, available_h / src_h)
    resized_w = max(1, int(round(src_w * scale)))
    resized_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(homography_img, (resized_w, resized_h), interpolation=cv2.INTER_AREA)

    x0 = (w - resized_w) // 2
    y0 = panel_top + max(0, (available_h - resized_h) // 2)
    y1 = min(out.shape[0], y0 + resized_h)
    x1 = min(w, x0 + resized_w)
    out[y0:y1, x0:x1] = resized[: y1 - y0, : x1 - x0]
    return out


def main(video_path: str, output_video_path: str) -> None:
    base_dir = Path(__file__).resolve().parent
    run_detection_module = _load_module("run_detection_base", base_dir / "run-detection.py")

    output_video = Path(output_video_path).resolve()
    output_root = output_video.parent
    output_root.mkdir(parents=True, exist_ok=True)
    per_update_dir = output_root / "frames-processed-2fps"
    per_update_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Nao foi possivel abrir o video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        cap.release()
        raise RuntimeError("FPS invalido no video.")

    update_every_frames = max(1, int(round(fps / DETECTION_UPDATES_PER_SECOND)))

    writer, used_codec = _create_video_writer(
        output_video,
        fps,
        width,
        height + HOMOGRAPHY_PANEL_HEIGHT,
    )

    frame_idx = 0
    sample_idx = 0
    last_class_name: Optional[str] = None
    last_boxes: List[Dict[str, int]] = []
    last_homography_img: Optional[np.ndarray] = None
    updates_ok = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            should_update = (frame_idx % update_every_frames) == 0

            if should_update:
                frame_dir = per_update_dir / f"frame-{sample_idx:06d}"
                frame_dir.mkdir(parents=True, exist_ok=True)
                frame_path = frame_dir / "input-frame.jpg"
                cv2.imwrite(str(frame_path), frame)

                try:
                    class_name, class_3_boxes, homography_img = _run_detection_for_frame(
                        run_detection_module,
                        str(frame_path),
                        frame_dir / "inference",
                        frame_dir / "lines",
                        frame_dir / "homography-point.png",
                    )
                    last_class_name = class_name
                    last_boxes = class_3_boxes
                    last_homography_img = homography_img
                    updates_ok += 1
                    print(
                        f"[OK] sample={sample_idx} frame_idx={frame_idx} "
                        f"class3={len(last_boxes)}"
                    )
                except Exception as exc:
                    print(
                        f"[ERRO] sample={sample_idx} frame_idx={frame_idx}: {exc}"
                    )

                sample_idx += 1

            output_frame = frame.copy()
            _draw_class3_boxes(output_frame, last_class_name, last_boxes)
            final_frame = _compose_output_frame(
                output_frame,
                last_homography_img,
                HOMOGRAPHY_PANEL_HEIGHT,
            )
            writer.write(final_frame)
            frame_idx += 1

    finally:
        cap.release()
        writer.release()

    print(f"\nFrames no video: {frame_count}")
    print(f"FPS video original: {fps:.3f}")
    print(f"Atualizacao de deteccao: {DETECTION_UPDATES_PER_SECOND:.2f} FPS")
    print(f"Intervalo de atualizacao: {update_every_frames} frames")
    print(f"Atualizacoes com sucesso: {updates_ok}")
    print(f"Codec usado na saida: {used_codec}")
    print(f"Saida: {output_video}")


if __name__ == "__main__":
    args = _parse_args()
    main(video_path=args.video_path, output_video_path=args.output_video_path)
