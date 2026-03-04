from pathlib import Path
from types import MethodType
from typing import Dict, List, Tuple
import re

import cv2
from ultralytics import YOLO


class SplitYOLOv8Inference:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        self.class_names = self._normalize_class_names(self.model.names)
        self.class_method_map: Dict[str, str] = {}
        self.last_boxes_by_class: Dict[str, List[Dict[str, int]]] = {
            class_name: [] for class_name in self.class_names.values()
        }
        self.last_all_boxes: List[Dict[str, int]] = []
        self._register_class_methods()

    @staticmethod
    def _normalize_class_names(names) -> Dict[int, str]:
        if isinstance(names, dict):
            return {int(class_id): str(class_name) for class_id, class_name in names.items()}
        if isinstance(names, list):
            return {class_id: str(class_name) for class_id, class_name in enumerate(names)}
        return {}

    @staticmethod
    def _sanitize_class_name(class_name: str) -> str:
        safe_name = re.sub(r"[^0-9a-zA-Z]+", "_", class_name.strip().lower()).strip("_")
        if not safe_name:
            safe_name = "class_name"
        if safe_name[0].isdigit():
            safe_name = f"class_{safe_name}"
        return safe_name

    def _register_class_methods(self) -> None:
        for class_name in self.class_names.values():
            method_name = f"get_{self._sanitize_class_name(class_name)}_boxes"
            self.class_method_map[class_name] = method_name

            if hasattr(self, method_name):
                continue

            def _getter(instance, target_class=class_name):
                return instance.get_boxes_by_class(target_class)

            setattr(self, method_name, MethodType(_getter, self))

    @staticmethod
    def _build_quadrants(image_height: int, image_width: int) -> List[Tuple[int, int, int, int]]:
        half_height = image_height // 2
        half_width = image_width // 2
        return [
            (0, half_height, 0, half_width),
            (0, half_height, half_width, image_width),
            (half_height, image_height, 0, half_width),
            (half_height, image_height, half_width, image_width),
        ]

    def run_inference(
        self,
        image_path: str,
        split_in_four: bool = True,
        save_images: bool = True,
        output_dir: str = "inference_splits",
    ) -> Dict[str, List[Dict[str, int]]]:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found or unreadable: {image_path}")

        image_height, image_width = image.shape[:2]
        if split_in_four:
            regions = self._build_quadrants(image_height, image_width)
        else:
            regions = [(0, image_height, 0, image_width)]

        boxes_by_class: Dict[str, List[Dict[str, int]]] = {
            class_name: [] for class_name in self.class_names.values()
        }
        all_boxes: List[Dict[str, int]] = []

        output_path = Path(output_dir)
        if save_images:
            output_path.mkdir(parents=True, exist_ok=True)

        for region_index, (y1, y2, x1, x2) in enumerate(regions):
            crop = image[y1:y2, x1:x2]
            # AQUI FAZ A INFERENCIA
            result = self.model(crop, conf=0.5)[0]

            if save_images:
                crop_with_boxes = result.plot(labels=False)
                if split_in_four:
                    cv2.imwrite(str(output_path / f"split_{region_index}.jpg"), crop_with_boxes)
                else:
                    cv2.imwrite(str(output_path / "full_image.jpg"), crop_with_boxes)

            if result.boxes is None or len(result.boxes) == 0:
                continue

            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            boxes_cls = result.boxes.cls.cpu().numpy().astype(int)

            for box, class_id in zip(boxes_xyxy, boxes_cls):
                x_min, y_min, x_max, y_max = box

                coordinates = {
                    "x_min": int(x_min + x1),
                    "x_max": int(x_max + x1),
                    "y_min": int(y_min + y1),
                    "y_max": int(y_max + y1),
                }

                class_name = self.class_names.get(int(class_id), f"class_{int(class_id)}")
                if class_name not in boxes_by_class:
                    boxes_by_class[class_name] = []

                boxes_by_class[class_name].append(coordinates)
                all_boxes.append({"class_name": class_name, **coordinates})

        if save_images:
            final_image = image.copy()
            for box_data in all_boxes:
                cv2.rectangle(
                    final_image,
                    (box_data["x_min"], box_data["y_min"]),
                    (box_data["x_max"], box_data["y_max"]),
                    (0, 255, 0),
                    2,
                )
            cv2.imwrite(str(output_path / "final_combined.jpg"), final_image)

        self.last_boxes_by_class = boxes_by_class
        self.last_all_boxes = all_boxes
        return boxes_by_class

    def get_boxes_by_class(self, class_name: str) -> List[Dict[str, int]]:
        return [box.copy() for box in self.last_boxes_by_class.get(class_name, [])]

    def get_all_boxes(self) -> List[Dict[str, int]]:
        return [box.copy() for box in self.last_all_boxes]


MODEL_PATH = r"C:\Users\lucas\OneDrive\Documentos\Lucas\roboflow\treinamentos\best.pt"
IMAGE_PATH = r"C:\Users\lucas\OneDrive\Documentos\Lucas\roboflow\test-full-process\160338_mp4-0008_jpg.rf.PycGYxW94XpQmfF9EGor.jpg"
OUTPUT_DIR = "inference_splits--" + IMAGE_PATH[-8:-4]


if __name__ == "__main__":
    detector = SplitYOLOv8Inference(MODEL_PATH)
    detector.run_inference(
        IMAGE_PATH,
        split_in_four=False,
        save_images=True,
        output_dir=OUTPUT_DIR,
    )

    print("Inferencia finalizada.")
    print(f"Resultados salvos em: {OUTPUT_DIR}")

    print("Metodos de acesso por classe:")
    for class_name, method_name in detector.class_method_map.items():
        print(f"- {class_name}: detector.{method_name}()")
