import os
import shutil
import numpy as np
import cv2

from PySide6.QtGui import QImageReader, QImage
from PySide6.QtCore import QSize

class Detector:
    def __init__(self, model, conf_threshold=0.25, max_dim=1920):
        self.model = model
        self.conf_threshold = conf_threshold
        self.max_dim = max_dim

    def set_conf(self, value):
        self.conf_threshold = value

    def process_images(self, images, results_dir, progress_callback=None, stop_flag=None):
        processed_paths = []
        per_image_detections = []
        total_detections = 0

        if os.path.exists(results_dir):
            shutil.rmtree(results_dir)
        os.makedirs(results_dir)

        for idx, path in enumerate(images):

            if stop_flag and stop_flag():
                break

            reader = QImageReader(path)
            reader.setAutoTransform(True)
            size = reader.size()

            if not size.isValid():
                continue

            if max(size.width(), size.height()) > self.max_dim:
                scale = self.max_dim / max(size.width(), size.height())
                reader.setScaledSize(QSize(
                    int(size.width() * scale),
                    int(size.height() * scale)
                ))

            image = reader.read()
            if image.isNull():
                continue

            image = image.convertToFormat(QImage.Format_RGB888)

            w, h = image.width(), image.height()
            ptr = image.bits()
            arr = np.frombuffer(ptr, dtype=np.uint8).copy()

            img = arr.reshape(h, image.bytesPerLine(), 1)[:, :w*3].reshape(h, w, 3)
            img = np.ascontiguousarray(img)

            image_detections = 0

            results = self.model(img, conf=self.conf_threshold, verbose=False)

            for r in results:
                if r.boxes is None:
                    continue

                for box in r.boxes:
                    total_detections += 1
                    image_detections += 1

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])

                    label = f"{self.model.names[cls]} {conf:.2f}"

                    if conf > 0.75:
                        color = (255, 0, 0)
                    elif conf > 0.45:
                        color = (255, 255, 0)
                    else:
                        color = (0, 255, 0)

                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            per_image_detections.append(image_detections)

            save_path = os.path.join(results_dir, f"result_{idx:04d}.jpg")

            cv2.imwrite(
                save_path,
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            )

            processed_paths.append(save_path)

            if progress_callback:
                progress_callback(idx + 1, len(images), total_detections)

        return processed_paths, per_image_detections, total_detections
    


