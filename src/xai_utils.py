
import numpy as np
import cv2
import pydicom
import torch

from pathlib import Path
from torchvision import transforms, models


IMAGE_WIDTH = 384
IMAGE_HEIGHT = 320

BLUR_LEVELS = {
    "Original": 0.0,
    "Mild Blur": 0.5,
    "Moderate Blur": 1.5,
    "Severe Blur": 3.0
}

NOISE_LEVELS = {
    "Original": 0.0,
    "Mild Noise": 0.02,
    "Moderate Noise": 0.05,
    "Severe Noise": 0.10
}


def load_dicom(path):
    ds = pydicom.dcmread(str(path))
    image = ds.pixel_array.astype(np.float32)

    if ds.PhotometricInterpretation == "MONOCHROME1":
        image = image.max() - image

    min_val = image.min()
    max_val = image.max()

    if max_val > min_val:
        image = (image - min_val) / (max_val - min_val)
    else:
        image = np.zeros_like(image)

    return image


class MedicalImageTransform:
    def __init__(self):
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __call__(self, image):
        image = cv2.resize(
            np.asarray(image, dtype=np.float32),
            (IMAGE_WIDTH, IMAGE_HEIGHT),
            interpolation=cv2.INTER_AREA
        )

        image = np.stack([image, image, image], axis=0)

        return self.normalize(
            torch.from_numpy(image).float()
        )


medical_transform = MedicalImageTransform()


def prepare_corrupted_image(image):
    # Expects a resized grayscale image in [0, 1]
    rgb = np.stack([image, image, image], axis=0)

    tensor = torch.from_numpy(
        rgb.astype(np.float32)
    )

    return medical_transform.normalize(tensor)


def resize_image(image):
    return cv2.resize(
        np.asarray(image, dtype=np.float32),
        (IMAGE_WIDTH, IMAGE_HEIGHT),
        interpolation=cv2.INTER_AREA
    )


def apply_gaussian_blur(image, sigma):
    if sigma == 0:
        return image.copy()

    return cv2.GaussianBlur(
        image,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma
    )


def apply_gaussian_noise(image, sigma, seed=42):
    if sigma == 0:
        return image.copy()

    rng = np.random.default_rng(seed)

    noise = rng.normal(
        loc=0.0,
        scale=sigma,
        size=image.shape
    )

    noisy_image = np.clip(
        image + noise,
        0.0,
        1.0
    )

    return noisy_image.astype(np.float32)


def build_opacity_lookup(annotation_records):
    # Final adjudicated lung-opacity boxes only
    lookup = {}

    for record in annotation_records:
        if (
            record.get("labelId") == "L_v8n"
            and isinstance(record.get("data"), dict)
        ):
            uid = str(record["SOPInstanceUID"])
            lookup.setdefault(uid, []).append(record)

    return lookup


def calculate_alignment(cam_map, boxes):
    h, w = cam_map.shape
    mask = np.zeros((h, w), dtype=bool)

    for record in boxes:
        box = record["data"]

        sx = w / record["width"]
        sy = h / record["height"]

        x1 = int(np.floor(box["x"] * sx))
        y1 = int(np.floor(box["y"] * sy))
        x2 = int(np.ceil((box["x"] + box["width"]) * sx))
        y2 = int(np.ceil((box["y"] + box["height"]) * sy))

        x1, x2 = np.clip([x1, x2], 0, w)
        y1, y2 = np.clip([y1, y2], 0, h)

        mask[y1:y2, x1:x2] = True

    total = cam_map.sum()

    return (
        float(cam_map[mask].sum() / total)
        if total > 0 else np.nan
    )


def load_trained_model(checkpoint_path, device):
    model = models.resnet50(
        weights=None,
        num_classes=1
    )

    # Only load checkpoints you trust
    checkpoint = torch.load(
        str(checkpoint_path),
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    return model.to(device).eval()
