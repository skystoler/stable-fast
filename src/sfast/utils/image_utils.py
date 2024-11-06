from PIL import Image, ImageOps
from typing import Union, Optional, Callable, List
import requests
import os
import numpy as np
import torch
import io

PIL_INTERPOLATION = {
    "linear": Image.Resampling.BILINEAR,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
    "nearest": Image.Resampling.NEAREST,
}
    
def load_image(
    image: Union[str, Image.Image], convert_method: Optional[Callable[[Image.Image], Image.Image]] = None
) -> Image.Image:
    if isinstance(image, str):
        if image.startswith("http://") or image.startswith("https://"):
            image = Image.open(requests.get(image).raw)
        elif os.path.isfile(image):
            image = Image.open(image)
        else:
            raise ValueError(
                f"Incorrect path or URL. URLs must start with `http://` or `https://`, and {image} is not a valid path."
            )
    elif isinstance(image, Image.Image):
        image = image
    else:
        raise ValueError(
            "Incorrect format used for the image. Should be a URL linking to an image, a local path, or a PIL image."
        )

    image = ImageOps.exif_transpose(image)

    if convert_method is not None:
        image = convert_method(image)
    else:
        image = image.convert("RGB")

    return image


def numpy_to_pil(images: np.ndarray) -> List[Image.Image]:
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def pil_to_numpy(images: Union[List[Image.Image], Image.Image]) -> np.ndarray:
    if not isinstance(images, list):
        images = [images]
    images = [np.array(image).astype(np.float32) / 255.0 for image in images]
    images = np.stack(images, axis=0)

    return images


def numpy_to_pt(images: np.ndarray) -> torch.Tensor:
    if images.ndim == 3:
        images = images[..., None]

    images = torch.from_numpy(images.transpose(0, 3, 1, 2))
    return images


def resize(
    self,
    image: Union[Image.Image, np.ndarray, torch.Tensor],
    height: int,
    width: int,
    resize_mode: str = "default",  # "default", "fill", "crop"
) -> Union[Image.Image, np.ndarray, torch.Tensor]:
    if resize_mode != "default" and not isinstance(image, Image.Image):
        raise ValueError(f"Only PIL image input is supported for resize_mode {resize_mode}")
    if isinstance(image, Image.Image):
        if resize_mode == "default":
            image = image.resize((width, height), resample=PIL_INTERPOLATION["bicubic"])
        elif resize_mode == "fill":
            image = self._resize_and_fill(image, width, height)
        elif resize_mode == "crop":
            image = self._resize_and_crop(image, width, height)
        else:
            raise ValueError(f"resize_mode {resize_mode} is not supported")

    elif isinstance(image, torch.Tensor):
        image = torch.nn.functional.interpolate(
            image,
            size=(height, width),
        )
    elif isinstance(image, np.ndarray):
        image = self.numpy_to_pt(image)
        image = torch.nn.functional.interpolate(
            image,
            size=(height, width),
        )
        image = self.pt_to_numpy(image)
    return image