from pydantic import BaseModel


class InferenceConfig(BaseModel):
    image_width: int = 1200
    image_height: int = 1200
    radius: float = 1.2
    phi: float = 0.25 * 3.14
    n_images: int = 300
    distance: float = 1.5