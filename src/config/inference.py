from pydantic import BaseModel
from typing import Optional


class InferenceConfig(BaseModel):
    image_width: int = 1200
    image_height: int = 1200
    radius: float = 1.2
    phi: float = 0.25 * 3.14
    n_images: int = 150
    distance: float = 1.5
    conditional_generation: bool = False
    show_steps: bool = False
    background_color: str = "white"
    inference_steps: int = 1000

    class Condition(BaseModel):
        label_text: str = "A yellow chair"
        input_image_path: Optional[str] = None

    class Video(BaseModel):
        frame_rate: int = 30
        n_videos: int = 5

    condition: Condition = Condition()
    video: Video = Video()
