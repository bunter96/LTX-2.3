from pydantic import BaseModel, Field


class TextToVideoRequest(BaseModel):
    prompt: str
    seed: int = 10
    height: int = Field(default=1024, description="Must be divisible by 64")
    width: int = Field(default=1536, description="Must be divisible by 64")
    num_frames: int = Field(default=121, description="Number of frames (8k+1 format recommended, e.g. 97, 121, 193)")
    frame_rate: float = 24.0
    enhance_prompt: bool = False


class ImageConditioningItem(BaseModel):
    image_b64: str = Field(description="Base64-encoded image (JPEG or PNG)")
    frame_idx: int = Field(default=0, description="Frame index to condition on")
    strength: float = Field(default=1.0, ge=0.0, le=1.0)
    crf: int = Field(default=33, description="Compression quality for encoding")


class ImageToVideoRequest(BaseModel):
    prompt: str
    images: list[ImageConditioningItem] = Field(min_length=1)
    seed: int = 10
    height: int = Field(default=1024, description="Must be divisible by 64")
    width: int = Field(default=1536, description="Must be divisible by 64")
    num_frames: int = Field(default=121)
    frame_rate: float = 24.0
    enhance_prompt: bool = False


class JobResponse(BaseModel):
    job_id: str
    status: str
