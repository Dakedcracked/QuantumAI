from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import datetime

class UserCreate(BaseModel):
    email: EmailStr
    full_name: str
    password: str
    role: str = Field(default="doctor")

class UserRead(BaseModel):
    id: int
    email: EmailStr
    full_name: str
    role: str
    created_at: datetime
    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class PredictionCreate(BaseModel):
    model_type: str  # brain|lung

class PredictionRead(BaseModel):
    id: int
    user_id: int
    model_type: str
    image_path: str
    predicted_label: str
    confidence: float
    raw_scores: Optional[str]
    created_at: datetime
    class Config:
        from_attributes = True
