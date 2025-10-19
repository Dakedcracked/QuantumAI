from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Float, Enum
from sqlalchemy.orm import relationship
from datetime import datetime
from ..db.database import Base
import enum

class UserRole(str, enum.Enum):
    admin = "admin"
    doctor = "doctor"

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    full_name = Column(String(255), nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role = Column(Enum(UserRole), default=UserRole.doctor, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    predictions = relationship("Prediction", back_populates="user")

class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    model_type = Column(String(50), nullable=False)  # brain|lung
    image_path = Column(Text, nullable=False)
    predicted_label = Column(String(255), nullable=False)
    confidence = Column(Float, nullable=False)
    raw_scores = Column(Text, nullable=True)  # JSON string of raw logits/softmax
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="predictions")
