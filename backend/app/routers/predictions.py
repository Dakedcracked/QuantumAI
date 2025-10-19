from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.orm import Session
import shutil
import uuid
import json
from pathlib import Path

from ..db.database import get_db
from ..models.models import Prediction
from ..schemas.schemas import PredictionCreate, PredictionRead
from ..core.config import settings
from ..services.model_service import ModelService
from ..core.security import get_current_user
from ..models.models import User, UserRole

router = APIRouter()

uploads_dir = Path("uploads")
uploads_dir.mkdir(parents=True, exist_ok=True)

# Initialize service
model_service = ModelService(settings.brain_model_path, settings.lung_model_path)

@router.post("/", response_model=PredictionRead)
async def create_prediction(model_type: str, file: UploadFile = File(...), db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    if model_type not in {"brain", "lung"}:
        raise HTTPException(status_code=400, detail="model_type must be 'brain' or 'lung'")

    # Validate content type
    if file.content_type not in {"image/jpeg", "image/png", "image/jpg", "image/bmp", "image/tiff"}:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    # Save uploaded file with secure unique name
    ext = {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/bmp": ".bmp",
        "image/tiff": ".tiff",
    }.get(file.content_type, "")
    out_path = uploads_dir / f"{uuid.uuid4().hex}{ext}"
    with out_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        label, conf, raw = model_service.predict(model_type=model_type, image_path=str(out_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    pred = Prediction(
        user_id=current_user.id,
        model_type=model_type,
        image_path=str(out_path),
        predicted_label=label,
        confidence=conf,
        raw_scores=json.dumps(raw.tolist()),
    )
    db.add(pred)
    db.commit()
    db.refresh(pred)
    return pred

@router.get("/", response_model=list[PredictionRead])
async def list_predictions(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    q = db.query(Prediction)
    if current_user.role != UserRole.admin:
        q = q.filter(Prediction.user_id == current_user.id)
    return q.order_by(Prediction.created_at.desc()).all()
