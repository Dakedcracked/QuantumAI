from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import auth, predictions, users, pages
from .db.database import Base, engine, SessionLocal
from .models.models import User, UserRole
from .core.security import get_password_hash
from .core.config import settings

app = FastAPI(title="QuantumAI Medical Platform", version="0.1.0")

# CORS (adjust origins via env in production)
origins = [o.strip() for o in settings.cors_origins.split(",")] if settings.cors_origins else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(predictions.router, prefix="/api/predictions", tags=["predictions"])
app.include_router(users.router, prefix="/api/users", tags=["users"])
app.include_router(pages.router, tags=["pages"])  # public pages like about/investors


@app.get("/health")
def health():
    return {"status": "ok"}

@app.on_event("startup")
def on_startup():
    # Create tables
    Base.metadata.create_all(bind=engine)
    # Seed admin user if not exists
    db = SessionLocal()
    try:
        admin = db.query(User).filter(User.email == settings.admin_email).first()
        if not admin:
            admin = User(
                email=settings.admin_email,
                full_name="Administrator",
                hashed_password=get_password_hash(settings.admin_password),
                role=UserRole.admin,
            )
            db.add(admin)
            db.commit()
    finally:
        db.close()
