from fastapi import APIRouter

router = APIRouter()

@router.get("/api/about")
def about():
    return {
        "name": "QuantumAI",
        "mission": "Advance medical diagnostics with hybrid deep learning models.",
        "vision": "Accessible, accurate cancer screening for all.",
        "team": [
            {"name": "Founding Scientist", "bio": "Expert in medical AI."},
            {"name": "CTO", "bio": "Scaled ML systems to production."},
        ],
        "investors": [
            {"name": "Healthcare Ventures", "focus": "AI in medicine"},
            {"name": "DeepTech Fund", "focus": "AI-first companies"}
        ],
    }

@router.get("/")
def root():
    return {"message": "Welcome to QuantumAI"}
