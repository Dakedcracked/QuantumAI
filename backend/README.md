QuantumAI Backend (FastAPI)

Endpoints
- GET /health
- POST /api/auth/token (OAuth2 password flow)
- POST /api/predictions?model_type=brain|lung (multipart image upload)
- GET /api/predictions
- POST /api/users
- GET /api/users

Local run
- Create and fill .env (see .env.example)
- Start stack: docker compose up --build
- API docs: http://localhost:8000/docs
