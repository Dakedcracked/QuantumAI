QuantumAI Web App - Quick Deploy

Prereqs
- Docker and Docker Compose
- Models saved at models/saved/<brain|lung>_effresnet_vit_classifier.h5

Steps
1) Copy .env.example to .env and adjust values
2) docker compose up --build
3) Open http://localhost:8000/docs for API, http://localhost:5173 for frontend (after npm dev)

Frontend dev
- cd frontend && npm install && npm run dev

Notes
- Admin user is seeded on backend start (admin@quantumai.local / admin123). Change in .env.
