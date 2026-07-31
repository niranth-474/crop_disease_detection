# Crop Leaf Disease Detection App

This repository now contains a full-stack application for crop leaf disease detection with:

- a FastAPI backend for inference
- a React + Vite + Tailwind frontend for image upload and prediction results
- a PyTorch CNN model loaded from the existing saved checkpoint

## Model assumptions

The current implementation is built around the model in [models/model.py](models/model.py) and expects:

- input shape: 3 x 256 x 256
- class count: 15
- weights file: [saved_models/best_model.pth](saved_models/best_model.pth)

The class labels are defined in [PlantVillage/class_names.json](PlantVillage/class_names.json).

## Backend setup

1. Open a terminal in the project root.
2. Create and activate a Python virtual environment.
3. Install dependencies:

   ```bash
   cd backend
   pip install -r requirements.txt
   ```

4. Start the API:

   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

5. The API will be available at http://localhost:8000.

### Backend environment variables

Create a [backend/.env](backend/.env) file if you need to override defaults:

```env
MODEL_PATH=../saved_models/best_model.pth
CLASS_NAMES_PATH=../PlantVillage/class_names.json
ALLOWED_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
```

## Free deployment with Render

This repository is now ready to deploy on Render for free.

1. Push the repo to GitHub.
2. Go to Render.com and connect the repository.
3. Add `render.yaml` at the repository root. Render will create both services automatically.
4. If the frontend or backend URLs differ, update the environment variables in Render:

```text
VITE_API_BASE_URL=https://crop-disease-backend.onrender.com/api
ALLOWED_ORIGINS=https://crop-disease-frontend.onrender.com
```

### What Render will run

- Frontend: builds from `frontend/`, publishes `frontend/dist`
- Backend: builds from `backend/Dockerfile`, runs FastAPI on port `8000`

### Alternative free option

If you want only the frontend deployed for free, you can use GitHub Pages or Vercel and point `VITE_API_BASE_URL` to your backend URL.

## Frontend setup

1. Install Node.js 18+ if needed.
2. Install dependencies:

   ```bash
   cd frontend
   npm install
   ```

3. Start the development server:

   ```bash
   npm run dev
   ```

4. Open the local URL shown by Vite, usually http://localhost:5173.

### Frontend environment variables

Create a [frontend/.env](frontend/.env) file with:

```env
VITE_API_BASE_URL=http://localhost:8000/api
```

## Project structure

- [backend/app](backend/app) — FastAPI app, routes, model loader, schemas, and utilities
- [frontend/src](frontend/src) — React app components and pages
- [models/model.py](models/model.py) — PyTorch CNN architecture
- [saved_models](saved_models) — pretrained weights and evaluation artifacts

## Notes

- The backend accepts a base64-encoded image and returns the top prediction plus the top 3 confidences.
- The frontend uses a drag-and-drop uploader and displays loading, preview, and error states.