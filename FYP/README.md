# Final-Year Project (FYP) - Main Application

This repository contains the full-stack application (frontend and backend) for the Multimodal Object Locator System.

## Tech Stack & Architecture
- **Backend:** FastAPI, Python, Supabase
- **Frontend:** HTML/CSS/JavaScript (served via local HTTP server)

## Running the Application

To run the full application locally, you will need **two separate terminal windows** (using Windows PowerShell).

### Step 1: Start the Backend Server
Open your first terminal window, navigate to the backend directory, and start the FastAPI development server:
```powershell
cd FYP/backend
uvicorn main:app --reload
```

### Step 2: Start the Frontend Server
Open a second terminal window, navigate to the frontend directory, and start a local HTTP server:
```powershell
cd FYP/frontend
python -m http.server 5500
```


