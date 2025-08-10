import os
from pathlib import Path
from typing import Dict, Any, List
import tempfile
import shutil
from contextlib import asynccontextmanager
import urllib.request
import logging

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the fix
try:
    import yolo_loader_fix
except ImportError:
    logger.warning("yolo_loader_fix not found")

import cv2
from basketball_referee import ImprovedFreeThrowScorer, CVATDatasetConverter, FreeThrowModelTrainer

# Configuration for Render
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "best.pt")
MODEL_URL = os.getenv("MODEL_URL", None)  # Set this in Render dashboard
PORT = int(os.getenv("PORT", 10000))

# Global variables
scorer_instance = None


def download_model():
    """Download model from URL if not present"""
    if os.path.exists(MODEL_PATH):
        logger.info(f"Model already exists at {MODEL_PATH}")
        return True

    if not MODEL_URL:
        logger.warning("Model not found locally and MODEL_URL not set")
        logger.warning("Please set MODEL_URL environment variable in Render dashboard")
        return False

    try:
        logger.info(f"Downloading model from {MODEL_URL}")
        os.makedirs(MODEL_DIR, exist_ok=True)

        # Download with progress
        def download_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded * 100 / total_size, 100)
            logger.info(f"Download progress: {percent:.1f}%")

        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH, reporthook=download_progress)
        logger.info("Model downloaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    global scorer_instance

    logger.info("\n" + "=" * 60)
    logger.info("AI BASKETBALL REFEREE API STARTING (RENDER)")
    logger.info("=" * 60)
    logger.info(f"Port: {PORT}")
    logger.info(f"Model path: {MODEL_PATH}")

    # Download model if needed
    if download_model():
        try:
            logger.info("Loading model...")
            scorer_instance = ImprovedFreeThrowScorer(MODEL_PATH)
            logger.info("✅ Model loaded successfully!")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            import traceback
            traceback.print_exc()
    else:
        logger.warning("⚠️  Running without model - only training endpoint available")

    logger.info("=" * 60 + "\n")

    yield

    logger.info("\nShutting down API...")


# Create FastAPI app
app = FastAPI(
    title="AI Basketball Referee API",
    description="Automated basketball free throw scoring using AI",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with simple UI"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Basketball Referee API</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                max-width: 800px; 
                margin: 0 auto; 
                padding: 20px;
                background: #f5f5f5;
            }
            .container {
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 { color: #2196F3; }
            .status { 
                padding: 10px; 
                background: #e3f2fd; 
                border-radius: 5px; 
                margin: 20px 0;
            }
            .endpoint { 
                background: #f5f5f5; 
                padding: 10px; 
                margin: 10px 0; 
                border-radius: 5px;
                font-family: monospace;
            }
            a { color: #2196F3; text-decoration: none; }
            a:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🏀 AI Basketball Referee API</h1>
            <p>Welcome to the Basketball Referee API deployed on Render!</p>

            <div class="status">
                <strong>Status:</strong> API is running<br>
                <strong>Model:</strong> """ + ("Loaded ✅" if scorer_instance else "Not loaded ❌") + """
            </div>

            <h2>Available Endpoints:</h2>
            <div class="endpoint">GET /</div>
            <div class="endpoint">GET /health</div>
            <div class="endpoint">GET /status</div>
            <div class="endpoint">POST /score_video/</div>
            <div class="endpoint">GET /docs</div>

            <p style="margin-top: 30px;">
                <a href="/docs">📚 Interactive API Documentation</a>
            </p>

            """ + ("" if scorer_instance else """
            <div style="background: #ffebee; padding: 20px; border-radius: 5px; margin-top: 20px;">
                <strong>⚠️ Model Not Loaded</strong><br>
                Please set the MODEL_URL environment variable in Render dashboard to your model's download URL.
            </div>
            """) + """
        </div>
    </body>
    </html>
    """


@app.get("/health")
async def health():
    """Health check endpoint for Render"""
    return {"status": "healthy", "model_loaded": scorer_instance is not None}


@app.get("/status")
async def status():
    """Detailed status endpoint"""
    return {
        "status": "online",
        "model_loaded": scorer_instance is not None,
        "model_path": MODEL_PATH,
        "model_exists": os.path.exists(MODEL_PATH),
        "environment": "render",
        "port": PORT
    }


@app.post("/score_video/")
async def score_video(video_file: UploadFile = File(...)) -> Dict[str, Any]:
    """Analyze a basketball video to detect and score free throws."""

    if scorer_instance is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please set MODEL_URL environment variable."
        )

    # Check file size (Render has memory limits)
    contents = await video_file.read()
    size_mb = len(contents) / (1024 * 1024)

    if size_mb > 50:  # Limit to 50MB for Render free tier
        raise HTTPException(
            status_code=413,
            detail="Video too large. Maximum size is 50MB on Render free tier."
        )

    logger.info(f"Processing video: {video_file.filename} ({size_mb:.1f}MB)")

    # Process video
    with tempfile.TemporaryDirectory() as temp_dir:
        video_path = Path(temp_dir) / video_file.filename

        with open(video_path, "wb") as f:
            f.write(contents)

        # Reset scorer
        scorer_instance.made_shots = 0
        scorer_instance.missed_shots = 0
        scorer_instance.shot_attempts = 0
        scorer_instance.shot_tracker.reset()

        # Process video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file")

        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Process every nth frame to speed up on Render
        frame_skip = 2  # Process every 2nd frame

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Skip frames
            if frame_count % frame_skip != 0:
                continue

            # Run detection
            detections = scorer_instance.detect_objects(frame)
            hoop_info = scorer_instance.update_hoop_position(detections)
            ball_info = scorer_instance.find_ball(detections)
            player_bboxes = scorer_instance.find_players(detections)

            # Update shot tracking
            old_phase = scorer_instance.shot_tracker.shot_phase
            result = scorer_instance.shot_tracker.update(ball_info, hoop_info, player_bboxes, False)

            # Count attempts
            if old_phase == 'idle' and scorer_instance.shot_tracker.shot_phase == 'rising':
                scorer_instance.shot_attempts += 1

            # Count results
            if result == 'score':
                scorer_instance.made_shots += 1
                scorer_instance.shot_tracker.reset()
            elif result == 'miss':
                scorer_instance.missed_shots += 1
                scorer_instance.shot_tracker.reset()

        cap.release()

        # Calculate accuracy
        accuracy = 0.0
        if scorer_instance.shot_attempts > 0:
            accuracy = round((scorer_instance.made_shots / scorer_instance.shot_attempts) * 100, 1)

        return {
            "made_shots": scorer_instance.made_shots,
            "missed_shots": scorer_instance.missed_shots,
            "total_attempts": scorer_instance.shot_attempts,
            "accuracy_percentage": accuracy,
            "frames_processed": frame_count // frame_skip,
            "video_info": {
                "filename": video_file.filename,
                "size_mb": round(size_mb, 2),
                "duration_seconds": round(total_frames / fps, 1) if fps > 0 else 0,
                "fps": round(fps, 2)
            }
        }


if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting server on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)