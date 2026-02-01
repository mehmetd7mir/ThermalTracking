"""
REST API Server for ThermalTracking
-------------------------------------
FastAPI based API for integration with other systems.

Endpoints:
    POST /detect - detect objects in image
    POST /stream/start - start video stream processing
    GET  /tracks - get active tracks
    GET  /stats - get statistics
    
Run with:
    uvicorn api_server:app --reload
    
Author: Mehmet Demir
"""

import io
import time
import base64
import uuid
from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager

import numpy as np


# try import fastapi
try:
    from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Query
    from fastapi.responses import JSONResponse, StreamingResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("FastAPI not installed. Run: pip install fastapi uvicorn python-multipart")


# global model instance
_model = None
_tracker = None


class DetectionResult(BaseModel):
    """Detection result model"""
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]  # x1, y1, x2, y2


class DetectionResponse(BaseModel):
    """Response for detection endpoint"""
    success: bool
    detections: List[DetectionResult]
    processing_time_ms: float
    image_size: List[int]


class StreamConfig(BaseModel):
    """Configuration for video stream"""
    source: str  # video path or rtsp url
    confidence: float = 0.25
    save_output: bool = False


class StreamStatus(BaseModel):
    """Status of running stream"""
    stream_id: str
    status: str  # running, stopped, error
    frames_processed: int
    fps: float
    start_time: str
    detections_total: int


# pydantic models for requests
class ImageRequest(BaseModel):
    image_base64: str
    confidence: float = 0.25


# store active streams
active_streams: Dict[str, Dict] = {}


def load_model(weights: str = "best.pt"):
    """Load YOLO model."""
    global _model
    if _model is None:
        from ultralytics import YOLO
        if Path(weights).exists():
            _model = YOLO(weights)
        else:
            _model = YOLO("yolov8n.pt")
    return _model


# class name mapping
CLASS_NAMES = {0: 'bird', 1: 'drone', 2: 'helicopter', 3: 'plane'}


if FASTAPI_AVAILABLE:
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # startup
        load_model()
        print("Model loaded")
        yield
        # shutdown
        print("Shutting down")
    
    # create app
    app = FastAPI(
        title="ThermalTracking API",
        description="REST API for thermal object detection and tracking",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # add cors
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.get("/")
    async def root():
        """Health check endpoint."""
        return {
            "name": "ThermalTracking API",
            "status": "running",
            "version": "1.0.0"
        }
    
    @app.get("/health")
    async def health():
        """Health check."""
        return {"status": "healthy"}
    
    @app.post("/detect", response_model=DetectionResponse)
    async def detect_image(file: UploadFile = File(...), confidence: float = 0.25):
        """
        Detect objects in uploaded image.
        
        Upload image file and get detection results.
        """
        model = load_model()
        
        # read image
        contents = await file.read()
        
        # convert to numpy array
        import cv2
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Could not read image")
        
        # run detection
        start_time = time.time()
        results = model.predict(image, conf=confidence, verbose=False)
        processing_time = (time.time() - start_time) * 1000
        
        # parse results
        detections = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                det = DetectionResult(
                    class_id=int(box.cls),
                    class_name=CLASS_NAMES.get(int(box.cls), "unknown"),
                    confidence=float(box.conf),
                    bbox=box.xyxy[0].tolist()
                )
                detections.append(det)
        
        return DetectionResponse(
            success=True,
            detections=detections,
            processing_time_ms=processing_time,
            image_size=[image.shape[1], image.shape[0]]
        )
    
    @app.post("/detect/base64", response_model=DetectionResponse)
    async def detect_base64(request: ImageRequest):
        """
        Detect objects from base64 encoded image.
        
        Send image as base64 string in JSON body.
        """
        model = load_model()
        
        # decode base64
        try:
            image_bytes = base64.b64decode(request.image_base64)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 image")
        
        # convert to numpy
        import cv2
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        
        # detect
        start_time = time.time()
        results = model.predict(image, conf=request.confidence, verbose=False)
        processing_time = (time.time() - start_time) * 1000
        
        # parse
        detections = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                det = DetectionResult(
                    class_id=int(box.cls),
                    class_name=CLASS_NAMES.get(int(box.cls), "unknown"),
                    confidence=float(box.conf),
                    bbox=box.xyxy[0].tolist()
                )
                detections.append(det)
        
        return DetectionResponse(
            success=True,
            detections=detections,
            processing_time_ms=processing_time,
            image_size=[image.shape[1], image.shape[0]]
        )
    
    @app.post("/detect/annotated")
    async def detect_and_annotate(
        file: UploadFile = File(...),
        confidence: float = 0.25
    ):
        """
        Detect and return annotated image.
        
        Returns image with bounding boxes drawn.
        """
        model = load_model()
        
        # read image
        import cv2
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image")
        
        # detect
        results = model.predict(image, conf=confidence, verbose=False)
        
        # annotate
        annotated = results[0].plot()
        
        # encode to bytes
        _, buffer = cv2.imencode('.jpg', annotated)
        
        return StreamingResponse(
            io.BytesIO(buffer.tobytes()),
            media_type="image/jpeg"
        )
    
    @app.post("/stream/start")
    async def start_stream(config: StreamConfig, background_tasks: BackgroundTasks):
        """
        Start processing video stream in background.
        
        Returns stream_id to track status.
        """
        stream_id = str(uuid.uuid4())[:8]
        
        # initialize stream state
        active_streams[stream_id] = {
            "status": "starting",
            "source": config.source,
            "frames_processed": 0,
            "fps": 0.0,
            "start_time": datetime.now().isoformat(),
            "detections_total": 0
        }
        
        # start processing in background
        background_tasks.add_task(
            process_stream_background,
            stream_id,
            config
        )
        
        return {"stream_id": stream_id, "status": "started"}
    
    @app.get("/stream/{stream_id}", response_model=StreamStatus)
    async def get_stream_status(stream_id: str):
        """Get status of running stream."""
        if stream_id not in active_streams:
            raise HTTPException(status_code=404, detail="Stream not found")
        
        stream = active_streams[stream_id]
        return StreamStatus(
            stream_id=stream_id,
            **stream
        )
    
    @app.delete("/stream/{stream_id}")
    async def stop_stream(stream_id: str):
        """Stop running stream."""
        if stream_id not in active_streams:
            raise HTTPException(status_code=404, detail="Stream not found")
        
        active_streams[stream_id]["status"] = "stopped"
        return {"stream_id": stream_id, "status": "stopped"}
    
    @app.get("/streams")
    async def list_streams():
        """List all active streams."""
        return {
            "streams": [
                {"stream_id": sid, "status": s["status"]}
                for sid, s in active_streams.items()
            ]
        }
    
    @app.get("/stats")
    async def get_stats():
        """Get overall statistics."""
        total_detections = sum(s["detections_total"] for s in active_streams.values())
        active_count = sum(1 for s in active_streams.values() if s["status"] == "running")
        
        return {
            "active_streams": active_count,
            "total_streams": len(active_streams),
            "total_detections": total_detections
        }
    
    @app.get("/classes")
    async def get_classes():
        """Get available detection classes."""
        return {"classes": CLASS_NAMES}


def process_stream_background(stream_id: str, config: StreamConfig):
    """Background task to process video stream."""
    import cv2
    
    model = load_model()
    
    try:
        cap = cv2.VideoCapture(config.source)
        
        if not cap.isOpened():
            active_streams[stream_id]["status"] = "error"
            active_streams[stream_id]["error"] = "Could not open video"
            return
        
        active_streams[stream_id]["status"] = "running"
        
        frame_count = 0
        start_time = time.time()
        
        while active_streams[stream_id]["status"] == "running":
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # do detection
            results = model.predict(frame, conf=config.confidence, verbose=False)
            
            # count detections
            if results[0].boxes is not None:
                active_streams[stream_id]["detections_total"] += len(results[0].boxes)
            
            frame_count += 1
            active_streams[stream_id]["frames_processed"] = frame_count
            
            # calculate fps
            elapsed = time.time() - start_time
            if elapsed > 0:
                active_streams[stream_id]["fps"] = frame_count / elapsed
        
        cap.release()
        active_streams[stream_id]["status"] = "completed"
        
    except Exception as e:
        active_streams[stream_id]["status"] = "error"
        active_streams[stream_id]["error"] = str(e)


# run with uvicorn if main
if __name__ == "__main__":
    if FASTAPI_AVAILABLE:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        print("FastAPI not available. Install: pip install fastapi uvicorn")
