import os
import sys
import tempfile
import logging
from typing import List, Dict, Any
from contextlib import asynccontextmanager
import numpy as np

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
import uvicorn

# import this before faiss_loader - essentia always needs to be imported first
from extract_embeddings import VGGishEmbeddingsModel

from faiss_loader import FaissLoader
from youtube_downloader import YouTubeDownloader

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for loaded models
faiss_loader = None
embedding_model = None


class SimilarityRequest(BaseModel):
    youtube_url: str
    top_k: int = 20


class SimilarityResponse(BaseModel):
    query_url: str
    similar_songs: List[Dict[str, Any]]
    processing_time: float


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool


def initialize_models():
    """Initialize the FAISS loader and embedding model."""
    global faiss_loader, embedding_model
    
    try:
        # Clear any existing TensorFlow sessions
        # tf.keras.backend.clear_session()
        
        # Change to project root directory to find model files
        original_cwd = os.getcwd()
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        os.chdir(project_root)
        
        logger.info("Initializing FAISS loader...")
        faiss_loader = FaissLoader()
        
        logger.info("Initializing VGGish embedding model...")
        embedding_model = VGGishEmbeddingsModel()
        
        logger.info("Models initialized successfully!")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
        # Clear session on error
        # tf.keras.backend.clear_session()
        return False
    finally:
        # Restore original working directory
        if 'original_cwd' in locals():
            os.chdir(original_cwd)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown."""
    # Startup
    success = initialize_models()
    if not success:
        logger.error("Failed to initialize models. Server may not work properly.")
    yield
    # Shutdown (cleanup if needed)
    logger.info("Shutting down server...")
    # tf.keras.backend.clear_session()


# Initialize FastAPI app
app = FastAPI(
    title="Music Clustering API",
    description="API for finding similar music based on audio embeddings",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    models_loaded = faiss_loader is not None and embedding_model is not None
    return HealthResponse(
        status="healthy" if models_loaded else "models_not_loaded",
        models_loaded=models_loaded
    )


@app.post("/similarity", response_model=SimilarityResponse)
async def find_similar_songs(request: SimilarityRequest):
    """
    Find similar songs based on a YouTube URL.
    
    Args:
        request: Contains YouTube URL and number of similar songs to return
        
    Returns:
        List of similar songs with metadata
    """
    if faiss_loader is None or embedding_model is None:
        raise HTTPException(
            status_code=503, 
            detail="Models not loaded. Please check server health."
        )
    
    import time
    start_time = time.time()
    
    try:
        # Download audio from YouTube
        downloader = YouTubeDownloader()
        temp_audio_path = None
        
        try:
            logger.info(f"Processing YouTube URL: {request.youtube_url}")
            temp_audio_path = downloader.download_audio(request.youtube_url)
            
            # Extract embeddings
            logger.info("Extracting audio embeddings...")
            embeddings = embedding_model.get_embeddings(temp_audio_path, normalize=True)
            
            # Average the embeddings (since we get multiple embeddings per audio file)
            query_embedding = np.mean(embeddings, axis=0).reshape(1, -1)
            
            # Find similar songs
            logger.info(f"Searching for {request.top_k} similar songs...")
            faiss_loader.index.nprobe = 16  # Set search parameters
            distances, indices = faiss_loader.index.search(query_embedding, request.top_k)
            
            # Get metadata for similar songs
            similar_songs = []
            with faiss_loader.lmdb.begin() as txn:
                for idx, dist in zip(indices[0], distances[0]):
                    if idx == -1:  # Skip invalid indices
                        continue
                    
                    meta_bytes = txn.get(str(idx).encode())
                    if meta_bytes is None:
                        continue
                    
                    import pickle
                    meta = pickle.loads(meta_bytes)
                    video_id, start_time, end_time, labels = meta
                    
                    # Create YouTube URL
                    youtube_url = f"https://www.youtube.com/watch?v={video_id}&t={start_time:.0f}s"
                    
                    similar_songs.append({
                        "youtube_url": youtube_url,
                        "video_id": video_id,
                        "start_time": start_time,
                        "end_time": end_time,
                        "similarity_score": float(1.0 / (1.0 + dist)),  # Convert distance to similarity
                        "distance": float(dist),
                        "labels": labels
                    })
            
            processing_time = time.time() - start_time
            
            return SimilarityResponse(
                query_url=request.youtube_url,
                similar_songs=similar_songs,
                processing_time=processing_time
            )
            
        finally:
            # Clean up temporary files
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            downloader.cleanup()
            
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Music Clustering API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "similarity": "/similarity",
            "docs": "/docs"
        }
    }


def main():
    """Run the web server."""
    # Clear TensorFlow session to avoid memory issues
    # tf.keras.backend.clear_session()
    
    # Set TensorFlow to use CPU only to avoid CUDA issues
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    # Run the server
    uvicorn.run(
        "web_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Set to True for development
        log_level="info"
    )


if __name__ == "__main__":
    main()