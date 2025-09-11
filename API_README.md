# Music Clustering Web Server

A web server that takes YouTube links, extracts audio embeddings, and returns the top 20 most similar songs from the AudioSet dataset.

## Features

- 🎵 **YouTube Audio Processing**: Downloads and processes audio from YouTube videos
- 🧠 **AI-Powered Similarity**: Uses VGGish embeddings for audio similarity
- 🔍 **Fast Search**: FAISS-based similarity search for quick results
- 🌐 **REST API**: Clean FastAPI-based web interface
- 📊 **Rich Metadata**: Returns similarity scores, timestamps, and labels

## Prerequisites

### System Requirements
- Python 3.8+
- FFmpeg (for audio processing)
- yt-dlp (for YouTube downloads)

### Model Files
You need the following files in your project directory:
- `audioset.index` - FAISS index file
- `audioset.lmdb` - LMDB database with metadata
- `model/audioset-vggish-3.pb` - VGGish model file

## Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install system dependencies:**
   ```bash
   # Ubuntu/Debian
   sudo apt update
   sudo apt install ffmpeg
   
   # macOS
   brew install ffmpeg
   
   # Windows
   # Download FFmpeg from https://ffmpeg.org/download.html
   ```

## Quick Start

1. **Start the server:**
   ```bash
   python start_server.py
   ```

2. **Test the API:**
   ```bash
   python test_client.py
   ```

3. **Access the API documentation:**
   Open http://localhost:8000/docs in your browser

## API Endpoints

### Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true
}
```

### Find Similar Songs
```http
POST /similarity
```

**Request:**
```json
{
  "youtube_url": "https://www.youtube.com/watch?v=VIDEO_ID",
  "top_k": 20
}
```

**Response:**
```json
{
  "query_url": "https://www.youtube.com/watch?v=VIDEO_ID",
  "processing_time": 15.23,
  "similar_songs": [
    {
      "youtube_url": "https://www.youtube.com/watch?v=SIMILAR_ID&t=30s",
      "video_id": "SIMILAR_ID",
      "start_time": 30.0,
      "end_time": 31.0,
      "similarity_score": 0.8542,
      "distance": 0.1708,
      "labels": ["Music", "Pop music", "Rock music"]
    }
  ]
}
```

## Usage Examples

### Python Client
```python
import requests

# Find similar songs
response = requests.post("http://localhost:8000/similarity", json={
    "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "top_k": 10
})

result = response.json()
for song in result['similar_songs']:
    print(f"Similar: {song['youtube_url']} (score: {song['similarity_score']:.3f})")
```

### cURL
```bash
curl -X POST "http://localhost:8000/similarity" \
     -H "Content-Type: application/json" \
     -d '{
       "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
       "top_k": 5
     }'
```

### JavaScript/Node.js
```javascript
const response = await fetch('http://localhost:8000/similarity', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    youtube_url: 'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
    top_k: 10
  })
});

const result = await response.json();
console.log('Similar songs:', result.similar_songs);
```

## Configuration

### Environment Variables
- `PORT`: Server port (default: 8000)
- `HOST`: Server host (default: 0.0.0.0)
- `LOG_LEVEL`: Logging level (default: info)

### FAISS Search Parameters
The server uses the following FAISS parameters:
- `nprobe`: 16 (number of clusters to search)
- `top_k`: 20 (default number of results)

You can modify these in `src/web_server.py` if needed.

## Performance Notes

- **First Request**: May take longer due to model loading
- **Audio Download**: Depends on video length and internet speed
- **Embedding Extraction**: ~2-5 seconds for typical songs
- **Similarity Search**: <1 second with FAISS index

## Troubleshooting

### Common Issues

1. **"Models not loaded" error:**
   - Check if `audioset.index`, `audioset.lmdb`, and `model/audioset-vggish-3.pb` exist
   - Verify file permissions

2. **"yt-dlp not found" error:**
   ```bash
   pip install yt-dlp
   ```

3. **"FFmpeg not found" error:**
   - Install FFmpeg system-wide
   - Ensure it's in your PATH

4. **Memory issues:**
   - The server loads the entire FAISS index into memory
   - Ensure you have sufficient RAM (4GB+ recommended)

5. **Slow performance:**
   - Check if the FAISS index is optimized
   - Consider reducing `nprobe` parameter
   - Use SSD storage for better I/O performance

### Logs
Check the server logs for detailed error information:
```bash
python start_server.py 2>&1 | tee server.log
```

## Development

### Project Structure
```
Music-Clustering/
├── src/
│   ├── web_server.py          # Main FastAPI server
│   ├── youtube_downloader.py  # YouTube audio downloader
│   ├── faiss_loader.py        # FAISS index management
│   ├── extract_embeddings.py  # VGGish embedding extraction
│   └── tf_parser.py           # TensorFlow record parser
├── model/                     # VGGish model files
├── audioset.index            # FAISS index
├── audioset.lmdb/            # LMDB metadata database
├── requirements.txt          # Python dependencies
├── start_server.py           # Server startup script
└── test_client.py            # API test client
```

### Adding New Features
1. Modify `src/web_server.py` for new endpoints
2. Update `requirements.txt` for new dependencies
3. Test with `test_client.py`
4. Update this documentation

## License

This project uses the AudioSet dataset and VGGish model. Please check their respective licenses for usage terms.

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review server logs for error details
3. Ensure all dependencies are properly installed
4. Verify model files are present and accessible

