import os
import tempfile
import logging
from typing import Optional
import re
import yt_dlp

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YouTubeDownloader:
    """Downloads audio from YouTube videos and extracts it as WAV files."""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def extract_video_id(self, youtube_url: str) -> Optional[str]:
        """Extract video ID from various YouTube URL formats."""
        patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)',
            r'youtube\.com\/v\/([^&\n?#]+)',
            r'youtube\.com\/watch\?.*v=([^&\n?#]+)',
            r'music\.youtube\.com\/watch\?.*v=([^&\n?#]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, youtube_url)
            if match:
                return match.group(1)
        
        return None
    
    def _check_ffmpeg(self):
        """Check if ffmpeg is available."""
        try:
            import subprocess
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def download_audio(self, youtube_url: str, output_path: Optional[str] = None) -> str:
        """
        Download audio from YouTube URL and save as WAV file.
        
        Args:
            youtube_url: YouTube video URL
            output_path: Optional output path. If None, uses temp file.
            
        Returns:
            Path to the downloaded audio file
            
        Raises:
            Exception: If download fails
        """
        if output_path is None:
            output_path = os.path.join(self.temp_dir, "temp_audio.wav")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Check if ffmpeg is available
        if not self._check_ffmpeg():
            raise Exception(
                "ffmpeg not found. Please install ffmpeg:\n"
                "Ubuntu/Debian: sudo apt update && sudo apt install ffmpeg\n"
                "macOS: brew install ffmpeg\n"
                "Windows: Download from https://ffmpeg.org/download.html"
            )
        
        # Configure yt-dlp options
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': output_path.replace('.wav', '.%(ext)s'),
            'extractaudio': True,
            'audioformat': 'wav',
            'audioquality': '0',  # Best quality
            'noplaylist': True,
            'quiet': True,
            'no_warnings': True,
        }
        
        try:
            logger.info(f"Downloading audio from: {youtube_url}")
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Extract info first to validate URL
                info = ydl.extract_info(youtube_url, download=False)
                
                # Check for common issues
                if info.get('is_live'):
                    raise Exception("Cannot download live streams")
                
                if info.get('age_limit', 0) > 0:
                    raise Exception("Video requires age verification")
                
                # Download the audio
                ydl.download([youtube_url])
            
            # Find the actual output file (yt-dlp might change extension)
            base_path = output_path.replace('.wav', '')
            for ext in ['.wav', '.webm', '.m4a', '.mp3']:
                potential_path = base_path + ext
                if os.path.exists(potential_path):
                    if ext != '.wav':
                        # Convert to WAV if needed
                        wav_path = base_path + '.wav'
                        import subprocess
                        convert_cmd = ['ffmpeg', '-i', potential_path, '-acodec', 'pcm_s16le', '-ar', '16000', wav_path, '-y']
                        subprocess.run(convert_cmd, capture_output=True, check=True)
                        os.remove(potential_path)  # Clean up original file
                        return wav_path
                    return potential_path
            
            raise Exception("Downloaded file not found")
            
        except yt_dlp.DownloadError as e:
            error_msg = str(e)
            if "Video unavailable" in error_msg:
                raise Exception("Video is unavailable or private")
            elif "Sign in to confirm your age" in error_msg:
                raise Exception("Video requires age verification")
            elif "Private video" in error_msg:
                raise Exception("Video is private")
            else:
                raise Exception(f"Download failed: {error_msg}")
        except Exception as e:
            if "ffmpeg" in str(e).lower():
                raise Exception(
                    "ffmpeg/ffprobe not found. Please install ffmpeg:\n"
                    "Ubuntu/Debian: sudo apt update && sudo apt install ffmpeg\n"
                    "macOS: brew install ffmpeg\n"
                    "Windows: Download from https://ffmpeg.org/download.html"
                )
            raise Exception(f"Download failed: {str(e)}")
    
    def cleanup(self):
        """Clean up temporary files."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def __del__(self):
        """Cleanup on destruction."""
        self.cleanup()


def main():
    """Test the YouTube downloader."""
    downloader = YouTubeDownloader()
    
    # Test with a sample URL
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Rick Roll for testing
    
    try:
        video_id = downloader.extract_video_id(test_url)
        print(f"Extracted video ID: {video_id}")
        
        audio_path = downloader.download_audio(test_url)
        print(f"Downloaded audio to: {audio_path}")
        
        # Check if file exists and has reasonable size
        if os.path.exists(audio_path):
            size = os.path.getsize(audio_path)
            print(f"File size: {size} bytes")
        else:
            print("File not found!")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        downloader.cleanup()


if __name__ == "__main__":
    main()

