#!/usr/bin/env python3
"""
Test client for the Music Clustering API.
"""

import requests
import json
import time
from typing import Dict, Any


class MusicClusteringClient:
    """Client for interacting with the Music Clustering API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
    
    def health_check(self) -> Dict[str, Any]:
        """Check if the API is healthy and models are loaded."""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def find_similar_songs(self, youtube_url: str, top_k: int = 20) -> Dict[str, Any]:
        """
        Find similar songs based on a YouTube URL.
        
        Args:
            youtube_url: YouTube video URL
            top_k: Number of similar songs to return
            
        Returns:
            API response with similar songs
        """
        payload = {
            "youtube_url": youtube_url,
            "top_k": top_k
        }
        
        response = requests.post(
            f"{self.base_url}/similarity",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json()
    
    def print_similar_songs(self, result: Dict[str, Any]):
        """Pretty print the similar songs results."""
        print(f"\n🎵 Query: {result['query_url']}")
        print(f"⏱️  Processing time: {result['processing_time']:.2f} seconds")
        print(f"🔍 Found {len(result['similar_songs'])} similar songs:\n")
        
        for i, song in enumerate(result['similar_songs'], 1):
            print(f"{i:2d}. {song['youtube_url']}")
            print(f"    📊 Similarity: {song['similarity_score']:.4f}")
            print(f"    ⏰ Segment: {song['start_time']:.1f}s - {song['end_time']:.1f}s")
            if song['labels']:
                print(f"    🏷️  Labels: {song['labels'][:3]}")  # Show first 3 labels
            print()


def main():
    """Test the Music Clustering API."""
    client = MusicClusteringClient()
    
    # Test URLs (you can replace these with any YouTube URLs)
    test_urls = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",  # Rick Roll
        "https://www.youtube.com/watch?v=9bZkp7q19f0",  # PSY - GANGNAM STYLE
    ]
    
    try:
        # Health check
        print("🔍 Checking API health...")
        health = client.health_check()
        print(f"✅ API Status: {health['status']}")
        print(f"🤖 Models loaded: {health['models_loaded']}")
        
        if not health['models_loaded']:
            print("❌ Models not loaded. Please check the server logs.")
            return
        
        # Test similarity search
        for url in test_urls:
            print(f"\n{'='*60}")
            print(f"Testing with: {url}")
            print('='*60)
            
            try:
                result = client.find_similar_songs(url, top_k=10)
                client.print_similar_songs(result)
            except requests.exceptions.RequestException as e:
                print(f"❌ Error: {e}")
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
            
            # Small delay between requests
            time.sleep(1)
    
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to the API. Make sure the server is running on http://localhost:8000")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
