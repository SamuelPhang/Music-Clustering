#!/usr/bin/env python3
"""
Example usage of the Music Clustering API.
This script demonstrates how to use the API to find similar songs.
"""

import requests
import json
import time


def find_similar_music(youtube_url, top_k=10):
    """
    Find similar music using the Music Clustering API.
    
    Args:
        youtube_url (str): YouTube video URL
        top_k (int): Number of similar songs to return
        
    Returns:
        dict: API response with similar songs
    """
    api_url = "http://localhost:8000/similarity"
    
    payload = {
        "youtube_url": youtube_url,
        "top_k": top_k
    }
    
    try:
        print(f"🎵 Searching for songs similar to: {youtube_url}")
        print("⏳ Processing... (this may take a few moments)")
        
        response = requests.post(api_url, json=payload, timeout=120)
        response.raise_for_status()
        
        return response.json()
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to the API. Make sure the server is running:")
        print("   python start_server.py")
        return None
    except requests.exceptions.Timeout:
        print("❌ Request timed out. The video might be too long or the server is busy.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"❌ API request failed: {e}")
        return None


def display_results(result):
    """Display the similarity search results in a nice format."""
    if not result:
        return
    
    print(f"\n✅ Found {len(result['similar_songs'])} similar songs!")
    print(f"⏱️  Processing time: {result['processing_time']:.2f} seconds")
    print("\n" + "="*80)
    
    for i, song in enumerate(result['similar_songs'], 1):
        print(f"\n{i:2d}. 🎵 {song['youtube_url']}")
        print(f"    📊 Similarity Score: {song['similarity_score']:.4f}")
        print(f"    ⏰ Time Segment: {song['start_time']:.1f}s - {song['end_time']:.1f}s")
        
        if song['labels']:
            # Show first few labels
            labels_str = ", ".join(song['labels'][:3])
            if len(song['labels']) > 3:
                labels_str += f" (+{len(song['labels'])-3} more)"
            print(f"    🏷️  Labels: {labels_str}")
    
    print("\n" + "="*80)


def main():
    """Main function with example usage."""
    print("🎼 Music Clustering API Example")
    print("=" * 50)
    
    # Example YouTube URLs to test with
    example_urls = [
        {
            "name": "Classic Rock",
            "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Rick Roll
        },
        {
            "name": "Pop Music", 
            "url": "https://www.youtube.com/watch?v=9bZkp7q19f0"  # PSY - GANGNAM STYLE
        }
    ]
    
    print("\nChoose an example to test:")
    for i, example in enumerate(example_urls, 1):
        print(f"{i}. {example['name']}")
    
    print("3. Enter custom YouTube URL")
    
    try:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            url = example_urls[0]["url"]
            name = example_urls[0]["name"]
        elif choice == "2":
            url = example_urls[1]["url"]
            name = example_urls[1]["name"]
        elif choice == "3":
            url = input("Enter YouTube URL: ").strip()
            name = "Custom"
        else:
            print("❌ Invalid choice. Using default example.")
            url = example_urls[0]["url"]
            name = example_urls[0]["name"]
        
        print(f"\n🎵 Testing with {name} example...")
        
        # Find similar songs
        result = find_similar_music(url, top_k=15)
        
        if result:
            display_results(result)
            
            # Ask if user wants to save results
            save = input("\n💾 Save results to file? (y/n): ").strip().lower()
            if save == 'y':
                filename = f"similarity_results_{int(time.time())}.json"
                with open(filename, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"✅ Results saved to {filename}")
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()