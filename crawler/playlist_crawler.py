import re
from pytube import Playlist

def extract_playlist_links(playlist_url):
    try:
        playlist = Playlist(playlist_url)
        
        # This line resolves the issue with playlist.video_urls
        playlist._video_regex = re.compile(r"\"url\":\"(/watch\?v=[\w-]*)")
        
        print(f'Number of videos in playlist: {len(playlist.video_urls)}')
        
        for url in playlist.video_urls:
            print(url)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Replace this with your actual playlist URL
playlist_url = "https://www.youtube.com/watch?v=ZoqgAy3h4OM&list=PLQ-uHSnFig5MiLRb-l6yiCBGyqfVyVf17&pp=iAQB"

extract_playlist_links(playlist_url)