import os
import yt_dlp
import re
import unicodedata

def normalize_filename(filename):
    filename = unicodedata.normalize("NFKD", filename).encode("ASCII", "ignore").decode("ASCII")
    filename = re.sub(r'[\\/*?:"<>|]', "", filename)
    filename = filename.replace(" ", "_")
    return filename.lower()

def download_youtube_audio(url, output_path):
    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        "outtmpl": os.path.join(output_path, "%(title)s.%(ext)s"),
        "quiet": False,
        "no_warnings": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            title = info["title"]
            filename = normalize_filename(f"{title}.mp3")
            full_path = os.path.join(output_path, filename)

            print(f"Downloading: {title}")
            ydl.download([url])
            print(f"Download complete: {title}")
            
            return full_path
    except Exception as e:
        print(f"An error occurred while downloading {url}: {str(e)}")
        return None

def main():
    output_path = "downloads"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    url = input("Enter the YouTube video URL: ")
    downloaded_file = download_youtube_audio(url, output_path)

    if downloaded_file:
        print(f"MP3 file downloaded: {downloaded_file}")
    else:
        print("Failed to download the MP3 file.")

if __name__ == "__main__":
    main()