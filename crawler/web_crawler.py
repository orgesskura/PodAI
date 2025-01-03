import os
import yt_dlp
import re
import unicodedata
import time


def normalize_filename(filename):
    filename = (
        unicodedata.normalize("NFKD", filename)
        .encode("ASCII", "ignore")
        .decode("ASCII")
    )
    filename = re.sub(r'[\\/*?:"<>|]', "", filename)
    filename = filename.replace(" ", "_")
    return filename.lower()


def get_processed_files(output_path):
    return {
        normalize_filename(file): file
        for file in os.listdir(output_path)
        if file.endswith((".mp3", ".part"))
    }


def download_youtube_audio(url, output_path, processed_files, max_retries=3):
    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "outtmpl": os.path.join(output_path, "%(title)s.%(ext)s"),
        "quiet": False,
        "no_warnings": True,
        "ignoreerrors": True,
        "continuedl": True,
    }

    for attempt in range(max_retries):
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                title = info["title"]
                expected_filename = normalize_filename(f"{title}.mp3")
                expected_part_filename = normalize_filename(f"{title}.part")

                if expected_filename in processed_files:
                    print(
                        f"Skipping already downloaded: {processed_files[expected_filename]}"
                    )
                    return False
                elif expected_part_filename in processed_files:
                    print(
                        f"Resuming partial download: {processed_files[expected_part_filename]}"
                    )

                print(f"Downloading: {title} (Attempt {attempt + 1}/{max_retries})")
                ydl.download([url])
                print(f"Download complete: {title}")
                return True
        except yt_dlp.utils.DownloadError as e:
            print(f"Download failed (Attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                print(f"Retrying in 10 seconds...")
                time.sleep(10)
            else:
                print(f"Max retries reached. Skipping {url}")
                return False
        except Exception as e:
            print(f"An unexpected error occurred while processing {url}: {str(e)}")
            return False


def process_url_file(file_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    processed_files = get_processed_files(output_path)

    with open(file_path, "r", encoding="utf-8") as file:
        urls = file.read().splitlines()

    total_urls = len(urls)
    processed_count = len([f for f in processed_files.values() if f.endswith(".mp3")])
    for index, url in enumerate(urls, start=1):
        url = url.strip()
        print(f"Processing URL {index} of {total_urls}: {url}")
        if download_youtube_audio(url, output_path, processed_files):
            processed_count += 1

        print(f"Progress: {processed_count}/{total_urls}")


def main():
    url_file = "yc.txt"
    output_path = "ycdownloads/"

    if not os.path.exists(url_file):
        print(f"File not found: {url_file}")
        return

    process_url_file(url_file, output_path)


if __name__ == "__main__":
    main()

# https://www.youtube.com/watch?v=p3lsYlod5OU
