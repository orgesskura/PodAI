import yt_dlp
import os

soundcloud_url = "https://w.soundcloud.com/player/?url=https%3A//api.soundcloud.com/users/264034133&amp;color=ff5500&amp;auto_play=false&amp;hide_related=false&amp;show_comments=true&amp;show_user=true&amp;show_reposts=false"

output_folder = "soundcloud_downloads"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

ydl_opts = {
    "format": "bestaudio/best",
    "postprocessors": [
        {
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }
    ],
    "outtmpl": os.path.join(output_folder, "%(title)s.%(ext)s"),
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([soundcloud_url])
