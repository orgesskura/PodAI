import requests
import xml.etree.ElementTree as ET
import re
from bs4 import BeautifulSoup
import os


def extract_transcript_links(rss_url):
    response = requests.get(rss_url)
    root = ET.fromstring(response.content)

    pattern = r"Transcript:\s*(https?://lexfridman\.com/[^/\s]+(?:-transcript)?)"
    transcript_links = []

    for item in root.findall(".//item"):
        description = item.find("description")
        if description is not None and description.text:
            matches = re.findall(pattern, description.text)
            transcript_links.extend(matches)

    return list(set(transcript_links))  # Remove duplicates


def extract_conversation(transcript_url):
    response = requests.get(transcript_url)
    soup = BeautifulSoup(response.content, "html.parser")

    segments = soup.find_all("div", class_="ts-segment")

    conversation = []
    for segment in segments:
        name = segment.find("span", class_="ts-name")
        timestamp = segment.find("span", class_="ts-timestamp")
        text = segment.find("span", class_="ts-text")

        if name and text:
            name_text = name.text.strip()
            timestamp_text = timestamp.text.strip() if timestamp else ""
            dialogue = text.text.strip()
            conversation.append(f"{name_text} {timestamp_text}: {dialogue}\n")

    return conversation


def save_conversation(conversation, filename):
    with open(filename, "w", encoding="utf-8") as f:
        f.writelines(conversation)


# Main execution
rss_url = "https://lexfridman.com/feed/podcast/"
transcript_links = extract_transcript_links(rss_url)

print(f"Found {len(transcript_links)} transcript links:")
for link in transcript_links:
    print(link)

# Create a directory to store the transcripts
if not os.path.exists("transcripts"):
    os.makedirs("transcripts")

for i, link in enumerate(transcript_links, 1):
    try:
        print(f"Processing transcript {i}: {link}")
        conversation = extract_conversation(link)

        if not conversation:
            print(f"No conversation extracted from {link}")
            continue

        # Create a filename based on the episode number
        filename = f"transcripts/episode_{i:03d}_transcript.txt"

        save_conversation(conversation, filename)
        print(
            f"Saved transcript for episode {i} to {filename} ({len(conversation)} lines)"
        )
    except Exception as e:
        print(f"Error processing {link}: {str(e)}")

print("Finished processing transcripts.")
