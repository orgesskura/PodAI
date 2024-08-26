import os
import requests
import json
import time

API_KEY = "a59e9de77b014650ae71a766d09534c0"
base_url = "https://api.assemblyai.com/v2"

headers = {"authorization": API_KEY}


def transcribe_file(file_path):
    print(f"Processing file: {file_path}")

    # Upload the file
    with open(file_path, "rb") as f:
        response = requests.post(base_url + "/upload", headers=headers, data=f)

    upload_url = response.json()["upload_url"]

    # Request transcription
    data = {"audio_url": upload_url, "speaker_labels": True}

    url = base_url + "/transcript"
    response = requests.post(url, json=data, headers=headers)
    transcript_id = response.json()["id"]
    polling_endpoint = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"

    # Wait for transcription to complete
    while True:
        transcription_result = requests.get(polling_endpoint, headers=headers).json()

        if transcription_result["status"] == "completed":
            return transcription_result["text"], transcription_result["utterances"]
        elif transcription_result["status"] == "error":
            raise RuntimeError(f"Transcription failed: {transcription_result['error']}")
        else:
            time.sleep(3)


def main():
    folder_path = "unique_downloads"
    output_folder = "lex_diarized"

    os.makedirs(output_folder, exist_ok=True)

    # Get a list of all MP3 files in the folder
    mp3_files = [f for f in os.listdir(folder_path) if f.endswith(".mp3")]

    for filename in mp3_files:
        file_path = os.path.join(folder_path, filename)
        try:
            transcript_text, utterances = transcribe_file(file_path)

            output_file = os.path.join(
                output_folder, f"{os.path.splitext(filename)[0]}_transcript.txt"
            )

            with open(output_file, "w", encoding="utf-8") as out_file:
                out_file.write(f"Transcription for {filename}:\n")
                out_file.write(f"Full transcript: {transcript_text}\n\n")
                out_file.write("Utterances:\n")
                for utterance in utterances:
                    speaker = utterance["speaker"]
                    text = utterance["text"]
                    out_file.write(f"Speaker {speaker}: {text}\n")

            print(f"Completed transcription for {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")


if __name__ == "__main__":
    main()
