import os
import torch
import librosa
import numpy as np
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pyannote.audio import Pipeline as DiarizationPipeline
import multiprocessing

HF_TOKEN = "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"


def load_whisper_pipeline():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )


def process_audio_chunk(chunk, sample_rate, diarization_pipeline, asr_pipeline):
    waveform_torch = torch.from_numpy(chunk).unsqueeze(0)

    diarization = diarization_pipeline(
        {"waveform": waveform_torch, "sample_rate": sample_rate}
    )

    result = asr_pipeline(chunk, generate_kwargs={"language": "english"})

    transcript = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start, end = turn.start, turn.end
        text = result["text"][int(start * 10) : int(end * 10)]  # Improved alignment
        if text.strip():
            transcript.append(f"Speaker {speaker}: {text.strip()}")

    return transcript


def process_audio_file(file_path, output_dir):
    print(f"Processing {file_path}...")

    chunk_length = 30
    stream = librosa.stream(
        file_path, block_length=chunk_length, frame_length=2048, hop_length=512
    )

    diarization_pipeline = DiarizationPipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN
    )
    asr_pipeline = load_whisper_pipeline()

    with multiprocessing.Pool() as pool:
        results = pool.starmap(
            process_audio_chunk,
            [(chunk, 16000, diarization_pipeline, asr_pipeline) for chunk in stream],
        )

    transcript = [item for sublist in results for item in sublist]

    output_filename = (
        os.path.splitext(os.path.basename(file_path))[0] + "_transcript.txt"
    )
    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, "w") as f:
        f.write("\n".join(transcript))

    print(f"Transcript saved to {output_path}")


def process_audio_files(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".mp3"):
            file_path = os.path.join(input_dir, filename)
            process_audio_file(file_path, output_dir)


def main():
    input_dir = "/Users/chankit/PodAI/podAI-1/downloads"
    output_dir = "/Users/chankit/PodAI/podAI-1/transcripts/"

    process_audio_files(input_dir, output_dir)


if __name__ == "__main__":
    main()
