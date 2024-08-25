import os
import wget
from omegaconf import OmegaConf
import json
import shutil
import torch
import torchaudio
from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from deepmultilingualpunctuation import PunctuationModel
import re
import logging
import nltk
from whisperx.utils import LANGUAGES, TO_LANGUAGE_CODE
from ctc_forced_aligner import (
    load_alignment_model,
    generate_emissions,
    preprocess_text,
    get_alignments,
    get_spans,
    postprocess_results,
)

punct_model_langs = [
    "en",
    "fr",
    "de",
    "es",
    "it",
    "nl",
    "pt",
    "bg",
    "pl",
    "cs",
    "sk",
    "sl",
]
langs_to_iso = {
    "af": "afr",
    "am": "amh",
    "ar": "ara",
    "as": "asm",
    "az": "aze",
    "ba": "bak",
    "be": "bel",
    "bg": "bul",
    "bn": "ben",
    "bo": "tib",
    "br": "bre",
    "bs": "bos",
    "ca": "cat",
    "cs": "cze",
    "cy": "wel",
    "da": "dan",
    "de": "ger",
    "el": "gre",
    "en": "eng",
    "es": "spa",
    "et": "est",
    "eu": "baq",
    "fa": "per",
    "fi": "fin",
    "fo": "fao",
    "fr": "fre",
    "gl": "glg",
    "gu": "guj",
    "ha": "hau",
    "haw": "haw",
    "he": "heb",
    "hi": "hin",
    "hr": "hrv",
    "ht": "hat",
    "hu": "hun",
    "hy": "arm",
    "id": "ind",
    "is": "ice",
    "it": "ita",
    "ja": "jpn",
    "jw": "jav",
    "ka": "geo",
    "kk": "kaz",
    "km": "khm",
    "kn": "kan",
    "ko": "kor",
    "la": "lat",
    "lb": "ltz",
    "ln": "lin",
    "lo": "lao",
    "lt": "lit",
    "lv": "lav",
    "mg": "mlg",
    "mi": "mao",
    "mk": "mac",
    "ml": "mal",
    "mn": "mon",
    "mr": "mar",
    "ms": "may",
    "mt": "mlt",
    "my": "bur",
    "ne": "nep",
    "nl": "dut",
    "nn": "nno",
    "no": "nor",
    "oc": "oci",
    "pa": "pan",
    "pl": "pol",
    "ps": "pus",
    "pt": "por",
    "ro": "rum",
    "ru": "rus",
    "sa": "san",
    "sd": "snd",
    "si": "sin",
    "sk": "slo",
    "sl": "slv",
    "sn": "sna",
    "so": "som",
    "sq": "alb",
    "sr": "srp",
    "su": "sun",
    "sv": "swe",
    "sw": "swa",
    "ta": "tam",
    "te": "tel",
    "tg": "tgk",
    "th": "tha",
    "tk": "tuk",
    "tl": "tgl",
    "tr": "tur",
    "tt": "tat",
    "uk": "ukr",
    "ur": "urd",
    "uz": "uzb",
    "vi": "vie",
    "yi": "yid",
    "yo": "yor",
    "yue": "yue",
    "zh": "chi",
}


whisper_langs = sorted(LANGUAGES.keys()) + sorted(
    [k.title() for k in TO_LANGUAGE_CODE.keys()]
)


def create_config(output_dir):
    DOMAIN_TYPE = "telephonic"  # Can be meeting, telephonic, or general based on domain type of the audio file
    CONFIG_FILE_NAME = f"diar_infer_{DOMAIN_TYPE}.yaml"
    CONFIG_URL = f"https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/{CONFIG_FILE_NAME}"
    MODEL_CONFIG = os.path.join(output_dir, CONFIG_FILE_NAME)
    if not os.path.exists(MODEL_CONFIG):
        MODEL_CONFIG = wget.download(CONFIG_URL, output_dir)

    config = OmegaConf.load(MODEL_CONFIG)

    data_dir = os.path.join(output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    meta = {
        "audio_filepath": os.path.join(output_dir, "mono_file.wav"),
        "offset": 0,
        "duration": None,
        "label": "infer",
        "text": "-",
        "rttm_filepath": None,
        "uem_filepath": None,
    }
    with open(os.path.join(data_dir, "input_manifest.json"), "w") as fp:
        json.dump(meta, fp)
        fp.write("\n")

    pretrained_vad = "vad_multilingual_marblenet"
    pretrained_speaker_model = "titanet_large"
    config.num_workers = 0  # Workaround for multiprocessing hanging with ipython issue
    config.diarizer.manifest_filepath = os.path.join(data_dir, "input_manifest.json")
    config.diarizer.out_dir = (
        output_dir  # Directory to store intermediate files and prediction outputs
    )

    config.diarizer.speaker_embeddings.model_path = pretrained_speaker_model
    config.diarizer.oracle_vad = (
        False  # compute VAD provided with model_path to vad config
    )
    config.diarizer.clustering.parameters.oracle_num_speakers = False

    # Here, we use our in-house pretrained NeMo VAD model
    config.diarizer.vad.model_path = pretrained_vad
    config.diarizer.vad.parameters.onset = 0.8
    config.diarizer.vad.parameters.offset = 0.6
    config.diarizer.vad.parameters.pad_offset = -0.05
    config.diarizer.msdd_model.model_path = (
        "diar_msdd_telephonic"  # Telephonic speaker diarization model
    )

    return config


def get_word_ts_anchor(s, e, option="start"):
    if option == "end":
        return e
    elif option == "mid":
        return (s + e) / 2
    return s


def get_words_speaker_mapping(wrd_ts, spk_ts, word_anchor_option="start"):
    s, e, sp = spk_ts[0]
    wrd_pos, turn_idx = 0, 0
    wrd_spk_mapping = []
    for wrd_dict in wrd_ts:
        ws, we, wrd = (
            int(wrd_dict["start"] * 1000),
            int(wrd_dict["end"] * 1000),
            wrd_dict["text"],
        )
        wrd_pos = get_word_ts_anchor(ws, we, word_anchor_option)
        while wrd_pos > float(e):
            turn_idx += 1
            turn_idx = min(turn_idx, len(spk_ts) - 1)
            s, e, sp = spk_ts[turn_idx]
            if turn_idx == len(spk_ts) - 1:
                e = get_word_ts_anchor(ws, we, option="end")
        wrd_spk_mapping.append(
            {"word": wrd, "start_time": ws, "end_time": we, "speaker": sp}
        )
    return wrd_spk_mapping


sentence_ending_punctuations = ".?!"


def get_first_word_idx_of_sentence(word_idx, word_list, speaker_list, max_words):
    is_word_sentence_end = (
        lambda x: x >= 0 and word_list[x][-1] in sentence_ending_punctuations
    )
    left_idx = word_idx
    while (
        left_idx > 0
        and word_idx - left_idx < max_words
        and speaker_list[left_idx - 1] == speaker_list[left_idx]
        and not is_word_sentence_end(left_idx - 1)
    ):
        left_idx -= 1

    return left_idx if left_idx == 0 or is_word_sentence_end(left_idx - 1) else -1


def get_last_word_idx_of_sentence(word_idx, word_list, max_words):
    is_word_sentence_end = (
        lambda x: x >= 0 and word_list[x][-1] in sentence_ending_punctuations
    )
    right_idx = word_idx
    while (
        right_idx < len(word_list) - 1
        and right_idx - word_idx < max_words
        and not is_word_sentence_end(right_idx)
    ):
        right_idx += 1

    return (
        right_idx
        if right_idx == len(word_list) - 1 or is_word_sentence_end(right_idx)
        else -1
    )


def get_realigned_ws_mapping_with_punctuation(
    word_speaker_mapping, max_words_in_sentence=50
):
    is_word_sentence_end = (
        lambda x: x >= 0
        and word_speaker_mapping[x]["word"][-1] in sentence_ending_punctuations
    )
    wsp_len = len(word_speaker_mapping)

    words_list, speaker_list = [], []
    for k, line_dict in enumerate(word_speaker_mapping):
        word, speaker = line_dict["word"], line_dict["speaker"]
        words_list.append(word)
        speaker_list.append(speaker)

    k = 0
    while k < len(word_speaker_mapping):
        line_dict = word_speaker_mapping[k]
        if (
            k < wsp_len - 1
            and speaker_list[k] != speaker_list[k + 1]
            and not is_word_sentence_end(k)
        ):
            left_idx = get_first_word_idx_of_sentence(
                k, words_list, speaker_list, max_words_in_sentence
            )
            right_idx = (
                get_last_word_idx_of_sentence(
                    k, words_list, max_words_in_sentence - k + left_idx - 1
                )
                if left_idx > -1
                else -1
            )
            if min(left_idx, right_idx) == -1:
                k += 1
                continue

            spk_labels = speaker_list[left_idx : right_idx + 1]
            mod_speaker = max(set(spk_labels), key=spk_labels.count)
            if spk_labels.count(mod_speaker) < len(spk_labels) // 2:
                k += 1
                continue

            speaker_list[left_idx : right_idx + 1] = [mod_speaker] * (
                right_idx - left_idx + 1
            )
            k = right_idx

        k += 1

    k, realigned_list = 0, []
    while k < len(word_speaker_mapping):
        line_dict = word_speaker_mapping[k].copy()
        line_dict["speaker"] = speaker_list[k]
        realigned_list.append(line_dict)
        k += 1

    return realigned_list


def get_sentences_speaker_mapping(word_speaker_mapping, spk_ts):
    sentence_checker = nltk.tokenize.PunktSentenceTokenizer().text_contains_sentbreak
    s, e, spk = spk_ts[0]
    prev_spk = spk

    snts = []
    snt = {"speaker": f"Speaker {spk}", "start_time": s, "end_time": e, "text": ""}

    for wrd_dict in word_speaker_mapping:
        wrd, spk = wrd_dict["word"], wrd_dict["speaker"]
        s, e = wrd_dict["start_time"], wrd_dict["end_time"]
        if spk != prev_spk or sentence_checker(snt["text"] + " " + wrd):
            snts.append(snt)
            snt = {
                "speaker": f"Speaker {spk}",
                "start_time": s,
                "end_time": e,
                "text": "",
            }
        else:
            snt["end_time"] = e
        snt["text"] += wrd + " "
        prev_spk = spk

    snts.append(snt)
    return snts


def get_speaker_aware_transcript(sentences_speaker_mapping, f):
    previous_speaker = sentences_speaker_mapping[0]["speaker"]
    f.write(f"{previous_speaker}: ")

    for sentence_dict in sentences_speaker_mapping:
        speaker = sentence_dict["speaker"]
        sentence = sentence_dict["text"]

        # If this speaker doesn't match the previous one, start a new paragraph
        if speaker != previous_speaker:
            f.write(f"\n\n{speaker}: ")
            previous_speaker = speaker

        # No matter what, write the current sentence
        f.write(sentence + " ")


def format_timestamp(
    milliseconds: float, always_include_hours: bool = False, decimal_marker: str = "."
):
    assert milliseconds >= 0, "non-negative timestamp expected"

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return (
        f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"
    )


def write_srt(transcript, file):
    """
    Write a transcript to a file in SRT format.

    """
    for i, segment in enumerate(transcript, start=1):
        # write srt lines
        print(
            f"{i}\n"
            f"{format_timestamp(segment['start_time'], always_include_hours=True, decimal_marker=',')} --> "
            f"{format_timestamp(segment['end_time'], always_include_hours=True, decimal_marker=',')}\n"
            f"{segment['speaker']}: {segment['text'].strip().replace('-->', '->')}\n",
            file=file,
            flush=True,
        )


def find_numeral_symbol_tokens(tokenizer):
    numeral_symbol_tokens = [
        -1,
    ]
    for token, token_id in tokenizer.get_vocab().items():
        has_numeral_symbol = any(c in "0123456789%$£" for c in token)
        if has_numeral_symbol:
            numeral_symbol_tokens.append(token_id)
    return numeral_symbol_tokens


def _get_next_start_timestamp(word_timestamps, current_word_index, final_timestamp):
    # if current word is the last word
    if current_word_index == len(word_timestamps) - 1:
        return word_timestamps[current_word_index]["start"]

    next_word_index = current_word_index + 1
    while current_word_index < len(word_timestamps) - 1:
        if word_timestamps[next_word_index].get("start") is None:
            # if next word doesn't have a start timestamp
            # merge it with the current word and delete it
            word_timestamps[current_word_index]["word"] += (
                " " + word_timestamps[next_word_index]["word"]
            )

            word_timestamps[next_word_index]["word"] = None
            next_word_index += 1
            if next_word_index == len(word_timestamps):
                return final_timestamp

        else:
            return word_timestamps[next_word_index]["start"]


def filter_missing_timestamps(
    word_timestamps, initial_timestamp=0, final_timestamp=None
):
    # handle the first and last word
    if word_timestamps[0].get("start") is None:
        word_timestamps[0]["start"] = (
            initial_timestamp if initial_timestamp is not None else 0
        )
        word_timestamps[0]["end"] = _get_next_start_timestamp(
            word_timestamps, 0, final_timestamp
        )

    result = [
        word_timestamps[0],
    ]

    for i, ws in enumerate(word_timestamps[1:], start=1):
        # if ws doesn't have a start and end
        # use the previous end as start and next start as end
        if ws.get("start") is None and ws.get("word") is not None:
            ws["start"] = word_timestamps[i - 1]["end"]
            ws["end"] = _get_next_start_timestamp(word_timestamps, i, final_timestamp)

        if ws["word"] is not None:
            result.append(ws)
    return result


def cleanup(path: str):
    """path could either be relative or absolute."""
    # check if file or directory exists
    if os.path.isfile(path) or os.path.islink(path):
        # remove file
        os.remove(path)
    elif os.path.isdir(path):
        # remove directory and all its content
        shutil.rmtree(path)
    else:
        raise ValueError("Path {} is not a file or dir.".format(path))


def process_language_arg(language: str, model_name: str):
    """
    Process the language argument to make sure it's valid and convert language names to language codes.
    """
    if language is not None:
        language = language.lower()
    if language not in LANGUAGES:
        if language in TO_LANGUAGE_CODE:
            language = TO_LANGUAGE_CODE[language]
        else:
            raise ValueError(f"Unsupported language: {language}")

    if model_name.endswith(".en") and language != "en":
        if language is not None:
            logging.warning(
                f"{model_name} is an English-only model but received '{language}'; using English instead."
            )
        language = "en"
    return language


def transcribe_batched(
    audio_file: str,
    language: str,
    batch_size: int,
    model_name: str,
    compute_dtype: str,
    suppress_numerals: bool,
    device: str,
):
    import whisperx

    # Faster Whisper batched
    whisper_model = whisperx.load_model(
        model_name,
        device,
        compute_type=compute_dtype,
        asr_options={"suppress_numerals": suppress_numerals},
    )
    audio = whisperx.load_audio(audio_file)
    result = whisper_model.transcribe(audio, language=language, batch_size=batch_size)
    del whisper_model
    torch.cuda.empty_cache()
    return result["segments"], result["language"], audio


import os
import logging
import torch
import torchaudio
from glob import glob
import time
from tqdm import tqdm
import re
from google.colab import drive
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

def process_folder(input_folder, storage_account, output_type, output_container, **kwargs):
    # Mount Google Drive
    drive.mount('/content/drive')

    # Set up Azure Blob Storage client
    connect_str = os.environ.get('DefaultEndpointsProtocol=https;AccountName=audio7712972043;AccountKey=R9el+JUinydgn+SbKhyjL+1IE0l4FSj3tQwT3Oi20ZixbWub9Y5ZTAxjtqDOWT+0rY1xC686P9NI+AStMlBzmg==;EndpointSuffix=core.windows.net')
    if not connect_str:
        raise ValueError("AZURE_STORAGE_CONNECTION_STRING environment variable is not set")
    
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    container_client = blob_service_client.get_container_client(output_container)

    # Ensure the container exists
    try:
        container_client.create_container()
    except Exception as e:
        logging.warning(f"Container creation failed: {str(e)}. Proceeding with existing container.")

    # Get list of audio files from Google Drive input folder
    audio_files = glob(os.path.join('/content/drive/MyDrive', input_folder, "*.mp3")) + \
                  glob(os.path.join('/content/drive/MyDrive', input_folder, "*.wav"))

    successful_files = 0
    failed_files = 0
    skipped_files = 0

    for audio_file in tqdm(audio_files, desc="Processing audio files"):
        try:
            # Check if output files already exist
            if output_files_exist(audio_file, output_container, container_client):
                logging.info(f"Skipping {audio_file} as output files already exist.")
                skipped_files += 1
                continue

            # Process the audio file
            success = process_single_file(audio_file, '/tmp', **kwargs)
            
            if success:
                # Upload the results to Azure Blob Storage
                base_name = os.path.splitext(os.path.basename(audio_file))[0]
                for output_file in ['transcript.txt', 'transcript.srt']:
                    local_file_path = os.path.join('/tmp', base_name, output_file)
                    blob_name = f"{base_name}/{output_file}"
                    blob_client = container_client.get_blob_client(blob_name)
                    with open(local_file_path, "rb") as data:
                        blob_client.upload_blob(data, overwrite=True)
                
                successful_files += 1
            else:
                failed_files += 1

            # Clean up temporary files
            shutil.rmtree(os.path.join('/tmp', base_name), ignore_errors=True)

        except Exception as e:
            logging.error(f"An error occurred while processing {audio_file}: {str(e)}")
            failed_files += 1

        # Clear CUDA cache after each file
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Add a small delay to allow system resources to stabilize
        time.sleep(1)

    logging.info(f"Processing completed. Successful: {successful_files}, Failed: {failed_files}, Skipped: {skipped_files}")

def sanitize_filename(filename):
    return re.sub(r'[^\w\-_\. ]', '_', filename)

def check_files_in_blob_storage(container_client, file_paths):
    result = {}
    for file_path in file_paths:
        blob_client = container_client.get_blob_client(file_path)
        result[file_path] = blob_client.exists()
    return result

def output_files_exist(audio_path, output_container, container_client):
    base_name = sanitize_filename(os.path.splitext(os.path.basename(audio_path))[0])
    relative_txt_path = f"{base_name}/transcript.txt"
    relative_srt_path = f"{base_name}/transcript.srt"

    files_to_check = [relative_txt_path, relative_srt_path]
    existence_status = check_files_in_blob_storage(container_client, files_to_check)

    return all(existence_status.values())

def process_single_file(audio_path, output_dir, enable_stemming=True, whisper_model_name="large-v2", suppress_numerals=True, batch_size=8, language=None):
    # Set up logging for this file
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(os.path.basename(audio_path))

    # Check if output files already exist
    if output_files_exist(audio_path, output_dir):
        logger.info(f"Output files already exist for {audio_path}. Skipping processing.")
        return True

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Create a subdirectory for this file's outputs
        sanitized_base_name = sanitize_filename(os.path.splitext(os.path.basename(audio_path))[0])
        file_output_dir = os.path.join(output_dir, sanitized_base_name)
        os.makedirs(file_output_dir, exist_ok=True)

        # Stemming
        if enable_stemming:
            logger.info(f"Performing source separation on {audio_path}")
            return_code = os.system(
                f'python3 -m demucs.separate -n htdemucs --two-stems=vocals "{audio_path}" -o "{file_output_dir}"'
            )
            if return_code != 0:
                logger.warning("Source splitting failed, using original audio file.")
                vocal_target = audio_path
            else:
                vocal_target = os.path.join(
                    file_output_dir,
                    "htdemucs",
                    sanitized_base_name,
                    "vocals.wav",
                )
        else:
            vocal_target = audio_path

        # Transcription
        logger.info(f"Transcribing {vocal_target}")
        compute_type = "float16"
        whisper_results, detected_language, audio_waveform = transcribe_batched(
            vocal_target,
            language,
            batch_size,
            whisper_model_name,
            compute_type,
            suppress_numerals,
            device,
        )

        # Alignment
        logger.info(f"Aligning transcription for {vocal_target}")
        alignment_model, alignment_tokenizer, alignment_dictionary = load_alignment_model(
            device,
            dtype=torch.float16 if device == "cuda" else torch.float32,
        )

        audio_waveform = (
            torch.from_numpy(audio_waveform)
            .to(alignment_model.dtype)
            .to(alignment_model.device)
        )

        emissions, stride = generate_emissions(
            alignment_model, audio_waveform, batch_size=batch_size
        )

        del alignment_model
        torch.cuda.empty_cache()

        full_transcript = "".join(segment["text"] for segment in whisper_results)

        tokens_starred, text_starred = preprocess_text(
            full_transcript,
            romanize=True,
            language=langs_to_iso[detected_language],
        )

        segments, scores, blank_id = get_alignments(
            emissions,
            tokens_starred,
            alignment_dictionary,
        )

        spans = get_spans(tokens_starred, segments, alignment_tokenizer.decode(blank_id))

        word_timestamps = postprocess_results(text_starred, spans, stride, scores)

        # Convert to mono for NeMo compatibility
        logger.info(f"Converting {vocal_target} to mono")
        torchaudio.save(
            os.path.join(file_output_dir, "mono_file.wav"),
            audio_waveform.cpu().unsqueeze(0).float(),
            16000,
            channels_first=True,
        )

        # Speaker Diarization
        logger.info(f"Performing speaker diarization on {vocal_target}")
        msdd_model = NeuralDiarizer(cfg=create_config(file_output_dir)).to("cuda")
        msdd_model.diarize()

        del msdd_model
        torch.cuda.empty_cache()

        # Mapping Speakers to Sentences
        logger.info(f"Mapping speakers to sentences for {vocal_target}")
        speaker_ts = []
        with open(os.path.join(file_output_dir, "pred_rttms", "mono_file.rttm"), "r") as f:
            lines = f.readlines()
            for line in lines:
                line_list = line.split(" ")
                s = int(float(line_list[5]) * 1000)
                e = s + int(float(line_list[8]) * 1000)
                speaker_ts.append([s, e, int(line_list[11].split("_")[-1])])

        wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")

        # Realigning Speech segments
        if detected_language in punct_model_langs:
            logger.info(f"Restoring punctuation for {vocal_target}")
            punct_model = PunctuationModel(model="kredor/punctuate-all")
            words_list = list(map(lambda x: x["word"], wsm))
            labled_words = punct_model.predict(words_list, chunk_size=230)

            for word_dict, labeled_tuple in zip(wsm, labled_words):
                word = word_dict["word"]
                if (
                    word
                    and labeled_tuple[1] in ".?!"
                    and (word[-1] not in ".,;:!?" or re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", word))
                ):
                    word += labeled_tuple[1]
                    if word.endswith(".."):
                        word = word.rstrip(".")
                    word_dict["word"] = word
        else:
            logger.warning(
                f"Punctuation restoration is not available for {detected_language} language. Using the original punctuation."
            )

        wsm = get_realigned_ws_mapping_with_punctuation(wsm)
        ssm = get_sentences_speaker_mapping(wsm, speaker_ts)

        # Export results
        logger.info(f"Exporting results for {vocal_target}")
        with open(os.path.join(file_output_dir, "transcript.txt"), "w", encoding="utf-8-sig") as f:
            get_speaker_aware_transcript(ssm, f)

        with open(os.path.join(file_output_dir, "transcript.srt"), "w", encoding="utf-8-sig") as srt:
            write_srt(ssm, srt)

        # Cleanup temporary files
        cleanup(os.path.join(file_output_dir, "htdemucs"))
        cleanup(os.path.join(file_output_dir, "pred_rttms"))
        os.remove(os.path.join(file_output_dir, "mono_file.wav"))

        logger.info(f"Processing completed for {audio_path}")
        return True
    except Exception as e:
        logger.error(f"An error occurred while processing {audio_path}: {str(e)}")
        return False

def process_folder(input_folder, output_folder, **kwargs):
    os.makedirs(output_folder, exist_ok=True)
    audio_files = glob(os.path.join(input_folder, "*.mp3")) + glob(os.path.join(input_folder, "*.wav"))

    successful_files = 0
    failed_files = 0
    skipped_files = 0

    for audio_file in tqdm(audio_files, desc="Processing audio files"):
        if output_files_exist(audio_file, output_folder):
            logging.info(f"Skipping {audio_file} as output files already exist.")
            skipped_files += 1
            continue

        try:
            success = process_single_file(audio_file, output_folder, **kwargs)
            if success:
                successful_files += 1
            else:
                failed_files += 1
        except Exception as e:
            logging.error(f"An error occurred while processing {audio_file}: {str(e)}")
            failed_files += 1

        # Clear CUDA cache after each file
        torch.cuda.empty_cache()

        # Add a small delay to allow system resources to stabilize
        time.sleep(1)

    logging.info(f"Processing completed. Successful: {successful_files}, Failed: {failed_files}, Skipped: {skipped_files}")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Check CUDA availability
    logging.info(f"CUDA available: {torch.cuda.is_available()}")

    # Input and output details
    input_folder = "MyDrive/Dataset/lex_audio"  # Google Drive folder
    storage_account = "audio7712972043"
    output_container = "azureml"
    output_type = "azure"

    # Ensure all necessary environment variables are set
    required_env_vars = "DefaultEndpointsProtocol=https;AccountName=audio7712972043;AccountKey=R9el+JUinydgn+SbKhyjL+1IE0l4FSj3tQwT3Oi20ZixbWub9Y5ZTAxjtqDOWT+0rY1xC686P9NI+AStMlBzmg==;EndpointSuffix=core.windows.net"
    for var in required_env_vars:
        if not os.environ.get(var):
            raise ValueError(f"Environment variable {var} is not set")

    # Set up parameters for processing
    process_params = {
        "enable_stemming": True,
        "whisper_model_name": "large-v2",
        "suppress_numerals": True,
        "batch_size": 8,
        "language": None
    }

    # Run the processing
    try:
        process_folder(input_folder, storage_account, output_type, output_container, **process_params)
    except Exception as e:
        logging.error(f"An error occurred during processing: {str(e)}")
    finally:
        # Perform any necessary cleanup
        logging.info("Processing completed. Performing cleanup...")