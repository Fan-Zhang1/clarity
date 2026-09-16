import whisper
import torch
import json
import csv
import librosa

from pathlib import Path
from prepare_audio_mono import (
    get_clean_path,
    prepare_audio_mono,
)

# Audio path
root = Path("/scratch5/fazh9208/clarity"
          "/recipes/whisper_si/CPC1 data"
          "/clarity_CPC1_data.v1_1/"
          "clarity_CPC1_data")

audio_dir_enhanced =  root/ "clarity_data" / "HA_outputs" / "train"
audio_dir_clean = root/ "clarity_data" / "scenes"

print(audio_dir_enhanced)
print(audio_dir_clean)

print("Enhanced dir exists:", audio_dir_enhanced.exists())
print("Clean dir exists:", audio_dir_clean.exists())

ground_truth = root/ "metadata" / "CPC1.train.json"

output_dir = Path("/scratch5/fazh9208/clarity"
                  "/recipes/whisper_si/results")
output_dir.mkdir(
    parents=True,
    exist_ok=True,
)

hidden_dir = output_dir / "hidden_representations"
hidden_dir.mkdir(
    parents=True,
    exist_ok=True,
)

metadata_csv = output_dir / "metadata.csv"


# Load Whisper Medium

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)
print("GPU:", torch.cuda.get_device_name(0))


model = whisper.load_model("medium")
model.eval()


print("Whisper loaded.")
print("Device:", model.device)
print("Number of encoder blocks:", len(model.encoder.blocks))


# Extract encoder representations
def extract_encoder_states(audio, model):

    original_num_samples = len(audio)

    audio_padded = whisper.pad_or_trim(audio)

    mel = whisper.log_mel_spectrogram(
        audio_padded,
        n_mels=model.dims.n_mels,
    ).to(model.device)

    mel = mel.unsqueeze(0)

    # Extract encoder representations
    encoder_states = []

    def hook_fn(module, inputs, output):
        encoder_states.append(
            output.detach().cpu()
        )

    hooks = []

    for block in model.encoder.blocks:
        hooks.append(
            block.register_forward_hook(hook_fn)
        )

    with torch.no_grad():
        encoder_output = model.encoder(mel)

    for hook in hooks:
        hook.remove()

    return (
        encoder_output,
        encoder_states,
        original_num_samples,
    )

# extract decoder representations
def extract_decoder_states(
        tokens,
        encoder_output,
        model,
):
    # If Whisper produces no hypothesis,
    # decoder representations are unavailable
    if len(tokens) == 0:
        return None

    decoder_states = []

    def hook_fn(module, inputs, output):

        # Whisper decoder block may return a tuple
        if isinstance(output, tuple):
            output = output[0]

        decoder_states.append(
            output.detach().cpu()
        )

    hooks = []

    for block in model.decoder.blocks:
        hooks.append(
            block.register_forward_hook(hook_fn)
        )

    tokens = torch.tensor(
        tokens,
        dtype=torch.long,
        device=model.device,
    ).unsqueeze(0)

    with torch.no_grad():

        model.decoder(
            tokens,
            encoder_output,
        )

    for hook in hooks:
        hook.remove()

    return decoder_states

# Whisper transcription
def transcribe_audio(audio):

    result = model.transcribe(
        audio,
        language="en",
        temperature=0.0,
        fp16=False,
    )

    transcript = result["text"].strip()

    tokens = []

    for segment in result["segments"]:

        tokens.extend(
            segment["tokens"]
        )

    return (
        transcript,
        tokens,
    )

# read listeners' responses
with open(
        ground_truth,
        "r",
        encoding="utf-8",
) as file:

    ground_truth_data = json.load(
        file
    )

print(
    "\nNumber of stimuli:",
    len(ground_truth_data)
)

# produce transcription, hidden representations
metadata = []

for index, item in enumerate(
        ground_truth_data
):

    signal = item["signal"]

    enhanced_path = (
        audio_dir_enhanced
        / f"{signal}.wav"
    )

    # Skip if enhanced signal is missing
    if not enhanced_path.exists():

        print(
            f"\nMissing enhanced: "
            f"{enhanced_path}"
        )

        continue


    clean_path = get_clean_path(
        enhanced_path,
        audio_dir_clean,
    )


    # load audios
    enhanced = prepare_audio_mono(
        enhanced_path
    )

    clean = prepare_audio_mono(
        clean_path
    )

    # transcription for enhanced speech
    (
        transcript_enhanced,
        tokens_enhanced,
    ) = transcribe_audio(
        enhanced,
    )


    # transcription for clean speech
    (
        transcript_clean,
        tokens_clean,
    ) = transcribe_audio(
        clean,
    )

    # encoder representation for enhanced speech
    (
        encoder_output_enhanced,
        encoder_states_enhanced,
        num_samples_enhanced,
    ) = extract_encoder_states(
        enhanced,
        model,
    )


    # decoder representation for enhanced speech
    decoder_states_enhanced = (
        extract_decoder_states(
            tokens_enhanced,
            encoder_output_enhanced,
            model,
        )
    )


    # encoder representation for clean speech
    (
        encoder_output_clean,
        encoder_states_clean,
        num_samples_clean,
    ) = extract_encoder_states(
        clean,
        model,
    )

    # decoder representation for clean speech
    decoder_states_clean = (
        extract_decoder_states(
            tokens_clean,
            encoder_output_clean,
            model,
        )
    )


    # save hidden representations
    hidden_file = (
        hidden_dir
        / f"{signal}.pt"
    )


    torch.save(
        {
            "signal":
                signal,

            "encoder_clean":
                encoder_states_clean,

            "encoder_enhanced":
                encoder_states_enhanced,

            "decoder_clean":
                decoder_states_clean,

            "decoder_enhanced":
                decoder_states_enhanced,

            "tokens_clean":
                tokens_clean,

            "tokens_enhanced":
                tokens_enhanced,

            "num_samples_clean":
                num_samples_clean,

            "num_samples_enhanced":
                num_samples_enhanced,
        },
        hidden_file,
    )


    # save metadata
    metadata.append(
        {
            "signal":
                signal,

            "prompt":
                item["prompt"],

            "human_response":
                item["response"],

            "human_correctness":
                item["correctness"],

            "n_words":
                item["n_words"],

            "hits":
                item["hits"],

            "transcript_enhanced":
                transcript_enhanced,

            "transcript_clean":
                transcript_clean,

            "hidden_file":
                str(hidden_file),
        }
    )



    # save metadata continuously
    # Use csv module here so pandas is not required
    # during the expensive Whisper inference stage.
    with open(
            metadata_csv,
            "w",
            newline="",
            encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=metadata[0].keys(),
        )

        writer.writeheader()

        writer.writerows(
            metadata
        )



    # release GPU memory
    del encoder_output_enhanced
    del encoder_output_clean

    if torch.cuda.is_available():

        torch.cuda.empty_cache()



# finished
print(
    "\n======================================"
)

print("Finished.")

print(
    "Metadata saved to:",
    metadata_csv
)

print(
    "Hidden representations saved to:",
    hidden_dir
)