import whisper
import torch

from pathlib import Path

from prepare_audio_mono import (
    get_clean_path,
    prepare_audio_mono,
)


# ============================================================
# paths
# ============================================================

root = Path(
    "/scratch5/fazh9208/clarity"
    "/recipes/whisper_si/CPC1 data"
    "/clarity_CPC1_data.v1_1/"
    "clarity_CPC1_data"
)

audio_dir_enhanced = (
    root / "clarity_data" / "HA_outputs" / "train"
)

audio_dir_clean = (
    root / "clarity_data" / "scenes"
)


# ============================================================
# test file
# ============================================================

enhanced_path = (
    audio_dir_enhanced
    / "S08887_L0201_E003.wav"
)

clean_path = get_clean_path(
    enhanced_path,
    audio_dir_clean,
)


print("Enhanced file:", enhanced_path)
print("Clean file:", clean_path)

print(
    "Enhanced exists:",
    enhanced_path.exists()
)

print(
    "Clean exists:",
    clean_path.exists()
)


# ============================================================
# load Whisper
# ============================================================

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("\nUsing device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


model = whisper.load_model(
    "medium",
    device=device,
)

model.eval()

print("Whisper loaded.")


# ============================================================
# load audio
# ============================================================

enhanced = prepare_audio_mono(
    enhanced_path
)

clean = prepare_audio_mono(
    clean_path
)


# ============================================================
# diagnostic function
# ============================================================

def test_transcription(
        audio,
        name,
):

    print(
        "\n========================================"
    )

    print(name)

    print(
        "========================================"
    )

    result = model.transcribe(
        audio,
        language="en",
        temperature=0.0,
        fp16=False,
    )

    print("\nFull text:")

    print(
        repr(result["text"])
    )

    print(
        "\nNumber of segments:",
        len(result["segments"])
    )


    # --------------------------------------------------------
    # collect tokens exactly as current main code does
    # --------------------------------------------------------

    tokens = []


    for i, segment in enumerate(
            result["segments"]
    ):

        print(
            f"\n--- Segment {i} ---"
        )

        print(
            "text:",
            repr(segment["text"])
        )

        print(
            "tokens:",
            segment["tokens"]
        )

        print(
            "number of tokens:",
            len(segment["tokens"])
        )

        print(
            "no_speech_prob:",
            segment.get(
                "no_speech_prob"
            )
        )

        print(
            "avg_logprob:",
            segment.get(
                "avg_logprob"
            )
        )

        print(
            "compression_ratio:",
            segment.get(
                "compression_ratio"
            )
        )

        tokens.extend(
            segment["tokens"]
        )


    # --------------------------------------------------------
    # final tokens passed to decoder
    # --------------------------------------------------------

    print(
        "\nFinal collected tokens:"
    )

    print(tokens)

    print(
        "Final token length:",
        len(tokens)
    )


    if len(tokens) == 0:

        print(
            "\nWARNING: EMPTY TOKEN LIST!"
        )

    else:

        print(
            "\nToken list is NOT empty."
        )


    return (
        result,
        tokens,
    )


# ============================================================
# test enhanced
# ============================================================

result_enhanced, tokens_enhanced = (
    test_transcription(
        enhanced,
        "ENHANCED",
    )
)


# ============================================================
# test clean
# ============================================================

result_clean, tokens_clean = (
    test_transcription(
        clean,
        "CLEAN",
    )
)


# ============================================================
# final comparison
# ============================================================

print(
    "\n========================================"
)

print(
    "FINAL COMPARISON"
)

print(
    "========================================"
)


print(
    "Enhanced transcript:",
    repr(result_enhanced["text"])
)

print(
    "Enhanced token length:",
    len(tokens_enhanced)
)


print(
    "\nClean transcript:",
    repr(result_clean["text"])
)

print(
    "Clean token length:",
    len(tokens_clean)
)


if (
    len(tokens_enhanced) > 0
    and len(tokens_clean) > 0
):

    print(
        "\nBoth enhanced and clean "
        "have valid tokens."
    )

else:

    print(
        "\nAt least one side has "
        "an EMPTY token list."
    )

    # Enhanced
    # file: / scratch5 / fazh9208 / clarity / recipes / whisper_si / CPC1
    # data / clarity_CPC1_data.v1_1 / clarity_CPC1_data / clarity_data / HA_outputs / train / S08887_L0201_E003.wav
    # Clean
    # file: / scratch5 / fazh9208 / clarity / recipes / whisper_si / CPC1
    # data / clarity_CPC1_data.v1_1 / clarity_CPC1_data / clarity_data / scenes / S08887_target.wav
    # Enhanced
    # exists: True
    # Clean
    # exists: True
    #
    # Using
    # device: cuda
    # GPU: NVIDIA
    # RTX
    # A5000
    # Whisper
    # loaded.
    # S08887_L0201_E003.wav: shape = (2, 212800), sr = 32000
    # S08887_target.wav: shape = (284445,), sr = 44100
    #
    # == == == == == == == == == == == == == == == == == == == ==
    # ENHANCED
    # == == == == == == == == == == == == == == == == == == == ==
    #
    # Full
    # text:
    # ''
    #
    # Number
    # of
    # segments: 0
    #
    # Final
    # collected
    # tokens:
    # []
    # Final
    # token
    # length: 0
    #
    # WARNING: EMPTY
    # TOKEN
    # LIST!
    #
    # == == == == == == == == == == == == == == == == == == == ==
    # CLEAN
    # == == == == == == == == == == == == == == == == == == == ==
    #
    # Full
    # text:
    # " But he'll get it first thing in the morning."
    #
    # Number
    # of
    # segments: 1
    #
    # --- Segment
    # 0 - --
    # text: " But he'll get it first thing in the morning."
    # tokens: [50364, 583, 415, 603, 483, 309, 700, 551, 294, 264, 2446, 13, 50584]
    # number
    # of
    # tokens: 13
    # no_speech_prob: 0.2955455183982849
    # avg_logprob: -0.30444373403276714
    # compression_ratio: 0.8979591836734694
    #
    # Final
    # collected
    # tokens:
    # [50364, 583, 415, 603, 483, 309, 700, 551, 294, 264, 2446, 13, 50584]
    # Final
    # token
    # length: 13
    #
    # Token
    # list is NOT
    # empty.
    #
    # == == == == == == == == == == == == == == == == == == == ==
    # FINAL
    # COMPARISON
    # == == == == == == == == == == == == == == == == == == == ==
    # Enhanced
    # transcript: ''
    # Enhanced
    # token
    # length: 0
    #
    # Clean
    # transcript: " But he'll get it first thing in the morning."
    # Clean
    # token
    # length: 13
    #
    # At
    # least
    # one
    # side
    # has
    # an
    # EMPTY
    # token
    # list.