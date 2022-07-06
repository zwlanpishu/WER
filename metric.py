import os
import re
import librosa
import glob
import argparse
import soundfile as sf
import jiwer
from tqdm import tqdm

from asr_cn import MandarinASR
from asr_en import EnglishASR


def preprocess(audios):
    wavs = glob.glob(os.path.join(audios, "*.wav"))
    for wav in tqdm(wavs):
        data, sr = librosa.load(wav, 16000)
        sf.write(wav, data, sr)
    print("All audios have been preprocessed.")


def stt(audios, lang, model, scorer):
    wavs = sorted(glob.glob(os.path.join(audios, "*.wav")))
    if lang == "en":
        print("You are using the english asr model.")
        model = EnglishASR(model, scorer)
    elif lang == "cn":
        print("You are using the mandarin asr model.")
        # The mandarin asr is an online version, therefore no specific model path needed.
        model = MandarinASR()
    else:
        raise ("Args error!")

    txts = []
    for wav in tqdm(wavs, desc="recognizing the audios: "):
        txt = model.recognize(wav)
        txts.append(txt)
    return txts


def metric(ref_trans, asr_trans, lang):
    if lang == "en":
        transformation = jiwer.Compose(
            [
                jiwer.Strip(),
                jiwer.ToLowerCase(),
                jiwer.RemoveWhiteSpace(replace_by_space=True),
                jiwer.RemoveMultipleSpaces(),
                jiwer.RemoveEmptyStrings(),
                jiwer.RemovePunctuation(),
            ]
        )

        assert len(ref_trans) == len(asr_trans)
        ref_trans = transformation(ref_trans)
        asr_trans = transformation(asr_trans)
        cer = jiwer.cer(ref_trans, asr_trans)
        wer = jiwer.wer(ref_trans, asr_trans)
    elif lang == "cn":
        del_symblos = re.compile(r"[^\u4e00-\u9fa5]+")
        for idx in range(len(asr_trans)):
            sentence = re.sub(del_symblos, "", asr_trans[idx])
            sentence = list(sentence)
            sentence = " ".join(sentence)
            asr_trans[idx] = sentence

            sentence = re.sub(del_symblos, "", ref_trans[idx])
            sentence = list(sentence)
            sentence = " ".join(sentence)
            ref_trans[idx] = sentence
        asr_valid = set(asr_trans)
        assert len(asr_valid) == len(asr_trans)
        cer = None
        wer = jiwer.wer(ref_trans, asr_trans)

    else:
        raise ("Args error!")
    return cer, wer


def main(args):
    preprocess(args.input)
    asr_trans = stt(args.input, args.language, args.model, args.scorer)
    with open(args.reference) as f:
        ref_trans = [line.strip() for line in f]

    assert len(asr_trans) == len(ref_trans)
    print("Totally {} sentences are compared.".format(len(asr_trans)))
    cer, wer = metric(ref_trans, asr_trans, args.language)
    print(f"The final CER: {cer}, WER: {wer}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--language", default="en")
    parser.add_argument("-i", "--input", default="audios", help="Experimental audios.")
    parser.add_argument(
        "-m",
        "--model",
        default="./models_en/deepspeech-0.9.3-models.pbmm",
        help="ASR model employed.",
    )
    parser.add_argument(
        "-s",
        "--scorer",
        default="./models_en/deepspeech-0.9.3-models.scorer",
        help="ASR scorer employed.",
    )
    parser.add_argument(
        "-r",
        "--reference",
        default="./ref_trans.txt",
        help="The reference transcription for caculation metric.",
    )
    args = parser.parse_args()
    main(args)
