from deepspeech import Model
from scipy.io import wavfile


class EnglishASR(object):
    def __init__(self, model, scorer):
        self.model = Model(model)
        self.model.enableExternalScorer(scorer)

    def recognize(self, wav_path):
        fs, audio = wavfile.read(wav_path)
        assert fs == 16000
        result = self.model.stt(audio)
        return result


if __name__ == "__main__":
    # Use your own wav files for tests.
    wav = "audio_example/001.wav"
    asr = EnglishASR(
        "./models_en/deepspeech-0.9.3-models.pbmm",
        "./models_en/deepspeech-0.9.3-models.scorer",
    )
    res = asr.recognize(wav)
    print(res)
