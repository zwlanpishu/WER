# -*- coding: utf-8 -*-
import os
import hashlib
import time
import json
import urllib.request
import contextlib
import random


class MandarinASR(object):
    def __init__(self):
        """
        Register infomation for online services.
        """
        self.app_key = "195d5451"
        self.dev_key = "4811cb1e437414777250201c0d9b854a"
        self.url = "http://api.hcicloud.com:8880/asr/Recognise"
        self.plantform = "asr.cloud.freetalk"
        self.format = "pcm16k16bit"
        self.request_date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.session_key = hashlib.md5(
            (self.request_date + self.dev_key).encode("utf-8")
        ).hexdigest()
        self.MAC = "7e:55:75:2f:91:4d"

    def _get_file_content(self, wav):
        dir_name = os.path.dirname(wav)
        wav_name = os.path.basename(wav)
        pcm_name = wav_name.split(".")[0] + ".pcm"
        pcm = os.path.join(dir_name, pcm_name)
        state = os.system("ffmpeg -y -i {} -f s16le -ac 1 -ar 16000 {}".format(wav, pcm))
        assert state == 0
        time.sleep(2)
        with open(pcm, "rb") as fp:
            return fp.read()

    def recognize(self, wav_path):
        data = self._get_file_content(wav_path)
        random_mac = self._random_mac()
        headers = {
            "x-app-key": self.app_key,
            "x-request-date": self.request_date,
            "x-result-format": "json",
            "x-sdk-version": "7.0",
            "x-session-key": self.session_key,
            "x-task-config": "capkey=%s,audioformat=%s,identify=%s,index=%s,addpunc=yes"
            % (
                self.plantform,
                self.format,
                random_mac + self.request_date + str(len(self.request_date)),
                "-1",
            ),
            "x-udid": "101:1234567890",
        }

        request = urllib.request.Request(url=self.url, data=data, headers=headers)
        with contextlib.closing(urllib.request.urlopen(request)) as connection:
            result = connection.read()
            result = json.loads(result)
        assert result["ResponseInfo"]["ResCode"] == "Success"
        result = result["ResponseInfo"]["Result"]["Text"]
        return result

    def _random_mac(self):
        macstring = "0123456789abcdef" * 12
        macstringlist = random.sample(macstring, 12)
        return "{0[0]}{0[1]}:{0[2]}{0[3]}:{0[4]}{0[5]}:\
            {0[6]}{0[7]}:{0[8]}{0[9]}:{0[10]}{0[11]}".format(
            macstringlist
        )


if __name__ == "__main__":
    # Use your own wav files for tests.
    wav = "audio_example/002499.wav"
    asr = MandarinASR()
    res = asr.recognize(wav)
    print(res)
