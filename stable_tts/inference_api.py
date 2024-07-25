"""Copyright: Nabarun Goswami (2024)."""
from dataclasses import asdict
import torch
from huggingface_hub import hf_hub_download
from torch import nn
import torchaudio
from transformers.modeling_utils import ModuleUtilsMixin

from stable_tts.utils.audio import LogMelSpectrogram
from stable_tts.config import ModelConfig, VocosConfig, MelConfig
from stable_tts.models.model import StableTTS
from stable_tts.vocos_pytorch.models.model import Vocos
from stable_tts.text.english import english_to_ipa2
from stable_tts.text import cleaned_text_to_sequence
from stable_tts.text import symbols
from stable_tts.datas.dataset import intersperse


class StableTTSEN(nn.Module, ModuleUtilsMixin):
    def __init__(self, step=10):
        super().__init__()

        tts_checkpoint_path = hf_hub_download(repo_id="KdaiP/StableTTS", filename="checkpoint-en_0.pt", cache_dir=None)
        vocoder_checkpoint_path = hf_hub_download(repo_id="KdaiP/StableTTS", filename="vocoder.pt", cache_dir=None)

        n_vocab = len(symbols)
        model_config = ModelConfig()
        mel_config = MelConfig()
        vocoder_config = VocosConfig()

        self.tts_model = StableTTS(n_vocab, mel_config.n_mels, **asdict(model_config))

        self.mel_extractor = LogMelSpectrogram(mel_config)

        self.vocoder = Vocos(vocoder_config, mel_config)

        self.tts_model.load_state_dict(torch.load(tts_checkpoint_path, map_location='cpu'))
        self.tts_model.eval()

        self.vocoder.load_state_dict(torch.load(vocoder_checkpoint_path, map_location='cpu'))
        self.vocoder.eval()

        self.phonemizer = english_to_ipa2

        self.sample_rate = mel_config.sample_rate

        self.step = step

    @torch.inference_mode()
    def tts_to_file(self, text, save_path, speaker_prompt=None):

        x = torch.tensor(intersperse(cleaned_text_to_sequence(self.phonemizer(text)), item=0), dtype=torch.long,
                         device=self.device).unsqueeze(0)
        x_len = torch.tensor([x.size(-1)], dtype=torch.long, device=self.device)
        waveform, sr = torchaudio.load(speaker_prompt)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        y = self.mel_extractor(waveform).to(self.device)

        # inference
        mel = self.tts_model.synthesise(x, x_len, self.step, y=y, temperature=0.667, length_scale=1)['decoder_outputs']
        audio = self.vocoder(mel)
        audio = audio.cpu()
        torchaudio.save(save_path, audio, self.sample_rate)
