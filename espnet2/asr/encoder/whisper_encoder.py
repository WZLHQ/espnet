import copy
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from typeguard import typechecked

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.specaug.specaug import SpecAug


class OpenAIWhisperEncoder(AbsEncoder):
    """Transformer-based Speech Encoder from OpenAI's Whisper Model:

    URL: https://github.com/openai/whisper
    """

    @typechecked
    def __init__(
        self,
        input_size: int = 1,
        dropout_rate: float = 0.0,
        whisper_model: str = "small",
        download_dir: Optional[str] = None,
        use_specaug: bool = False,
        specaug_conf: Union[dict, None] = None,
        do_pad_trim: bool = False,
        model_twins=False,
        same_twins=False,
        use_adapterH=False,
        adapter_dim=35,
        downsample_list=None,
        bottleneck_list=None,
    ):
        try:
            import whisper
            from whisper.audio import HOP_LENGTH, N_FFT, N_MELS, N_SAMPLES
        except Exception as e:
            print("Error: whisper is not properly installed.")
            print(
                "Please install whisper with: cd ${MAIN_ROOT}/tools &&",
                "./installers/install_whisper.sh",
            )
            raise e

        super().__init__()

        self.model_twins=model_twins
        self.same_twins=same_twins

        self.n_fft = N_FFT
        self.win_length = N_FFT
        self.hop_length = HOP_LENGTH
        self.n_mels = N_MELS

        self.mel_filters = whisper.audio.mel_filters

        # note that originally Whisper doesn't use dropouts
        self.dropout = torch.nn.Dropout(dropout_rate)

        assert whisper_model in whisper.available_models()
        _model = whisper.load_model(
            whisper_model, download_root=download_dir, device="cpu", use_adapterH=use_adapterH, adapter_dim=adapter_dim, downsample_list=downsample_list, bottleneck_list=bottleneck_list
        )

        self.use_adapterH=use_adapterH

        # we define "self.encoders" as student encoder
        self.encoders = copy.deepcopy(_model.encoder)
        self.encoders.train()

        if self.model_twins:
            if self.same_twins:
                self.teacher_blocks=self.encoders.blocks
            else:
                # student encoder and teacher encoder share the "_model.encoder.blocks"-beside modules
                # thus, we only deepcopy the _model.encoder.blocks as teacher
                self.teacher_blocks = copy.deepcopy(_model.encoder.blocks)
                self.teacher_blocks.train()

        del _model

        if use_specaug:
            self.specaug = SpecAug(**specaug_conf)
        else:
            self.specaug = None

        self.do_pad_trim = do_pad_trim
        self.pad_samples = N_SAMPLES

    def output_size(self) -> int:
        return self.encoders.ln_post.normalized_shape[-1]

    def pad_or_trim(
        self,
        array: torch.Tensor,
        length: int,
        axis: int = -1,
    ) -> torch.Tensor:
        """Pad or trim the audio array to N_SAMPLES.

        Used in zero-shot inference cases.
        """
        if array.shape[axis] > length:
            array = array.index_select(
                dim=axis, index=torch.arange(length).to(array.device)
            )

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])

        return array

    def log_mel_spectrogram(
        self,
        audio: torch.Tensor,
        ilens: torch.Tensor = None,
    ) -> torch.Tensor:
        """Use log-mel spectrogram computation native to Whisper training"""
        window = torch.hann_window(self.win_length).to(audio.device)
        stft = torch.stft(
            audio, self.n_fft, self.hop_length, window=window, return_complex=True
        )

        # whisper deletes the last frame by default (Shih-Lun)
        magnitudes = stft[..., :-1].abs() ** 2

        filters = self.mel_filters(audio.device, self.n_mels)
        mel_spec = filters @ magnitudes

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()

        if ilens is not None:
            olens = ilens // self.hop_length
        else:
            olens = None

        log_spec = torch.maximum(
            log_spec,
            log_spec.view(audio.size(0), -1).max(dim=-1)[0][:, None, None] - 8.0,
        )
        log_spec = (log_spec + 4.0) / 4.0

        return log_spec, olens

    def whisper_encode(
        self,
        input: torch.Tensor,
        ilens: torch.Tensor = None,
        normal_input=None,
        mormal_ilens=None,
    ) -> torch.Tensor:
        
        if self.model_twins:

            #----------start forward student----------------#
            x = F.gelu(self.encoders.conv1(input))
            x = F.gelu(self.encoders.conv2(x))
            x = x.permute(0, 2, 1)

            n_frames = x.size(1)
            max_pos = self.encoders.positional_embedding.size(0)
            if n_frames <= max_pos:
                x = (x + self.encoders.positional_embedding[: x.size(1), :]).to(x.dtype)
            else:
                # due to positional encoding, audios >30 sec won't be accepted
                x = x[:, :max_pos, :] + self.encoders.positional_embedding

            # forward of student encoder blocks
            x = self.dropout(x)
            for layer, block in enumerate(self.encoders.blocks):
                x = block(x)
                if layer < len(self.encoders.blocks) - 1:
                    x = self.dropout(x)
            x_student = self.encoders.ln_post(x)

            if ilens is not None:
                olens = (
                    1
                    + (
                        ilens
                        - self.encoders.conv2.kernel_size[0]
                        + 2 * self.encoders.conv2.padding[0]
                    )
                    // self.encoders.conv2.stride[0]
                )
                olens_student = torch.clamp(olens, max=max_pos)
            else:
                olens_student = None
            #----------end forward student----------------#



            #----------start forward teacher----------------#
            x = F.gelu(self.encoders.conv1(normal_input))
            x = F.gelu(self.encoders.conv2(x))
            x = x.permute(0, 2, 1)

            n_frames = x.size(1)
            max_pos = self.encoders.positional_embedding.size(0)
            if n_frames <= max_pos:
                x = (x + self.encoders.positional_embedding[: x.size(1), :]).to(x.dtype)
            else:
                # due to positional encoding, audios >30 sec won't be accepted
                x = x[:, :max_pos, :] + self.encoders.positional_embedding

            # forward of teacher encoder blocks
            x = self.dropout(x)
            for layer, block in enumerate(self.teacher_blocks):
                x = block(x)
                if layer < len(self.teacher_blocks) - 1:
                    x = self.dropout(x)
            x_teacher = self.encoders.ln_post(x)

            if mormal_ilens is not None:
                olens = (
                    1
                    + (
                        mormal_ilens
                        - self.encoders.conv2.kernel_size[0]
                        + 2 * self.encoders.conv2.padding[0]
                    )
                    // self.encoders.conv2.stride[0]
                )
                olens_teacher = torch.clamp(olens, max=max_pos)
            else:
                olens_teacher = None
            #----------end forward teacher----------------#

            return x_student, olens_student, x_teacher, olens_teacher
        
        else:

            x = F.gelu(self.encoders.conv1(input))
            x = F.gelu(self.encoders.conv2(x))
            x = x.permute(0, 2, 1)

            n_frames = x.size(1)
            max_pos = self.encoders.positional_embedding.size(0)
            if n_frames <= max_pos:
                x = (x + self.encoders.positional_embedding[: x.size(1), :]).to(x.dtype)
            else:
                # due to positional encoding, audios >30 sec won't be accepted
                x = x[:, :max_pos, :] + self.encoders.positional_embedding

            x = self.dropout(x)

            for layer, block in enumerate(self.encoders.blocks):
                x = block(x)
                if layer < len(self.encoders.blocks) - 1:
                    x = self.dropout(x)

            x = self.encoders.ln_post(x)

            if ilens is not None:
                olens = (
                    1
                    + (
                        ilens
                        - self.encoders.conv2.kernel_size[0]
                        + 2 * self.encoders.conv2.padding[0]
                    )
                    // self.encoders.conv2.stride[0]
                )
                olens = torch.clamp(olens, max=max_pos)
            else:
                olens = None

            return x, olens, None, None

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        normal_xs_pad=None,
        normal_ilens=None,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if self.do_pad_trim:
            xs_pad = self.pad_or_trim(xs_pad, self.pad_samples)
            if normal_xs_pad is not None:
                normal_xs_pad = self.pad_or_trim(normal_xs_pad, self.pad_samples)

        feats, feats_lens = self.log_mel_spectrogram(xs_pad, ilens)
        if normal_xs_pad is not None:
            normal_feats, normal_feats_lens = self.log_mel_spectrogram(normal_xs_pad, normal_ilens)
        else:
            normal_feats, normal_feats_lens = None, None

        if self.specaug is not None and self.encoders.training:
            feats = torch.transpose(feats, 1, 2)
            feats, feats_lens = self.specaug(feats, feats_lens)
            feats = torch.transpose(feats, 1, 2)
            if normal_xs_pad is not None:
                normal_feats = torch.transpose(normal_feats, 1, 2)
                normal_feats, normal_feats_lens = self.specaug(normal_feats, normal_feats_lens)
                normal_feats = torch.transpose(normal_feats, 1, 2)

        xs_pad, olens, normal_xs_pad, normal_olens = self.whisper_encode(feats, feats_lens, normal_feats, normal_feats_lens)

        return xs_pad, olens, normal_xs_pad, normal_olens, None
