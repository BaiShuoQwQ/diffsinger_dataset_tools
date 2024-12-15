import pathlib
import click
import yaml
from typing import Any
import lightning as pl
import numpy as np
import torch
import torch.nn as nn
import modules.AP_detector
import modules.g2p
from modules.layer.backbone.unet import UNetBackbone
from modules.layer.block.resnet_block import ResidualBasicBlock
from modules.layer.scaling.stride_conv import DownSampling, UpSampling
from modules.utils.get_melspec import MelSpecExtractor
from modules.utils.load_wav import load_wav
from modules.utils.plot import plot_for_valid
from modules.utils.export_tool import Exporter
from modules.utils.post_processing import post_processing
from einops import repeat

def forward_pass(T, S, prob_log, not_edge_prob_log, edge_prob_log, curr_ph_max_prob_log, dp, backtrack_s, ph_seq_id,
                 prob3_pad_len):
    for t in range(1, T):
        # [t-1,s] -> [t,s]
        prob1 = dp[t - 1, :] + prob_log[t, :] + not_edge_prob_log[t]

        prob2 = np.empty(S, dtype=np.float32)
        prob2[0] = -np.inf
        for i in range(1, S):
            prob2[i] = (
                    dp[t - 1, i - 1]
                    + prob_log[t, i - 1]
                    + edge_prob_log[t]
                    + curr_ph_max_prob_log[i - 1] * (T / S)
            )

        # [t-1,s-2] -> [t,s]
        prob3 = np.empty(S, dtype=np.float32)
        for i in range(prob3_pad_len):
            prob3[i] = -np.inf
        for i in range(prob3_pad_len, S):
            if i - prob3_pad_len + 1 < S - 1 and ph_seq_id[i - prob3_pad_len + 1] != 0:
                prob3[i] = -np.inf
            else:
                prob3[i] = (
                        dp[t - 1, i - prob3_pad_len]
                        + prob_log[t, i - prob3_pad_len]
                        + edge_prob_log[t]
                        + curr_ph_max_prob_log[i - prob3_pad_len] * (T / S)
                )

        stacked_probs = np.empty((3, S), dtype=np.float32)
        for i in range(S):
            stacked_probs[0, i] = prob1[i]
            stacked_probs[1, i] = prob2[i]
            stacked_probs[2, i] = prob3[i]

        for i in range(S):
            max_idx = 0
            max_val = stacked_probs[0, i]
            for j in range(1, 3):
                if stacked_probs[j, i] > max_val:
                    max_val = stacked_probs[j, i]
                    max_idx = j
            dp[t, i] = max_val
            backtrack_s[t, i] = max_idx

        for i in range(S):
            if backtrack_s[t, i] == 0:
                curr_ph_max_prob_log[i] = max(curr_ph_max_prob_log[i], prob_log[t, i])
            elif backtrack_s[t, i] > 0:
                curr_ph_max_prob_log[i] = prob_log[t, i]

        for i in range(S):
            if ph_seq_id[i] == 0:
                curr_ph_max_prob_log[i] = 0

    return dp, backtrack_s, curr_ph_max_prob_log

class LitForcedAlignmentTask(pl.LightningModule):
    def __init__(
        self,
        vocab_text,
        model_config,
        melspec_config,
        optimizer_config,
        loss_config,
        data_augmentation_enabled,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.vocab = yaml.safe_load(vocab_text)
        self.backbone = UNetBackbone(
            melspec_config["n_mels"],
            model_config["hidden_dims"],
            model_config["hidden_dims"],
            ResidualBasicBlock,
            DownSampling,
            UpSampling,
            down_sampling_factor=model_config["down_sampling_factor"],  # 3
            down_sampling_times=model_config["down_sampling_times"],  # 7
            channels_scaleup_factor=model_config["channels_scaleup_factor"],  # 1.5
        )
        self.head = nn.Linear(
            model_config["hidden_dims"], self.vocab["<vocab_size>"] + 2)
        self.melspec_config = melspec_config  # Required for inference
        self.optimizer_config = optimizer_config

        self.pseudo_label_ratio = loss_config["function"]["pseudo_label_ratio"]
        self.pseudo_label_auto_theshold = 0.5

        # get_melspec
        self.get_melspec = None

        self.inference_mode = "force"


    def _decode(self, ph_seq_id, ph_prob_log, edge_prob):
        # ph_seq_id: (S)
        # ph_prob_log: (T, vocab_size)
        # edge_prob: (T,2)
        T = ph_prob_log.shape[0]
        S = len(ph_seq_id)
        # not_SP_num = (ph_seq_id > 0).sum()
        prob_log = ph_prob_log[:, ph_seq_id]

        edge_prob_log = np.log(edge_prob + 1e-6).astype("float32")
        not_edge_prob_log = np.log(1 - edge_prob + 1e-6).astype("float32")

        # init
        curr_ph_max_prob_log = np.full(S, -np.inf)
        dp = np.full((T, S), -np.inf, dtype="float32")  # (T, S)
        backtrack_s = np.full_like(dp, -1, dtype="int32")
        # 如果mode==forced，只能从SP开始或者从第一个音素开始
        if self.inference_mode == "force":
            dp[0, 0] = prob_log[0, 0]
            curr_ph_max_prob_log[0] = prob_log[0, 0]
            if ph_seq_id[0] == 0 and prob_log.shape[-1] > 1:
                dp[0, 1] = prob_log[0, 1]
                curr_ph_max_prob_log[1] = prob_log[0, 1]
        # 如果mode==match，可以从任意音素开始
        elif self.inference_mode == "match":
            for i, ph_id in enumerate(ph_seq_id):
                dp[0, i] = prob_log[0, i]
                curr_ph_max_prob_log[i] = prob_log[0, i]

        # forward
        prob3_pad_len = 2 if S >= 2 else 1
        dp, backtrack_s, curr_ph_max_prob_log = forward_pass(
            T, S, prob_log, not_edge_prob_log, edge_prob_log, curr_ph_max_prob_log, dp, backtrack_s, ph_seq_id,
            prob3_pad_len
        )

        # backward
        ph_idx_seq = []
        ph_time_int = []
        frame_confidence = []
        # 如果mode==forced，只能从最后一个音素或者SP结束
        if self.inference_mode == "force":
            if S >= 2 and dp[-1, -2] > dp[-1, -1] and ph_seq_id[-1] == 0:
                s = S - 2
            else:
                s = S - 1
        # 如果mode==match，可以从任意音素结束
        elif self.inference_mode == "match":
            s = np.argmax(dp[-1, :])
        else:
            raise ValueError("inference_mode must be 'force' or 'match'")

        for t in np.arange(T - 1, -1, -1):
            assert backtrack_s[t, s] >= 0 or t == 0
            frame_confidence.append(dp[t, s])
            if backtrack_s[t, s] != 0:
                ph_idx_seq.append(s)
                ph_time_int.append(t)
                s -= backtrack_s[t, s]
        ph_idx_seq.reverse()
        ph_time_int.reverse()
        frame_confidence.reverse()
        frame_confidence = np.exp(
            np.diff(
                np.pad(frame_confidence, (1, 0), "constant", constant_values=0.0), 1
            )
        )

        return (
            np.array(ph_idx_seq),
            np.array(ph_time_int),
            np.array(frame_confidence),
        )

    def _infer_once(
        self,
        melspec,
        wav_length,
        ph_seq,
        word_seq=None,
        ph_idx_to_word_idx=None,
        return_ctc=False,
        return_plot=False,
    ):
        ph_seq_id = np.array([self.vocab[ph] for ph in ph_seq])
        ph_mask = np.zeros(self.vocab["<vocab_size>"])
        ph_mask[ph_seq_id] = 1
        ph_mask[0] = 1
        ph_mask = torch.from_numpy(ph_mask)
        if word_seq is None:
            word_seq = ph_seq
            ph_idx_to_word_idx = np.arange(len(ph_seq))

        # forward
        with torch.no_grad():
            (
                ph_frame_logits,  # (B, T, vocab_size)
                ph_edge_logits,  # (B, T)
                ctc_logits,  # (B, T, vocab_size)
            ) = self.forward(melspec.transpose(1, 2))
        if wav_length is not None:
            num_frames = int(
                (
                    (
                        wav_length
                        * self.melspec_config["scale_factor"]
                        * self.melspec_config["sample_rate"]
                        + 0.5
                    )
                )
                / self.melspec_config["hop_length"]
            )
            ph_frame_logits = ph_frame_logits[:, :num_frames, :]
            ph_edge_logits = ph_edge_logits[:, :num_frames]
            ctc_logits = ctc_logits[:, :num_frames, :]

        ph_mask = (
            ph_mask.to(ph_frame_logits.device).unsqueeze(0).unsqueeze(0).logical_not()
            * 1e9
        )
        ph_frame_pred = (
            torch.nn.functional.softmax(
                ph_frame_logits.float() - ph_mask.float(), dim=-1
            )
            .squeeze(0)
            .cpu()
            .numpy()
            .astype("float32")
        )
        ph_prob_log = (
            torch.log_softmax(ph_frame_logits.float() - ph_mask.float(), dim=-1)
            .squeeze(0)
            .cpu()
            .numpy()
            .astype("float32")
        )
        ph_edge_pred = (
            (torch.nn.functional.sigmoid(ph_edge_logits.float()) - 0.1) / 0.8
        ).clamp(0.0, 1.0)
        ph_edge_pred = ph_edge_pred.squeeze(0).cpu().numpy().astype("float32")
        ctc_logits = (
            ctc_logits.float().squeeze(0).cpu().numpy().astype("float32")
        )  # (ctc_logits.squeeze(0) - ph_mask)

        T, vocab_size = ph_frame_pred.shape

        # decode
        edge_diff = np.concatenate((np.diff(ph_edge_pred, axis=0), [0]), axis=0)
        edge_prob = (ph_edge_pred + np.concatenate(([0], ph_edge_pred[:-1]))).clip(0, 1)
        (
            ph_idx_seq,
            ph_time_int_pred,
            frame_confidence,
        ) = self._decode(
            ph_seq_id,
            ph_prob_log,
            edge_prob,
        )
        total_confidence = np.exp(np.mean(np.log(frame_confidence + 1e-6)) / 3)

        # postprocess
        frame_length = self.melspec_config["hop_length"] / (
            self.melspec_config["sample_rate"] * self.melspec_config["scale_factor"]
        )
        ph_time_fractional = (edge_diff[ph_time_int_pred] / 2).clip(-0.5, 0.5)
        ph_time_pred = frame_length * (
            np.concatenate(
                [
                    ph_time_int_pred.astype("float32") + ph_time_fractional,
                    [T],
                ]
            )
        )
        ph_intervals = np.stack([ph_time_pred[:-1], ph_time_pred[1:]], axis=1)

        ph_seq_pred = []
        ph_intervals_pred = []
        word_seq_pred = []
        word_intervals_pred = []

        word_idx_last = -1
        for i, ph_idx in enumerate(ph_idx_seq):
            # ph_idx只能用于两种情况：ph_seq和ph_idx_to_word_idx
            if ph_seq[ph_idx] == "SP":
                continue
            ph_seq_pred.append(ph_seq[ph_idx])
            ph_intervals_pred.append(ph_intervals[i, :])

            word_idx = ph_idx_to_word_idx[ph_idx]
            if word_idx == word_idx_last:
                word_intervals_pred[-1][1] = ph_intervals[i, 1]
            else:
                word_seq_pred.append(word_seq[word_idx])
                word_intervals_pred.append([ph_intervals[i, 0], ph_intervals[i, 1]])
                word_idx_last = word_idx
        ph_seq_pred = np.array(ph_seq_pred)
        ph_intervals_pred = np.array(ph_intervals_pred).clip(min=0, max=None)
        word_seq_pred = np.array(word_seq_pred)
        word_intervals_pred = np.array(word_intervals_pred).clip(min=0, max=None)

        # ctc decode
        ctc = None
        if return_ctc:
            ctc = np.argmax(ctc_logits, axis=-1)
            ctc_index = np.concatenate([[0], ctc])
            ctc_index = (ctc_index[1:] != ctc_index[:-1]) * ctc != 0
            ctc = ctc[ctc_index]
            ctc = np.array([self.vocab[ph] for ph in ctc if ph != 0])

        fig = None
        ph_intervals_pred_int = (
            (ph_intervals_pred / frame_length).round().astype("int32")
        )
        if return_plot:
            ph_idx_frame = np.zeros(T).astype("int32")
            last_ph_idx = 0
            for ph_idx, ph_time in zip(ph_idx_seq, ph_time_int_pred):
                ph_idx_frame[ph_time] += ph_idx - last_ph_idx
                last_ph_idx = ph_idx
            ph_idx_frame = np.cumsum(ph_idx_frame)
            args = {
                "melspec": melspec.cpu().numpy(),
                "ph_seq": ph_seq_pred,
                "ph_intervals": ph_intervals_pred_int,
                "frame_confidence": frame_confidence,
                "ph_frame_prob": ph_frame_pred[:, ph_seq_id],
                "ph_frame_id_gt": ph_idx_frame,
                "edge_prob": edge_prob,
            }
            fig = plot_for_valid(**args)

        return (
            ph_seq_pred,
            ph_intervals_pred,
            word_seq_pred,
            word_intervals_pred,
            total_confidence,
            ctc,
            fig,
        )

    def set_inference_mode(self, mode):
        self.inference_mode = mode

    def on_predict_start(self):
        if self.get_melspec is None:
            self.get_melspec = MelSpecExtractor(**self.melspec_config)

    def predict_step(self, batch, batch_idx):
        try:
            wav_path, ph_seq, word_seq, ph_idx_to_word_idx = batch
            waveform = load_wav(
                wav_path, self.device, self.melspec_config["sample_rate"]
            )
            wav_length = waveform.shape[0] / self.melspec_config["sample_rate"]
            melspec = self.get_melspec(waveform).detach().unsqueeze(0)
            melspec = (melspec - melspec.mean()) / melspec.std()
            melspec = repeat(
                melspec, "B C T -> B C (T N)", N=self.melspec_config["scale_factor"]
            )

            (
                ph_seq,
                ph_intervals,
                word_seq,
                word_intervals,
                confidence,
                _,
                _,
            ) = self._infer_once(
                melspec, wav_length, ph_seq, word_seq, ph_idx_to_word_idx, False, False
            )

            return (
                wav_path,
                wav_length,
                confidence,
                ph_seq,
                ph_intervals,
                word_seq,
                word_intervals,
            )
        except Exception as e:
            e.args += (f"{str(wav_path)}",)
            raise e

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        h = self.backbone(*args, **kwargs)
        logits = self.head(h)
        ph_frame_logits = logits[:, :, 2:]
        ph_edge_logits = logits[:, :, 0]
        ctc_logits = torch.cat([logits[:, :, [1]], logits[:, :, 3:]], dim=-1)
        return ph_frame_logits, ph_edge_logits, ctc_logits
    
def sofa_infer(
        ckpt,
        wav_folder,
        lab_folder,
        out_folder,
        mode,
        g2p,
        ap_detector,
        in_format,
        out_formats,
        save_confidence,
        **kwargs,
):
    if not g2p.endswith("G2P"):
        g2p += "G2P"
    g2p_class = getattr(modules.g2p, g2p)
    grapheme_to_phoneme = g2p_class(**kwargs)
    out_formats = [i.strip().lower() for i in out_formats.split(",")]

    if not ap_detector.endswith("APDetector"):
        ap_detector += "APDetector"
    AP_detector_class = getattr(modules.AP_detector, ap_detector)
    get_AP = AP_detector_class(**kwargs)

    grapheme_to_phoneme.set_in_format(in_format)
    dataset = grapheme_to_phoneme.get_dataset(pathlib.Path(wav_folder).rglob("*.wav"), lab_folder)

    torch.set_grad_enabled(False)
    model = LitForcedAlignmentTask.load_from_checkpoint(ckpt, strict=False)
    model.set_inference_mode(mode)
    trainer = pl.Trainer(logger=False)
    predictions = trainer.predict(model, dataloaders=dataset, return_predictions=True)

    predictions = get_AP.process(predictions)
    predictions, log = post_processing(predictions)
    exporter = Exporter(predictions, log)

    if save_confidence:
        out_formats.append('confidence')

    exporter.export(out_formats, out_folder)

    print("Output files are saved to the same folder as the input wav files.")

@click.command()
@click.option(
    "--ckpt",
    "-c",
    default=None,
    required=True,
    type=str,
    help="path to the checkpoint",
)
@click.option(
    "--folder", "-f", default="segments", type=str, help="path to the input folder"
)
@click.option(
    "--out_folder", "-f", default="segments", type=str, help="path to the output folder"
)
@click.option(
    "--mode", "-m", default="force", type=click.Choice(["force", "match"])
)  # TODO: add asr mode
@click.option(
    "--g2p", "-g", default="Dictionary", type=str, help="name of the g2p class"
)
@click.option(
    "--ap_detector",
    "-a",
    default="LoudnessSpectralcentroidAPDetector",  # "NoneAPDetector",
    type=str,
    help="name of the AP detector class",
)
@click.option(
    "--in_format",
    "-if",
    default="lab",
    required=False,
    type=str,
    help="File extension of input transcriptions. Default: lab",
)
@click.option(
    "--out_formats",
    "-of",
    default="textgrid",
    required=False,
    type=str,
    help="Types of output file, separated by comma. Supported types:"
         "textgrid(praat),"
         " htk(lab,nnsvs,sinsy),"
         " transcriptions.csv(diffsinger,trans,transcription,transcriptions)",
)
@click.option(
    "--save_confidence",
    "-sc",
    is_flag=True,
    default=False,
    show_default=True,
    help="save confidence.csv",
)
@click.option(
    "--dictionary",
    "-d",
    default="dictionary/opencpop-extension.txt",
    type=str,
    help="(only used when --g2p=='Dictionary') path to the dictionary",
)
def main(ckpt,
        wav_folder,
        lab_folder,
        out_folder,
        mode,
        g2p,
        ap_detector,
        in_format,
        out_formats,
        save_confidence,
        **kwargs,):
    sofa_infer(
        ckpt,
        wav_folder,
        lab_folder,
        out_folder,
        mode,
        g2p,
        ap_detector,
        in_format,
        out_formats,
        save_confidence,
        **kwargs,
)

if __name__ == "__main__":
    main()
