from pathlib import Path
import time
import numpy as np
import librosa
import soundfile
from scipy.ndimage import maximum_filter1d, uniform_filter1d
 
def timeit(func):
    def run(*args, **kwargs):
        t = time.time()
        res = func(*args, **kwargs)
        print('executing \'%s\' cost %.3fs' % (func.__name__, time.time() - t))
        return res
    return run
 
def _window_maximum(arr, win_sz):
    return maximum_filter1d(arr, size=win_sz)[win_sz // 2: win_sz // 2 + arr.shape[0] - win_sz + 1]
 
def _window_rms(arr, win_sz):
    filtered = np.sqrt(uniform_filter1d(np.power(arr, 2), win_sz) - np.power(uniform_filter1d(arr, win_sz), 2))
    return filtered[win_sz // 2: win_sz // 2 + arr.shape[0] - win_sz + 1]
 
def level2db(levels, eps=1e-12):
    return 20 * np.log10(np.clip(levels, a_min=eps, a_max=1))
 
def _apply_slice(audio, begin, end):
    if len(audio.shape) > 1:
        return audio[:, begin: end]
    else:
        return audio[begin: end]
 
class Slicer:
    def __init__(self, sr, db_threshold=-40, min_length=5000, win_l=300, win_s=20, max_silence_kept=500):
        self.db_threshold = db_threshold
        self.min_samples = round(sr * min_length / 1000)
        self.win_ln = round(sr * win_l / 1000)
        self.win_sn = round(sr * win_s / 1000)
        self.max_silence = round(sr * max_silence_kept / 1000)
        if not self.min_samples >= self.win_ln >= self.win_sn:
            raise ValueError('The following condition must be satisfied: min_length >= win_l >= win_s')
        if not self.max_silence >= self.win_sn:
            raise ValueError('The following condition must be satisfied: max_silence_kept >= win_s')
 
    @timeit
    def slice(self, audio):
        samples = audio if len(audio.shape) == 1 else librosa.to_mono(audio)
        if samples.shape[0] <= self.min_samples:
            return [audio]
        abs_amp = np.abs(samples - np.mean(samples))
        win_max_db = level2db(_window_maximum(abs_amp, win_sz=self.win_ln))
        sil_tags = []
        left = right = 0
        while right < win_max_db.shape[0]:
            if win_max_db[right] < self.db_threshold:
                right += 1
            elif left == right:
                left += 1
                right += 1
            else:
                if left == 0:
                    split_loc_l = left
                else:
                    sil_left_n = min(self.max_silence, (right + self.win_ln - left) // 2)
                    rms_db_left = level2db(_window_rms(samples[left: left + sil_left_n], win_sz=self.win_sn))
                    split_win_l = left + np.argmin(rms_db_left)
                    split_loc_l = split_win_l + np.argmin(abs_amp[split_win_l: split_win_l + self.win_sn])
                if len(sil_tags) != 0 and split_loc_l - sil_tags[-1][1] < self.min_samples and right < win_max_db.shape[0] - 1:
                    right += 1
                    left = right
                    continue
                if right == win_max_db.shape[0] - 1:
                    split_loc_r = right + self.win_ln
                else:
                    sil_right_n = min(self.max_silence, (right + self.win_ln - left) // 2)
                    rms_db_right = level2db(_window_rms(samples[right + self.win_ln - sil_right_n: right + self.win_ln], win_sz=self.win_sn))
                    split_win_r = right + self.win_ln - sil_right_n + np.argmin(rms_db_right)
                    split_loc_r = split_win_r + np.argmin(abs_amp[split_win_r: split_win_r + self.win_sn])
                sil_tags.append((split_loc_l, split_loc_r))
                right += 1
                left = right
        if left != right:
            sil_left_n = min(self.max_silence, (right + self.win_ln - left) // 2)
            rms_db_left = level2db(_window_rms(samples[left: left + sil_left_n], win_sz=self.win_sn))
            split_win_l = left + np.argmin(rms_db_left)
            split_loc_l = split_win_l + np.argmin(abs_amp[split_win_l: split_win_l + self.win_sn])
            sil_tags.append((split_loc_l, samples.shape[0]))
        if len(sil_tags) == 0:
            return [audio]
        else:
            chunks = []
            if sil_tags[0][0] > 0:
                chunks.append(_apply_slice(audio, 0, sil_tags[0][0]))
            for i in range(0, len(sil_tags) - 1):
                chunks.append(_apply_slice(audio, sil_tags[i][1], sil_tags[i + 1][0]))
            if sil_tags[-1][1] < samples.shape[0] - 1:
                chunks.append(_apply_slice(audio, sil_tags[-1][1], samples.shape[0]))
            return chunks
 
def slice_audio(audio_path, out_dir, db_threshold=-40, min_length=3000, win_l=200, win_s=16, max_silence_kept=500):
    audio_path = Path(audio_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audio, sr = librosa.load(audio_path, sr=None)
    slicer = Slicer(sr=sr, db_threshold=db_threshold, min_length=min_length, win_l=win_l, win_s=win_s, max_silence_kept=max_silence_kept)
    chunks = slicer.slice(audio)
    base_name = Path(audio_path).stem
    for i, chunk in enumerate(chunks):
        output_path = out_dir / f'{base_name}_{i}.wav'
        soundfile.write(output_path, chunk, sr)
