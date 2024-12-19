import math
import re
import librosa
import numpy as np
from modules.ds_tools.get_pitch import get_pitch

class EstimateMidi():
    def __init__(self, round:bool=False, rest_uv_ratio:float=0.90, pe='rmvpe'):
        self.round = round
        self.rest_uv_ratio = rest_uv_ratio
        self.pe = pe
        
    def estimate_note(
            self,
            wav_path,
            ph_dur: list,
            ph_num: list,
    ):
        timestep = 512 / 44100
        assert sum(ph_num) == len(ph_dur), f'ph_num does not sum to number of ph_dur .'
    
        total_secs = sum(ph_dur)
        waveform, _ = librosa.load(wav_path, sr=44100, mono=True)
        _, f0, uv = get_pitch(self.pe, waveform, 512, 44100)
        pitch = librosa.hz_to_midi(f0)
        if pitch.shape[0] < total_secs / timestep:
            pad = math.ceil(total_secs / timestep) - pitch.shape[0]
            pitch = np.pad(pitch, [0, pad], mode='constant', constant_values=[0, pitch[-1]])
            uv = np.pad(uv, [0, pad], mode='constant')

        word_dur = []
        i = 0
        for num in ph_num:
            word_dur.append(sum(ph_dur[i: i + num]))
            i += num

        note_seq = []
        note_dur = []
        start = 0.0
        flag = True
        for dur in word_dur:
            end = start + dur
            start_idx = math.floor(start / timestep)
            end_idx = math.ceil(end / timestep)
            word_pitch = pitch[start_idx: end_idx]
            word_uv = uv[start_idx: end_idx]
            skip_frames = min(int(0.3 * (end_idx - start_idx)), end_idx - start_idx - 1)
            keep_frames = min(int(0.5 * (end_idx - start_idx)), end_idx - start_idx - skip_frames)
            word_valid_pitch = np.extract(~word_uv[skip_frames:(skip_frames+keep_frames)] & (word_pitch[skip_frames:(skip_frames+keep_frames)] >= 0), word_pitch[skip_frames:(skip_frames+keep_frames)])
            if (len(word_valid_pitch) < (1 - self.rest_uv_ratio) * (end_idx - start_idx)) or (flag == True and ph_num[0] == 1):
                note_seq.append('rest')
                flag = False
            else:
                counts = np.bincount(np.round(word_valid_pitch).astype(np.int64))
                midi = counts.argmax()
                midi = np.mean(word_valid_pitch[(word_valid_pitch >= midi - 0.5) & (word_valid_pitch < midi + 0.5)])
                note_seq.append(librosa.midi_to_note(midi, cents=True, unicode=False))
            note_dur.append(dur)
            start = end

        if self.round:
            note_seq = self.note_seq_round(note_seq)

        note_dur = ([str(round(d, 6)) for d in note_dur])
        return(note_seq, note_dur, f0, timestep)

    def note_seq_round(self, note_seq):
        rounded_notes = []
        for note in note_seq:
            if note == "rest":
                rounded_notes.append(note)
            else:
                match = re.match(r'([A-G](?:\#|b)?\d*)(?:[-+]\d+)?', note)
                if match:
                    pitch = match.group(1)
                    rounded_notes.append(pitch)
                else:
                    print(f"Warning: Unrecognized note format: {note}")
        return rounded_notes


