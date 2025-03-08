from textgrid import TextGrid
import librosa  
import numpy as np
import csv
import shutil
from pathlib import Path
from modules.ds_tools.audio_tools import wav_total_length


class FilterBad():
    def __init__(self, threshold_db=-24, frame_length=4096, frame_step=512, shortest_label=0.01, noisy_length=0.6):
        self.threshold_db=threshold_db
        self.frame_length=frame_length
        self.frame_step=frame_step
        self.shortest_label = shortest_label
        self.noisy_length=noisy_length
        self.allowed_short_phones = ["p", "t", "k", "d", "l", "w", "y"]
        
    #检测是否存在过短音素；是否存在过量有声SP
    def SP_bad(self, tg_file, wav_file):
        tg = TextGrid.fromFile(tg_file)
        wav, sr = librosa.load(wav_file, sr=None) 
        phones_tier = None
        for tier in tg:
            if tier.name == 'phones':
                phones_tier = tier 
        total_silent_duration = 0
        flag = 0
        for intervals in phones_tier:
            if intervals.mark == 'SP':  
                start_sample = librosa.time_to_samples(intervals.minTime, sr=sr)  
                end_sample = librosa.time_to_samples(intervals.maxTime, sr=sr)  
                start_sample = max(0, start_sample)  
                end_sample = min(len(wav), end_sample)  
                y_segment = wav[start_sample:end_sample]  
                frames = []  
                rms_values = []  
                for i in range(0, len(y_segment) - self.frame_length + 1, self.frame_step):  
                    frame = y_segment[i:i+self.frame_length]  
                    frames.append(frame)  
                    rms_value = np.sqrt(np.mean(frame**2))  
                    rms_values.append(rms_value)
                rms_db = 20 * np.log10(np.maximum(rms_values, 1e-15))  
                silent_frames = np.where(rms_db > self.threshold_db)[0]
                silent_duration = len(silent_frames) * self.frame_step / sr  
                total_silent_duration += silent_duration
            else:
                if ((intervals.maxTime - intervals.minTime) < self.shortest_label) and (intervals.mark not in self.allowed_short_phones):
                    flag = 1
        if (flag == 1) or (total_silent_duration>self.noisy_length):
            return(True)
        else:
            return(False)
        
    #过长；过短
    def length_bad(self, wav_file, length_min, length_max):
        wav_length = librosa.get_duration(filename=str(wav_file))
        if (length_min and wav_length<length_min) or (length_max and wav_length>length_max):
            return(True)
        else:
            return(False)
    
    #有效音素过少
    def phseq_num_bad(self, tg_file, min_num):
        tg = TextGrid.fromFile(tg_file)
        phones_tier = None
        for tier in tg:
            if tier.name == 'phones':
                phones_tier = tier
        syllables = [ph.mark for ph in phones_tier]
        phseq_num = sum(1 for syllable in syllables if syllable not in ['AP', 'SP'])
        if phseq_num < min_num:
            return(True)
        else:
            return(False)

    #置信度倒序删除
    def confidence_bad(self, wav_folder, tg_folder, confidence, out_folder, ratio):
        wav_info_list = []
        with open(confidence, mode='r', newline='', encoding='utf-8') as file:
            csvreader = csv.DictReader(file)
            for row in csvreader:
                wav_name = row['name']
                confidence = float(row['confidence'])
                wav_path = wav_folder / f"{wav_name}.wav"
                if wav_path.exists():
                    wav_info_list.append((wav_path, confidence))
        total_length = wav_total_length(wav_folder)
        length_to_move = total_length * ratio
        sorted_audio_info_list = sorted(wav_info_list , key=lambda x: x[1])
        moved_length = 0
        for wav_file, confidence in sorted_audio_info_list:
            wav_length = librosa.get_duration(filename=str(wav_file))/3600
            if moved_length + wav_length <= length_to_move:
                tg_file = (tg_folder / wav_file.stem).with_suffix(".TextGrid")
                target_wav_file = out_folder / 'wavs' / wav_file.name
                target_tg_file = out_folder / 'textgrids' / tg_file.name
                shutil.move(wav_file, target_wav_file)
                shutil.move(tg_file, target_tg_file)
                moved_length += wav_length
            else:
                break

def move_bad(wav_folder, tg_folder, out_folder, confidence=None, ratio=0.05):
    wav_folder = Path(wav_folder)
    tg_folder = Path(tg_folder)
    out_folder = Path(out_folder)
    (out_folder / 'textgrids').mkdir(parents=True, exist_ok=True)
    (out_folder / 'wavs').mkdir(parents=True, exist_ok=True)
    fb = FilterBad()
    for wav_file in wav_folder.glob('*.wav'):
        tg_file = tg_folder / f"{wav_file.stem}.TextGrid"
        #tg存在
        if not tg_file.is_file():
            new_wav_file = out_folder / 'wavs' / wav_file.name
            shutil.move(wav_file, new_wav_file)
        else:
            if fb.SP_bad(tg_file, wav_file) or fb.length_bad(wav_file, 1, 20) or fb.phseq_num_bad(tg_file, 4):
                new_wav_file = out_folder / 'wavs' / wav_file.name
                new_tg_file = out_folder / 'textgrids' / tg_file.name
                shutil.move(wav_file, new_wav_file)
                shutil.move(tg_file, new_tg_file)
    print("moved bad by rules")

    if confidence:
        confidence = Path(confidence)
        fb.confidence_bad(wav_folder, tg_folder, confidence, out_folder, ratio)
        print("moved bad by confidence")

if __name__ == '__main__':
    move_bad()