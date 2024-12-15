import textgrid
import librosa  
import numpy as np
import csv
import shutil
from pathlib import Path
from modules.ds_tools.audio_tools import wav_total_length

#个人经验,经过SOFA,有严重错误的textgrid标注通常有两类特征：存在极短音素；存在有声SP
class FilterBad():
    def __init__(self, threshold_db=-24, frame_length=4096, frame_step=512, shortest_label=0.01, noisy_length=0.6):
        self.threshold_db=threshold_db
        self.frame_length=frame_length
        self.frame_step=frame_step
        self.shortest_label = shortest_label
        self.noisy_length=noisy_length

    def select_bad(self, tg_file, wav_file):
        tg = textgrid.TextGrid.fromFile(tg_file)
        wav, sr = librosa.load(wav_file, sr=None) 
        phones_tier = None
        for tier in tg:
            if tier.name == 'phones':
                phones_tier = tier 
        total_silent_duration = 0
        flag = 0
        for intervals in phones_tier:
            if intervals.mark == 'SP':  
                # 将时间转换为样本点  
                start_sample = librosa.time_to_samples(intervals.minTime, sr=sr)  
                end_sample = librosa.time_to_samples(intervals.maxTime, sr=sr)  

                # 确保不超出音频边界  
                start_sample = max(0, start_sample)  
                end_sample = min(len(wav), end_sample)  

                # 获取SP区间内的音频片段  
                y_segment = wav[start_sample:end_sample]  

                # 初始化帧和 RMS 值列表  
                frames = []  
                rms_values = []  

                # 分割音频为帧,白烁制作
                for i in range(0, len(y_segment) - self.frame_length + 1, self.frame_step):  
                    frame = y_segment[i:i+self.frame_length]  
                    frames.append(frame)  
                    rms_value = np.sqrt(np.mean(frame**2))  
                    rms_values.append(rms_value)  

                # 转换为分贝
                rms_db = 20 * np.log10(np.maximum(rms_values, 1e-15))  

                # 计算大于阈值的帧的总时长  
                silent_frames = np.where(rms_db > self.threshold_db)[0]
                silent_duration = len(silent_frames) * self.frame_step / sr  

                # 累加时长  
                total_silent_duration += silent_duration
            else:
                if(intervals.maxTime - intervals.minTime) < self.shortest_label:
                    flag = 1
        if (flag == 1) or (total_silent_duration>self.noisy_length):
            return(True)
        else:
            return(False)
            
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
        
def move_bad(wav_folder, tg_folder, out_folder, confidence=None, ratio=0.1):
    wav_folder = Path(wav_folder)
    tg_folder = Path(tg_folder)
    out_folder = Path(out_folder)
    (out_folder / 'textgrids').mkdir(parents=True, exist_ok=True)
    (out_folder / 'wavs').mkdir(parents=True, exist_ok=True)
    fb = FilterBad()
    for wav_file in wav_folder.glob('*.wav'):
        tg_file = tg_folder / f"{wav_file.stem}.TextGrid"
        if tg_file.is_file():
            x = fb.select_bad(tg_file, wav_file)
            if x:
                new_wav_file = out_folder / 'wavs' / wav_file.name
                new_tg_file = out_folder / 'textgrids' / tg_file.name
                shutil.move(wav_file, new_wav_file)
                shutil.move(tg_file, new_tg_file)
        else:
            new_tg_file = out_folder / 'wavs' / tg_file.name
            shutil.move(wav_file, new_wav_file)
    print("moved bad by rules")

    if confidence:
        confidence = Path(confidence)
        fb.confidence_bad(wav_folder, tg_folder, confidence, out_folder, ratio)
        print("moved bad by confidence")