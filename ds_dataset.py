import csv
import pathlib
import librosa
import soundfile
from tqdm import tqdm
from modules.ds_tools.ds2csv import ds2csv

def ds_dataset(wav_folder, ds_folder, dataset_folder, skip_silence_insertion=False, wav_subtype="PCM_16"):
    wav_folder = pathlib.Path(wav_folder)
    ds_folder = pathlib.Path(ds_folder)
    dataset_folder = pathlib.Path(dataset_folder)
    filelist = list(wav_folder.glob('*.wav'))
    (dataset_folder / 'raw' / 'wavs').mkdir(parents=True, exist_ok=True)
    
    tran = dataset_folder / 'raw' / 'transcriptions.csv'
    ds2csv(ds_folder, tran, True)

    
    with open(tran, 'r', newline='', encoding='utf-8') as f:
        csvfile = csv.DictReader(f)
    samplerate = 44100
    min_sil = int(0.1 * samplerate)
    max_sil = int(0.5 * samplerate)
    for wavfile in tqdm(filelist):
        y, _ = librosa.load(wavfile, sr=samplerate, mono=True)
        soundfile.write(dataset_folder / 'raw' / 'wavs' / wavfile.name, y, samplerate, subtype=wav_subtype)
   
    #TODO:silence_insertion