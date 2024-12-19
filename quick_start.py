from pathlib import Path
import yaml
from tqdm import tqdm
import shutil
import librosa
from modules.ds_tools.loudness_norm import loudness_norm_file
from modules.ds_tools.slicer import slice_audio
from modules.ds_tools.wav2words import funasr_folder, pinyin_folder
from SOFA_infer import sofa_infer
from FBL_infer import export
from modules.ds_tools.filter_bad import move_bad
from modules.ds_tools.textgrid2ds import textgrid2ds
from ds_dataset import ds_dataset



def quick_start():
    with open('tools_config.yaml', 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
        singer = str(data.get('singer'))
        dictionary = str(data.get('dictionary'))
        SOFA_ckpt = str(data.get('SOFA_ckpt'))
        FBL_ckpt = str(data.get('FBL_ckpt'))
        SOME_ckpt = str(data.get('SOME_ckpt'))

    work_audios = Path('work_audios')
    base_path = Path('data') / singer
    original = base_path / 'original'
    norm = base_path / 'norm'
    wavs = base_path / 'wavs'
    lab = base_path / 'lab'
    textgrids = base_path / 'textgrids'
    bad =base_path / 'bad'
    ds = base_path / 'ds'
    dataset = base_path / (singer + '_dataset')
    
    for folder in [work_audios, base_path, original, norm, wavs, lab, textgrids, bad, ds, dataset]:
        folder.mkdir(parents=True, exist_ok=True)

    #移动音频
    if any(work_audios.iterdir()) and any(original.iterdir()):
        print(f"Error: folder {original} is not empty, change singer in tools_config.yaml or clear folder")
        exit(1)
    for file in work_audios.glob('*.wav'):
        destination_file = original / file.name
        shutil.move(file, destination_file)

    #响度匹配
    for f in tqdm(original.glob('*.wav')):
        norm_audio = norm / f.name
        loudness_norm_file(f, norm_audio)
    print("Step 1: loudness_norm complete")

    #音频切片
    for f in tqdm(norm.glob('*.wav')):
        if librosa.get_duration(filename=str(f)) > 10:
            slice_audio(f, wavs, db_threshold=-32)
    for f in tqdm(wavs.glob('*.wav')):
        if librosa.get_duration(filename=str(f)) > 15:
            slice_audio(f, wavs, db_threshold=-24)
            f.unlink()
    print("Step 2: slice complete")

    #生成lab
    funasr_folder(wavs, lab)
    pinyin_folder(lab, lab, "dictionary/phrases_dict.txt")
    print("Step 3: lab complete")

    #生成textgrid
    sofa_infer(SOFA_ckpt, wavs, lab, textgrids, "force", "Dictionary", "NoneAPDetector", "lab", "textgrid", True, dictionary=dictionary)
    print("Step 4: SOFA complete")

    #呼吸标注
    export(FBL_ckpt, wavs, textgrids, textgrids)
    print("Step 5: FBL complete")

    #筛选标注
    confidence = textgrids / "confidence.csv"
    move_bad(wavs, textgrids, bad, confidence)
    print("Step 6: move_bad complete")

    #生成ds
    textgrid2ds(textgrids, wavs, ds, dictionary, use_some=True, some_model=SOME_ckpt)
    print("Step 7: ds complete")

    #构建数据集
    ds_dataset(wavs, ds, dataset)
    print("Step 8: dataset complete")
    print("Congratulations! All the steps have been completed.\nThis project is produced by Bai_Shuo.")

    '''
    #清理文件
    shutil.rmtree(norm)
    shutil.rmtree(wavs)
    shutil.rmtree(lab)
    shutil.rmtree(textgrids)
    shutil.rmtree(bad)
    shutil.rmtree(ds)
    '''

if __name__ == '__main__':
    quick_start()