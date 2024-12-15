from pathlib import Path
import yaml
from tqdm import tqdm
import shutil
from modules.ds_tools.loudness_norm import loudness_norm_file
from modules.ds_tools.slicer import slice_audio
from modules.ds_tools.wav2words import funasr_folder, pinyin_folder
from SOFA_infer import sofa_infer
from FBL_infer import export
from modules.ds_tools.filter_bad import move_bad
from modules.ds_tools.textgrid2ds import textgrid2ds
from modules.ds_tools.ds2csv import ds2csv



def quick_start():
    with open('tools_config.yaml', 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
        singer = str(data.get('singer'))
        dictionary = str(data.get('dictionary'))
        SOFA_ckpt = str(data.get('SOFA_ckpt'))
        FBL_ckpt = str(data.get('FBL_ckpt'))

    base_path = Path('data') / singer
    norm = base_path / 'norm'
    wavs = base_path / 'wavs'
    lab = base_path / 'lab'
    textgrids = base_path / 'textgrids'
    bad =base_path / 'bad'
    ds = base_path / 'ds'
    dataset = base_path / singer
    for folder in [base_path, norm, wavs, lab, textgrids, bad, ds, dataset]:
        folder.mkdir(parents=True, exist_ok=True)

    #移动音频
    work_audios = Path('work_audios')
    for file in work_audios.glob('*.wav'):
        destination_file = norm / file.name
        shutil.move(file, destination_file)

    #响度匹配
    for f in tqdm(norm.glob('*.wav')):
        loudness_norm_file(f, f)
    print("loudness_norm complete")

    #音频切片
    for f in tqdm(norm.glob('*.wav')):
        slice_audio(f, wavs)
    print("slice complete")

    #生成lab
    funasr_folder(wavs, lab)
    pinyin_folder(lab, lab, "dictionary/phrases_dict.txt")
    print("lab complete")

    #生成textgrid
    sofa_infer(SOFA_ckpt, wavs, lab, textgrids, "force", "Dictionary", "NoneAPDetector", "lab", "textgrid", True, dictionary=dictionary)
    print("SOFA complete")

    #呼吸标注
    export(FBL_ckpt, wavs, textgrids, textgrids)
    print("FBL complete")

    #筛选标注
    confidence = textgrids / "confidence.csv"
    move_bad(wavs, textgrids, bad, confidence)
    print("move_bad complete")

    #生成ds
    textgrid2ds(textgrids, wavs, ds, dictionary)
    print("ds complete")

    #构建数据集


if __name__ == '__main__':
    quick_start()