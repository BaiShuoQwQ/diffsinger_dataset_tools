import re
from funasr import AutoModel
import pathlib
from pypinyin import lazy_pinyin, load_phrases_dict, Style

class Wav2Words():
    def __init__(self, funasr_model=None, pinyin_dict=None, lang=None):
        self.model = funasr_model or AutoModel(model="paraformer-zh", vad_model="fsmn-vad", disable_update=True)
        self.pinyin_dict = pinyin_dict
        self.lang = lang or 'zh'

    def run_funasr(self, wav_file):
            model = self.model
            results = model.generate(input=str(wav_file))
            full_text = ''.join([result['text'] for result in results])
            if self.lang == 'zh':
                #TODO: other languages
                characters = re.sub(r'[a-zA-Z\s]', '', full_text)
            return(characters)


    def text2pinyin(self, text):
        if self.pinyin_dict:
            load_phrases_dict(self.load_phrases_from_txt(self.pinyin_dict))
        pinyin_content = ' '.join(lazy_pinyin(text, style=Style.NORMAL, errors='ignore'))
        return(pinyin_content)


    def load_phrases_from_txt(self, dict_path):
        phrases_dict = {}
        with open(dict_path, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split(':')
                if len(parts) == 2:
                    phrase = parts[0]
                    pinyin_str_list = parts[1].split(',')
                    pinyin_list_of_lists = [[pinyin] for pinyin in pinyin_str_list]
                    phrases_dict[phrase] = pinyin_list_of_lists
        return(phrases_dict)
    
def funasr_folder(wav_folder, out_folder):
    wd = Wav2Words()
    wav_folder = pathlib.Path(wav_folder)
    out_folder = pathlib.Path(out_folder)
    if wav_folder.is_dir():
        wav_files = list(wav_folder.rglob('*.wav'))
    else:
        print(f"wav_folder error")
    for wav_file in wav_files:
        characters = wd.run_funasr(wav_file)
        lab_file_name = wav_file.stem + ".lab"
        with open(out_folder / lab_file_name, 'w', encoding='utf-8') as f:
            f.write(characters)

def pinyin_folder(lab_folder, out_folder, pinyin_dict):
    wd = Wav2Words(pinyin_dict=pinyin_dict)
    lab_folder = pathlib.Path(lab_folder)
    out_folder = pathlib.Path(out_folder)
    if lab_folder.is_dir():
        lab_files = list(lab_folder.rglob('*.lab'))
    for lab_file in lab_files:
        with open(lab_file, 'r', encoding='utf-8') as file:
            characters = file.read()
        pinyin_content = wd.text2pinyin(characters)
        lab_file_name = lab_file.stem + ".lab"
        with open(out_folder / lab_file_name, 'w', encoding='utf-8') as f:
            f.write(pinyin_content)