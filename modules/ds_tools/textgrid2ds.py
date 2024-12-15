import json
from textgrid import TextGrid
from pathlib import Path
from modules.ds_tools.add_ph_num import add_ph_num
from modules.ds_tools.estimate_midi import EstimateMidi
from modules.ds_tools.fix_words import convert_ph_to_words

def _gather_ds(tg_path, wav_path, dictionary, fix_words=False):
    tg_path = Path(tg_path)
    wav_path = Path(wav_path)
    tg = TextGrid.fromFile(tg_path)
    phones_tier = None
    words_tier = None
    for tier in tg:
        if tier.name == 'phones':
            phones_tier = tier
        elif tier.name == 'words':
            words_tier = tier

    ph_seq = [ph.mark for ph in phones_tier]
    ph_dur = [ph.maxTime - ph.minTime for ph in phones_tier]
    ph_num = add_ph_num(ph_seq, str(dictionary))
    
    if fix_words:
        word = convert_ph_to_words(ph_seq, ph_num, dictionary)
    else:
        word = [word.mark for word in words_tier]

    if (word[0]!= 'SP' and word[0]!= 'AP' and ph_num[0] == '1'):
         word.insert(0, 'SP')

    em = EstimateMidi()
    note_seq, note_dur, f0_seq, f0_timestep = em.estimate_note(wav_path, ph_dur, [int(x) for x in ph_num])

    ph_num_list = [int(x) for x in ph_num]
    index_in_ph_seq = 0
    for i, size in enumerate(ph_num_list):
        group = ph_seq[index_in_ph_seq:index_in_ph_seq + size]
        if any(word in {"AP", "SP"} for word in group):
            note_seq[i]='rest'
        index_in_ph_seq += size
    
    note_slur = ["0" for _ in note_dur]


    ds_content = [
                    {
                    "offset": 0.0,
                    "text": " ".join(word),
                    "ph_seq": " ".join(ph_seq),
                    "ph_dur": " ".join(str(round(d, 6)) for d in ph_dur),
                    "ph_num": " ".join(ph_num),
                    "note_seq": " ".join(note_seq),
                    "note_dur": " ".join(note_dur),
                    "note_slur": " ".join(note_slur),
                    "f0_seq": " ".join(str(x) for x in f0_seq),
                    "f0_timestep": str(f0_timestep),
                    }
                ]
    
    return(ds_content)

def textgrid2ds(tg_folder, wav_folder, ds_folder, dictionary, fix_words=False):
    tg_folder = Path(tg_folder)
    wav_folder = Path(wav_folder)
    ds_folder = Path(ds_folder)
    ds_folder.mkdir(parents=True, exist_ok=True)
    
    for tg_file in tg_folder.glob('*.TextGrid'):
        tg_file_name = tg_file.stem
        wav_file_path = wav_folder / f"{tg_file_name}.wav"
        
        if wav_file_path.exists():
            ds_content = _gather_ds(tg_file, wav_file_path, dictionary, fix_words)
            ds_file = ds_folder / f"{tg_file_name}.ds"
            with open(ds_file, "w", encoding="utf-8") as f:
                json.dump(ds_content, f, ensure_ascii=False, indent=4)
        else:
            print(f"No WAV file found for {tg_file_name}.wav")
