import textgrid
import pathlib

def load_dictionary(file_path):
    file_path = pathlib.Path(file_path)
    dictionary = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        rules = [ln.strip().split('\t') for ln in file.readlines()]
    for r in rules:
        phonemes = r[1].split()
        dictionary[r[0]] = phonemes
    return dictionary

def convert_ph_to_words(ph_seq, ph_num, dictionary):
    ph_dict = load_dictionary(dictionary)
    word_seq = []
    index = 0
    ph_num = [int(num) if isinstance(num, str) else num for num in ph_num]
    ph_num.insert(0, 1)
    ph_num.pop()
    for num in ph_num:
        word_ph_seq = ph_seq[index:index+num]
        word_ph_seq = ' '.join(str(ph) for ph in word_ph_seq)
        if ph_seq[index] == 'AP' or ph_seq[index] == 'SP':
            word_seq.append(ph_seq[index])
        else:
            for word, ph in ph_dict.items():
                if ph == word_ph_seq:
                    word_seq.append(word)
        index += num
    return word_seq

def fix_words(tg_path):
    #TODO
    tg_path = pathlib.Path(tg_path)
    tg = textgrid.TextGrid.fromFile(tg_path)