import pathlib

def add_ph_num(
        ph_seq: list,
        dictionary: str = None,
        vowels: str = None,
        consonants: str = None
):
    assert dictionary is not None or (vowels is not None and consonants is not None), \
        'Either dictionary file or vowels and consonants file should be specified.'
    if dictionary is not None:
        dictionary = pathlib.Path(dictionary).resolve()
        vowels = {'SP', 'AP'}
        consonants = set()
        all_phonemes = {'SP', 'AP'}
        with open(dictionary, 'r', encoding='utf8') as f:
            rules = f.readlines()
        for r in rules:
            syllable, phonemes = r.split('\t')
            phonemes = phonemes.split()
            all_phonemes.update(phonemes)
            if len(phonemes) == 2:
                consonants.add(phonemes[0])
        vowels = vowels | (all_phonemes - consonants)


    else:
        vowels_path = pathlib.Path(vowels).resolve()
        consonants_path = pathlib.Path(consonants).resolve()
        vowels = {'SP', 'AP'}
        consonants = set()
        with open(vowels_path, 'r', encoding='utf8') as f:
            vowels.update(f.read().split())
        with open(consonants_path, 'r', encoding='utf8') as f:
            consonants.update(f.read().split())
        overlapped = vowels.intersection(consonants)
        assert len(vowels.intersection(consonants)) == 0, \
            'Vowel set and consonant set overlapped. The following phonemes ' \
            'appear both as vowels and as consonants:\n' \
            f'{sorted(overlapped)}'

    for ph in ph_seq:
        assert ph in vowels or ph in consonants, \
            f'Invalid phoneme symbol \'{ph}\'.'
    ph_num = []
    i = 0
    while i < len(ph_seq):
        j = i + 1
        while j < len(ph_seq) and ph_seq[j] in consonants:
            j += 1
        ph_num.append(str(j - i))
        i = j

    return ph_num
