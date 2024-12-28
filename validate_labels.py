import pathlib
import csv
import click
import matplotlib.pyplot as plt
import tqdm
from modules.ds_tools import distribution
from textgrid import TextGrid

# noinspection PyShadowingBuiltins
def validate_labels(dir, wavs, dictionary, type, summary):
    # Load dictionary
    dict_path = pathlib.Path(dictionary)
    with open(dict_path, 'r', encoding='utf8') as f:
        rules = [ln.strip().split('\t') for ln in f.readlines()]
    dictionary = {}
    phoneme_set = set()
    ignore_set = {'SP', 'AP'}
    for r in rules:
        phonemes = r[1].split()
        dictionary[r[0]] = phonemes
        phoneme_set.update(phonemes)

    # Run checks
    check_failed = False
    covered = set()
    phoneme_map = {}
    filelist = []
    for ph in sorted(phoneme_set):
        phoneme_map[ph] = 0


    label_dir = pathlib.Path(dir)
    if wavs:
        wav_dir = pathlib.Path(wavs)

    if type=='lab':
        if wavs:
            filelist = list(wav_dir.glob('*.wav'))
        else:
            filelist = list(label_dir.glob('*.lab'))
        for file in tqdm.tqdm(filelist):
            filename = file.stem
            annotation = label_dir / f"{filename}.lab"
            if not annotation.exists():
                print(f'No lab found for \'{filename}\'!')
                check_failed = True
                continue
            with open(annotation, 'r', encoding='utf8') as f:
                syllables = f.read().strip().split()
            if not syllables:
                print(f' lab file \'{annotation}\' is empty!')
                check_failed = True
            else:
                oov = []
                for s in syllables:
                    if s not in dictionary:
                        oov.append(s)
                    else:
                        for ph in dictionary[s]:
                            phoneme_map[ph] += 1
                        covered.update(dictionary[s])
                if oov:
                    print(f'Syllable(s) {oov} not allowed in lab file \'{annotation}\'')
                    check_failed = True

    elif type=='textgrid':
        phoneme_set.update(ignore_set)
        covered.update(ignore_set)
        if wavs:
            filelist = list(wav_dir.glob('*.wav'))
        else:
            filelist = list(label_dir.glob('*.TextGrid'))
        for file in tqdm.tqdm(filelist):
            filename = file.stem
            annotation = label_dir / f"{filename}.TextGrid"
            if not annotation.exists():
                print(f'No TextGrid found for \'{filename}\'!')
                check_failed = True
                continue
            tg = TextGrid.fromFile(annotation)
            phones_tier = None
            for tier in tg:
                if tier.name == 'phones':
                    phones_tier = tier
            syllables = [ph.mark for ph in phones_tier]
            if not phones_tier:
                print(f'TextGrid file \'{annotation}\' phones tier not found!')
            elif not syllables:
                print(f'TextGrid file \'{annotation}\' is empty!')
                check_failed = True
            else:
                oov = []
                for s in syllables:
                    if s not in phoneme_set:
                        oov.append(s)
                    elif s not in ignore_set:
                        phoneme_map[s] += 1
                        covered.add(s)
                if oov:
                    print(f'Syllable(s) {oov} not allowed in TextGrid file \'{annotation}\'')
                    check_failed = True

    elif type=='csv':
        phoneme_set.update(ignore_set)
        covered.update(ignore_set)
        if label_dir.is_file() and label_dir.suffix.lower() == '.csv':
            csv_file = label_dir
        elif label_dir.is_dir():
            csv_files = list(label_dir.rglob('*.csv'))
            if not any(csv_files):
                print(f'No csv found!')
                csv_file = None
            else:
                csv_file = csv_files[0]
        if csv_file.is_file():
            item_names = []
            with open(csv_file, "r", encoding="utf-8") as f:
                for trans_line in tqdm.tqdm(csv.DictReader(f)):
                    item_name = trans_line["name"]
                    item_names.append(item_name)
                    syllables = trans_line["ph_seq"].strip().split()
                    if not syllables:
                        print(f'csv file \'{csv_file}\' is empty!')
                        check_failed = True
                    else:
                        oov = []
                        for s in syllables:
                            if s not in phoneme_set:
                                oov.append(s)
                            elif s not in ignore_set:
                                phoneme_map[s] += 1
                                covered.add(s)
                        if oov:
                            print(f'Syllable(s) {oov} not allowed in csv \'{csv_file}\'')
                            check_failed = True
            if wavs:
                filelist = list(wav_dir.glob('*.wav'))
                missing_wavs = [f.stem for f in filelist if f.stem not in item_names]
                if missing_wavs:
                    print(f'label not found in CSV: {missing_wavs}')

    # Phoneme coverage
    uncovered = phoneme_set - covered
    if uncovered:
        print(f'The following phonemes are not covered!')
        print(sorted(uncovered))
        print('Please add more recordings to cover these phonemes.')
        check_failed = True

    if not check_failed:
        print('All annotations are well prepared.')

    if summary:
        phoneme_list = sorted((phoneme_set - ignore_set))
        phoneme_counts = [phoneme_map.get(ph, 0) for ph in phoneme_list]

        distribution.draw_distribution(
            title='Phoneme Distribution Summary',
            x_label='Phoneme',
            y_label='Number of occurrences',
            items=phoneme_list,
            values=phoneme_counts
        )
        phoneme_summary = wav_dir / 'phoneme_distribution.jpg'
        plt.savefig(fname=phoneme_summary,
                    bbox_inches='tight',
                    pad_inches=0.25)
        print(f'Phoneme distribution summary saved to {phoneme_summary}')

@click.command(help='Validate transcription labels')
@click.option('--dir', required=True, help='Path to the label directory')
@click.option('--dictionary', default='dictionary/opencpop-extension.txt', help='Path to the dictionary file')
@click.option('--type', default='lab', help='label file type, lab or textgrid or csv')
@click.option('--wavs', help='wav folder, optional')
@click.option('--summary', is_flag=True, help='phoneme_summary')

def main(dir, wavs, dictionary, type, summary):
    validate_labels(dir, wavs, dictionary, type, summary)

if __name__ == '__main__':
    main()
