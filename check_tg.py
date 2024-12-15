import pathlib
import click
import tqdm

def check_tg(wavs, tg):
    wavs = pathlib.Path(wavs)
    tg = pathlib.Path(tg)
    missing = []
    filelist = list(wavs.glob('*.wav'))
    for wavfile in tqdm.tqdm(filelist):
        tgfile = tg / wavfile.with_suffix('.TextGrid').name
        if not tgfile.exists():
            missing.append(tgfile)
    if len(missing) > 0:
        print('These TextGrids are missing!')
        for fn in missing:
            print(f' - {fn}')
    else:
        print('All alignments have been successfully generated.')

@click.command('Check if all TextGrids are generated')
@click.option('--wavs', required=True, help='Path to the segments directory')
@click.option('--tg', required=True, help='Path to the TextGrids directory')
def main(wavs, tg):
    check_tg(wavs, tg)

if __name__ == '__main__':
    main()