import librosa
import pathlib
import shutil
import click

def validate_lengths(dir, delete=False, move=True):
    dir = pathlib.Path(dir)
    assert dir.exists() and dir.is_dir(), 'The chosen path does not exist or is not a directory.'

    short_files = []
    long_files = []
    filelist = list(dir.glob('*.wav'))
    total_length = 0.0
    for file in filelist:
        wave_seconds = librosa.get_duration(filename=str(file))
        if wave_seconds < 2.:
            short_files.append(file) 
            print(f'Too short! \'{file}\' has a length of {round(wave_seconds, 1)} seconds!')
        if wave_seconds > 20.:
            long_files.append(file)
            print(f'Too long! \'{file}\' has a length of {round(wave_seconds, 1)} seconds!')
        total_length += wave_seconds / 3600.

    if not short_files and not long_files:
        print('All segments have proper length.')
    elif delete:     
        for file in short_files:  
            file.unlink()
        for file in long_files: 
            file.unlink()
    elif move:
        new_folder = dir.parent / 'wrong_length'
        new_folder.mkdir(parents=True, exist_ok=True)
        for file in short_files + long_files:
            new_file = new_folder / file.name
            shutil.move(file, new_file)

@click.command()
@click.option("--dir", required=True, help='Path to the segments directory')
@click.option("--delete", is_flag=True, help='Delete files after validation')
@click.option("--move", is_flag=True, help='Move files after validation')
def main(dir, delete, move):
    validate_lengths(dir, delete, move)

if __name__ == '__main__':
    main()