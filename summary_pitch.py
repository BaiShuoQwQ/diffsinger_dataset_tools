import pathlib
import click
import librosa
import matplotlib.pyplot as plt
import numpy as np
import parselmouth as pm
import tqdm
from textgrid import TextGrid
from modules.ds_tools import distribution


def summary_pitch(wavs, tg):
    wavs = pathlib.Path(wavs)
    tg_dir = pathlib.Path(tg)
    del tg
    tglist = list(tg_dir.glob('*.TextGrid'))

    pit_map = {}
    f0_min = 40.
    f0_max = 1100.
    voicing_thresh_vowel = 0.45
    for tg_file in tqdm.tqdm(tglist):
        filename = tg_file.stem
        wavfile = wavs / f"{filename}.wav"
        tg = TextGrid()
        tg.read(tg_file)
        timestep = 0.01
        f0 = pm.Sound(str(wavfile)).to_pitch_ac(
            time_step=timestep,
            voicing_threshold=voicing_thresh_vowel,
            pitch_floor=f0_min,
            pitch_ceiling=f0_max,
        ).selected_array['frequency']
        pitch = 12. * np.log2(f0 / 440.) + 69.
        for word in tg[0]:
            if word.mark in ['AP', 'SP']:
                continue
            if word.maxTime - word.minTime < timestep:
                continue
            word_pit = pitch[int(word.minTime / timestep): int(word.maxTime / timestep)]
            word_pit = np.extract(word_pit >= 0, word_pit)
            if word_pit.shape[0] == 0:
                continue
            counts = np.bincount(word_pit.astype(np.int64))
            midi = counts.argmax()
            if midi in pit_map:
                pit_map[midi] += 1
            else:
                pit_map[midi] = 1
    midi_keys = sorted(pit_map.keys())
    midi_keys = list(range(midi_keys[0], midi_keys[-1] + 1))
    distribution.draw_distribution(
        title='Pitch Distribution Summary',
        x_label='Pitch',
        y_label='Number of occurrences',
        items=[librosa.midi_to_note(k) for k in midi_keys],
        values=[pit_map.get(k, 0) for k in midi_keys]
    )
    pitch_summary = wavs / 'pitch_distribution.jpg'
    plt.savefig(fname=pitch_summary,
                bbox_inches='tight',
                pad_inches=0.25)
    print(f'Pitch distribution summary saved to {pitch_summary}')

@click.command(help='Generate word-level pitch summary')
@click.option('--wavs', required=True, help='Path to the segments directory')
@click.option('--tg', required=True, help='Path to the TextGrids directory')
def main(wavs, tg):
    summary_pitch(wavs, tg)
if __name__ == '__main__':
    main()
