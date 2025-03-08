import pathlib
import click
import librosa
import soundfile
import textgrid
import tqdm

@click.command(help='Slice TextGrids and wavs from SP and AP')
@click.option(
    '--wavs', required=True,
    help='Directory containing the segmented wav files'
)
@click.option(
    '--tg', required=True,
    help='Directory containing the segmented TextGrid files'
)
@click.option(
    '--out', required=True,
    help='Path to output directory for sliced files'
)
@click.option(
    '--preserve_sentence_names', is_flag=True,
    help='Whether to use sentence marks as filenames (will be re-numbered by default)'
)
@click.option(
    '--wav_subtype', required=False, default='PCM_16',
    help='Wav subtype (defaults to PCM_16)'
)
@click.option(
    '--overwrite', is_flag=True,
    help='Overwrite existing files'
)
def slice_tg(wavs, tg, out, preserve_sentence_names, wav_subtype, overwrite):
    wav_path_in = pathlib.Path(wavs)
    tg_path_in = wav_path_in if tg is None else pathlib.Path(tg)
    del tg
    sliced_path_out = pathlib.Path(out)
    sliced_path_out.mkdir(parents=True, exist_ok=True)
    for tg_file in tqdm.tqdm(tg_path_in.glob('*.TextGrid')):
        tg = textgrid.TextGrid()
        tg.read(tg_file)
        wav, sr = librosa.load((wav_path_in / tg_file.name).with_suffix('.wav'), sr=None)
        sentences_tier = textgrid.IntervalTier(name="sentences")
        words_tier = tg.getFirst('words')
        phones_tier = tg.getFirst('phones')
        idx = 0
        min_slice_dur = 5.0
        max_slice_dur = 15.0
        max_sp_dur = 2.0
        max_ap_dur = 6.0

        start = 0.
        SP_mark = {'SP', 'pau', 'sil', 'cl'}
        AP_mark = {'AP', 'br', 'EP'}
        for ph in phones_tier:

            #print(ph.mark, ph.minTime, ph.maxTime)
            if (ph.mark in SP_mark and (ph.maxTime - ph.minTime)/2 > max_sp_dur) or (ph.mark in AP_mark and (ph.maxTime - ph.minTime) > max_ap_dur):
                if ph.minTime == 0.:
                    sentences_tier.add(start, ph.maxTime - max_sp_dur, '')
                    start = ph.maxTime - max_sp_dur
                else:
                    sentences_tier.add(start, ph.minTime + max_sp_dur, 'SP_long')
                    sentences_tier.add(ph.minTime + max_sp_dur, ph.maxTime - max_sp_dur, '')
                    start = ph.maxTime - max_sp_dur
            elif ph.mark in SP_mark and ph.maxTime - start >= min_slice_dur:
                sentences_tier.add(start, (ph.maxTime + ph.minTime)/2, 'SP_middle')
                start = (ph.maxTime + ph.minTime)/2

            elif ph.mark in AP_mark and ph.minTime - start >= min_slice_dur:
                sentences_tier.add(start, ph.minTime, 'AP_start')
                start = ph.minTime
            elif ph.mark in AP_mark and ph.maxTime - start >= min_slice_dur:
                sentences_tier.add(start, ph.maxTime, 'AP_end')
                start = ph.maxTime

            if phones_tier.maxTime - start <= max_slice_dur:
                sentences_tier.add(start, phones_tier.maxTime, 'tier_end')
                break
            


        for sentence in sentences_tier:
            if sentence.mark == '':
                continue
            sentence_tg = textgrid.TextGrid()
            sentence_words_tier = textgrid.IntervalTier(name='words')
            sentence_phones_tier = textgrid.IntervalTier(name='phones')
            
            if words_tier:
                for word in words_tier:
                    min_time = max(sentence.minTime, word.minTime)
                    max_time = min(sentence.maxTime, word.maxTime)
                    if min_time >= max_time:
                        continue
                    sentence_words_tier.add(
                        minTime=min_time - sentence.minTime, maxTime=max_time - sentence.minTime, mark=word.mark
                    )

            for phone in phones_tier:
                min_time = max(sentence.minTime, phone.minTime)
                max_time = min(sentence.maxTime, phone.maxTime)
                if min_time >= max_time:
                    continue
                sentence_phones_tier.add(
                    minTime=min_time - sentence.minTime, maxTime=max_time - sentence.minTime, mark=phone.mark
                )

            marks_set = {ph.mark for ph in sentence_phones_tier}
            if marks_set.issubset(SP_mark | AP_mark):
                continue

            if words_tier:
                sentence_tg.append(sentence_words_tier)
            sentence_tg.append(sentence_phones_tier)

            if preserve_sentence_names:
                tg_file_out = sliced_path_out / f'{sentence.mark}.TextGrid'
                wav_file_out = tg_file_out.with_suffix('.wav')
            else:
                tg_file_out = sliced_path_out / f'{tg_file.stem}_{str(idx).zfill(2)}.TextGrid'
                wav_file_out = tg_file_out.with_suffix('.wav')
            if tg_file_out.exists() and not overwrite:
                raise FileExistsError(str(tg_file_out))
            if wav_file_out.exists() and not overwrite:
                raise FileExistsError(str(wav_file_out))

            sentence_tg.write(tg_file_out)
            sentence_wav = wav[int(sentence.minTime * sr): min(wav.shape[0], int(sentence.maxTime * sr) + 1)]
            soundfile.write(
                wav_file_out,
                sentence_wav, samplerate=sr, subtype=wav_subtype
            )
            idx += 1


if __name__ == '__main__':
    slice_tg()
