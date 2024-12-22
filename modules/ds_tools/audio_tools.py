import librosa
import pathlib

def wav_total_length(wavs):
    #return hours
    total_length = 0.0
    if isinstance(wavs, list):
        for file in wavs:
            wave_seconds = librosa.get_duration(filename=str(file))
            total_length += wave_seconds / 3600.
        return total_length
    
    else:
        wav_path = pathlib.Path(wavs)
        if wav_path.is_file() and wav_path.suffix == '.wav':
            return librosa.get_duration(filename=str(wav_path)) / 3600.
        elif wav_path.is_dir():
            for ch in wav_path.iterdir():
                if ch.is_file() and ch.suffix == '.wav':
                    total_length += wav_total_length(ch)
            return total_length