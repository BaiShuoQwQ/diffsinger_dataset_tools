import csv
import json
from decimal import Decimal
from tqdm import tqdm

def ds2csv(ds_folder, transcription_file, overwrite):
    """Convert DS files to a transcription file"""
    if not overwrite and transcription_file.exists():
        raise FileExistsError(f"{transcription_file} already exist.")

    transcriptions = []
    any_with_glide = False
    # records that have corresponding wav files, assuming it's midi annotation
    for fp in tqdm(ds_folder.glob("*.ds"), ncols=80):
        if fp.with_suffix(".wav").exists():
            with open(fp, "r", encoding="utf-8") as f:
                ds = json.load(f)
                transcriptions.append(
                    {
                        "name": fp.stem,
                        "ph_seq": ds[0]["ph_seq"],
                        "ph_dur": " ".join(str(round(Decimal(d), 6)) for d in ds[0]["ph_dur"].split()),
                        "ph_num": ds[0]["ph_num"],
                        "note_seq": ds[0]["note_seq"],
                        "note_dur": " ".join(str(round(Decimal(d), 6)) for d in ds[0]["note_dur"].split()),
                        # "note_slur": ds[0]["note_slur"],
                    }
                )
                if "note_glide" in ds[0]:
                    any_with_glide = True
                    transcriptions[-1]["note_glide"] = ds[0]["note_glide"]
    # Lone DS files.
    for fp in tqdm(ds_folder.glob("*.ds"), ncols=80):
        if not fp.with_suffix(".wav").exists():
            with open(fp, "r", encoding="utf-8") as f:
                ds = json.load(f)
                for idx, sub_ds in enumerate(ds):
                    item_name = f"{fp.stem}#{idx}" if len(ds) > 1 else fp.stem
                    transcriptions.append(
                        {
                            "name": item_name,
                            "ph_seq": sub_ds["ph_seq"],
                            "ph_dur": " ".join(str(round(Decimal(d), 6)) for d in sub_ds["ph_dur"].split()),
                            "ph_num": sub_ds["ph_num"],
                            "note_seq": sub_ds["note_seq"],
                            "note_dur": " ".join(str(round(Decimal(d), 6)) for d in sub_ds["note_dur"].split()),
                            # "note_slur": sub_ds["note_slur"],
                        }
                    )
                    if "note_glide" in sub_ds:
                        any_with_glide = True
                        transcriptions[-1]["note_glide"] = sub_ds["note_glide"]
    if any_with_glide:
        for row in transcriptions:
            if "note_glide" not in row:
                row["note_glide"] = " ".join(["none"] * len(row["note_seq"].split()))
    with open(transcription_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "ph_seq",
                "ph_dur",
                "ph_num",
                "note_seq",
                "note_dur",
                # "note_slur",
            ] + (["note_glide"] if any_with_glide else []),
        )
        writer.writeheader()
        writer.writerows(transcriptions)
