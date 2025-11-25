import os
import librosa
import numpy as np
import pandas as pd

AUDIO_DIR = "./audio_files"
OUTPUT_CSV = "./audio_features.csv"
SAMPLE_SIZE = 5   # process only 5 for dev

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None, mono=True)

        # Basic
        duration = librosa.get_duration(y=y, sr=sr)
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        rms = float(np.mean(librosa.feature.rms(y=y)))
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

        # Spectral features
        spec_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        spec_bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
        spec_rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
        spec_contrast = float(np.mean(librosa.feature.spectral_contrast(y=y, sr=sr)))

        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_means = mfcc.mean(axis=1)

        # Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_means = chroma.mean(axis=1)

        feats = {
            "file": os.path.basename(file_path),
            "duration": duration,
            "zcr": zcr,
            "rms": rms,
            "tempo": tempo,
            "spectral_centroid": spec_centroid,
            "spectral_bandwidth": spec_bandwidth,
            "spectral_rolloff": spec_rolloff,
            "spectral_contrast": spec_contrast,
        }

        # Add MFCC values
        for i, v in enumerate(mfcc_means):
            feats[f"mfcc_{i+1}"] = float(v)

        # Add chroma values
        for i, v in enumerate(chroma_means):
            feats[f"chroma_{i+1}"] = float(v)

        return feats

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def main():
    files = [
        os.path.join(AUDIO_DIR, f)
        for f in os.listdir(AUDIO_DIR)
        if f.endswith(".mp3")
    ]

    if SAMPLE_SIZE:
        files = files[:SAMPLE_SIZE]

    print(f"Processing {len(files)} files sequentially...")

    all_feats = []

    for fpath in files:
        print(f"→ Processing {os.path.basename(fpath)}...")
        feats = extract_features(fpath)
        if feats:
            all_feats.append(feats)

    df = pd.DataFrame(all_feats)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nCompleted. Saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
