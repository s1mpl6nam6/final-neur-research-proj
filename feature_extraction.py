import os
import librosa
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

AUDIO_DIR = "./audio_files"
OUTPUT_CSV = "./audio_features.csv"
SAMPLE_SIZE = 5   # development mode


def spectral_entropy(S):
    ps = S**2
    ps_norm = ps / np.sum(ps)
    entropy = -np.sum(ps_norm * np.log2(ps_norm + 1e-10), axis=0)
    return np.mean(entropy)


def reduce_stats(name, arr, feats):
    feats[f"{name}_mean"] = float(np.mean(arr))
    feats[f"{name}_std"] = float(np.std(arr))
    feats[f"{name}_min"] = float(np.min(arr))
    feats[f"{name}_max"] = float(np.max(arr))


def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None, mono=True)

        feats = {
            "file": os.path.basename(file_path),
            "duration": librosa.get_duration(y=y, sr=sr),
        }

        # BASIC FEATURES
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        reduce_stats("zcr", zcr, feats)

        rms = librosa.feature.rms(y=y)[0]
        reduce_stats("rms", rms, feats)

        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        feats["tempo"] = float(tempo)
        feats["beat_count"] = len(beats)

        # SPECTRAL FEATURES
        S = np.abs(librosa.stft(y))  # DEFAULT = 2048 FFT, 512 hop

        centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
        reduce_stats("spectral_centroid", centroid, feats)

        bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]
        reduce_stats("spectral_bandwidth", bandwidth, feats)

        rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr)[0]
        reduce_stats("spectral_rolloff", rolloff, feats)

        contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
        feats["spectral_contrast_mean"] = float(np.mean(contrast))

        flatness = librosa.feature.spectral_flatness(S=S)[0]
        reduce_stats("spectral_flatness", flatness, feats)

        feats["spectral_entropy"] = float(spectral_entropy(S))
        feats["total_energy"] = float(np.sum(S))

        # MFCC + DELTAS
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        for i in range(mfcc.shape[0]):
            feats[f"mfcc_{i+1}_mean"] = float(np.mean(mfcc[i]))
            feats[f"mfcc_{i+1}_std"] = float(np.std(mfcc[i]))

        delta = librosa.feature.delta(mfcc)
        for i in range(delta.shape[0]):
            feats[f"delta_{i+1}_mean"] = float(np.mean(delta[i]))

        delta2 = librosa.feature.delta(mfcc, order=2)
        for i in range(delta2.shape[0]):
            feats[f"delta2_{i+1}_mean"] = float(np.mean(delta2[i]))

        # CHROMA FEATURES (librosa defaults, including CQT)
        chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
        feats["chroma_cqt_mean"] = float(np.mean(chroma_cqt))

        chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
        feats["chroma_cens_mean"] = float(np.mean(chroma_cens))

        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        feats["chroma_stft_mean"] = float(np.mean(chroma_stft))

        # TONAL FEATURES (Tonnetz)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        feats["tonnetz_mean"] = float(np.mean(tonnetz))

        # TEMPORAL FEATURES
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        feats["onset_strength_mean"] = float(np.mean(onset_env))
        feats["onset_rate"] = float(
            len(librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr))
        )

        return feats

    except Exception as e:
        return {"file": os.path.basename(file_path), "error": str(e)}


def process_file(fpath):
    return extract_features(fpath)


def main():
    files = [
        os.path.join(AUDIO_DIR, f)
        for f in os.listdir(AUDIO_DIR)
        if f.lower().endswith(".mp3")
    ]

    if SAMPLE_SIZE:
        print(f"Using sample size = {SAMPLE_SIZE} for development")
        files = files[:SAMPLE_SIZE]

    num_workers = min(cpu_count(), 8)
    print(f"Processing {len(files)} files with {num_workers} workers...")

    # multiprocess
    with Pool(num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(process_file, files), total=len(files)))

    # save dataframe including error rows
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nCompleted! Saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
