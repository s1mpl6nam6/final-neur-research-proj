import pandas as pd

df_meta = pd.read_csv("audio_data.csv")
df_feat = pd.read_csv("audio_features.csv")

# extract numeric ID from file column
# e.g. "100421.mp3" -> 100421
df_feat["id"] = df_feat["file"].str.replace(".mp3", "", regex=False).astype(int)

# merge on the ID column
df = pd.merge(df_feat, df_meta, on="id", how="inner")

# save final dataset
df.to_csv("final_audio_dataset.csv", index=False)

print("Merged shape:", df.shape)
df.head()
