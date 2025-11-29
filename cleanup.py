import pandas as pd

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("final_audio_dataset.csv")

# Convert "genres" column into list
df["genre_list"] = df["genres"].apply(lambda x: str(x).split("|"))

# -----------------------------
# Allowed genres list
# -----------------------------
allowed_genres = {
    "electronic", "pop", "rock", "hiphop", "house", "jazz", "rnb", "rap",
    "classical", "country", "folk", "ambient", "dance",
    "techno", "metal", "soul", "funk", "indie", "blues", "trap"
}

# -----------------------------
# Filter 1: Keep only allowed genres in each song
# -----------------------------
df["clean_genres"] = df["genre_list"].apply(
    lambda lst: [g for g in lst if g.lower() in allowed_genres]
)

# -----------------------------
# Filter 2: Remove songs with ZERO allowed genres
# -----------------------------
df_clean = df[df["clean_genres"].map(len) > 0].copy()

# -----------------------------
# Convert clean_genres back to "|" joined string
# -----------------------------
df_clean["genres"] = df_clean["clean_genres"].apply(lambda lst: "|".join(lst))

# Drop helper columns
df_clean = df_clean.drop(columns=["genre_list", "clean_genres"])

# -----------------------------
# Save cleaned dataset
# -----------------------------
df_clean.to_csv("cleaned_audio_dataset.csv", index=False)

print(f"Original dataset size: {len(df)}")
print(f"Cleaned dataset size: {len(df_clean)}")
print("Saved cleaned dataset as cleaned_audio_dataset.csv")
