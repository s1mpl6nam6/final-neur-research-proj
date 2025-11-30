import pandas as pd
from collections import Counter

# Load cleaned dataset
df = pd.read_csv("cleaned_audio_dataset.csv")

# Convert "genres" column into lists
df["genre_list"] = df["genres"].apply(lambda x: str(x).split("|"))

# Flatten all genre labels into a single list
all_genres = [genre for sublist in df["genre_list"] for genre in sublist]

# Count frequency of each genre
genre_counts = Counter(all_genres)

# Print nicely
print("Genre Frequency Distribution:\n")
for genre, count in genre_counts.most_common():
    print(f"{genre}: {count}")