import requests
from dotenv import load_dotenv
import os
from concurrent.futures import ThreadPoolExecutor
import csv

'''
Program downloads metadata and song files from each given genre: 
    "jazz", "hiphop", "electronic", "classical", "rock", "pop", "rnb", "country", "house"

Metadata downloaded is:
'id', 'genres, 'song_name', 'artist_name', 'url' 

All songs are from https://www.jamendo.com/
'''


load_dotenv()
CLIENT_ID = os.getenv("CLIENT_ID")
TRACKS_URL = "https://api.jamendo.com/v3.0/tracks"
DOWNLOAD_TO = "./audio_files"
DATA_FILE = "./audio_data.csv"

BATCH_FETCH_SIZE = 200

os.makedirs(DOWNLOAD_TO, exist_ok=True)


def main():
    params = {
        "client_id": CLIENT_ID,
        "format": "json",
        "limit": BATCH_FETCH_SIZE,
        "audioformat": "mp31",   
        "include": "licenses+musicinfo",
    }
    genres = ["jazz", "hiphop", "electronic", "classical", "rock", "pop", "rnb", "country", "house"]
    
    download_list = {}
    metadata = []
    seen_ids = set()

    for genre in genres:
        # Goes through 4 pages of results
        for i in range(4):
            params["tags"] = genre
            params["offset"] = i * BATCH_FETCH_SIZE
            result = get_songs(params)
            for song in result:
                download = song["audiodownload"] if song["audiodownload"]!= '' else song["audio"]
                if download == '':
                    print(f"Skipping song id: {song['id']} No download link")
                    continue
                download_list[song['id']] = download
                music_info = song['musicinfo']
                metadata.append({
                    'id': song['id'],
                    'genres': "|".join(music_info['tags']['genres']),
                    'song_name' : song['name'],
                    'artist_name' : song["artist_name"],
                    'url' : song['shareurl']
                })
        print(f"Downloading {genre}")
        threaded_downloads(download_list, DOWNLOAD_TO)
        download_list = {}

    print("Writing to CSV")
    write_to_csv(DATA_FILE, metadata)



def write_to_csv(file_name, data):
    if not data:
        print("No data to write to CSV")
        return

    with open(file_name, 'w', newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
        

def get_songs(params):
    response = requests.get(TRACKS_URL, params=params)
    response.raise_for_status()
    response = response.json()
    if response["headers"]["code"] != 0:
        print(f"Got error code from API: {response['headers']['code']}")
        return []
    result = response["results"]
    return result

def download_file(file_id, download_loc, download_link):
    filename = os.path.join(download_loc, f"{file_id}.mp3")
    audio = requests.get(download_link, timeout=10)
    audio.raise_for_status()

    with open(filename, "wb") as f:
        f.write(audio.content)

def threaded_downloads(download_list, download_loc):
    with ThreadPoolExecutor(max_workers = 30) as executor:
        for id_, link in download_list.items():
            executor.submit(download_file, id_, download_loc, link)



if __name__ == "__main__":
    main()