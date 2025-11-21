import requests
from dotenv import load_dotenv
import os
import json
from concurrent.futures import ThreadPoolExecutor
import csv


'''
http://developer.jamendo.com/v3.0/autocomplete
^ this is useful to get artist tags somehow


'''


load_dotenv()
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
TRACKS_URL = "https://api.jamendo.com/v3.0/tracks"
PLAYLISTS_URL = "https://api.jamendo.com/v3.0/playlists"
DOWNLOAD_TO = "./audio_files"
DATA_FILE = "./audio_data.csv"

def main():
    params = {
        "client_id": CLIENT_ID,
        "format": "json",
        "limit": 10,
        "audioformat": "mp31",   
        "include": ["licenses", "musicinfo"],
    }
    genres = ["jazz", "hiphop", "electronic", "classical", "rock", "pop", "rnb", "country", "house"]
    
    download_list = {}
    metadata = []

    for genre in genres:
        params["tags"] = genre
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
                'genres': music_info['tags']['genres'],
                'song_name' : song['name'],
                'artist_name' : song["artist_name"],
                'url' : song['shareurl']
            })
        print(f"Downloading {genre}")
        threaded_downloads(download_list)
        download_list = {}
    print("Writing to CSV")
    write_to_csv(DATA_FILE, metadata)


    print(result)

def threaded_downloads(download_list):
    pass


def write_to_csv(file_name, data):
    with open(file_name, 'w', newline="") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
        

def get_songs(params):
    response = requests.get(TRACKS_URL, params=params).json()
    if response["headers"]["code"] != 0:
        print(f"Got error code from API: {response['headers']['code']}")
    result = response["results"]
    return result

def download_file(file_id, download_loc, download_link):
    filename = os.path.join(download_loc, f"{file_id}.mp3")
    audio = requests.get(download_link)
    with open(filename, "wb") as f:
        f.write(audio.content)






if __name__ == "__main__":
    main()