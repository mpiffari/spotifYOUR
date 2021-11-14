import os
from math import pi
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import requests

import json

from sklearn.preprocessing import MinMaxScaler
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
__data_location__ = os.path.join(__location__, 'data')
__image_location__ = os.path.join(__location__, 'images')
__base_url__ = 'https://api.spotify.com/v1/' # base URL of all Spotify API endpoints
__auth_url__ = 'https://accounts.spotify.com/api/token'

def datasAnalyzer():
    client_id, client_secret = getClientIdAndSecret()

    # generate access token

    # POST
    auth_response = requests.post(__auth_url__, {
        'grant_type': 'client_credentials',
        'client_id': client_id,
        'client_secret': client_secret,
    })

    # convert the response to JSON
    auth_response_data = auth_response.json()

    # save the access token
    access_token = auth_response_data['access_token']

    # used for authenticating all API calls
    headers = {'Authorization': 'Bearer {token}'.format(token=access_token)}

    # read your 1+ StreamingHistory files (depending on how extensive your streaming history is) into pandas dataframes
    df_stream0 = pd.read_json(os.path.join(__data_location__, 'StreamingHistory0.json'))
    df_stream1 = pd.read_json(os.path.join(__data_location__, 'StreamingHistory1.json'))

    # merge streaming dataframes
    df_stream = pd.concat([df_stream0, df_stream1])

    # create a 'UniqueID' for each song by combining the fields 'artistName' and 'trackName'
    df_stream['UniqueID'] = df_stream['artistName'] + ":" + df_stream['trackName']

    df_stream.head()

    # read your edited Library json file into a pandas dataframe
    df_library = pd.read_json(os.path.join(__data_location__, 'YourLibrary1.json'))

    # add UniqueID column (same as above)
    df_library['UniqueID'] = df_library['artist'] + ":" + df_library['track']

    # add column with track URI stripped of 'spotify:track:'
    new = df_library["uri"].str.split(":", expand=True)
    df_library['track_uri'] = new[2]

    df_library.head()

    # create final dict as a copy df_stream
    df_tableau = df_stream.copy()

    # add column checking if streamed song is in library
    # not used in this project but could be helpful for cool visualizations
    df_tableau['In Library'] = np.where(df_tableau['UniqueID'].isin(df_library['UniqueID'].tolist()), 1, 0)

    # left join with df_library on UniqueID to bring in album and track_uri
    df_tableau = pd.merge(df_tableau, df_library[['album', 'UniqueID', 'track_uri']], how='left', on=['UniqueID'])

    df_tableau.head()

    dict_genre = {}

    # convert track_uri column to an iterable list
    track_uris = df_library['track_uri'].to_list()

    # loop through track URIs and pull artist URI using the API,
    # then use artist URI to pull genres associated with that artist
    # store all these in a dictionary
    for t_uri in track_uris:
        dict_genre[t_uri] = {'artist_uri': "", "genres": []}

        r = requests.get(__base_url__ + 'tracks/' + t_uri, headers=headers)
        r = r.json()
        a_uri = r['artists'][0]['uri'].split(':')[2]
        dict_genre[t_uri]['artist_uri'] = a_uri

        s = requests.get(__base_url__ + 'artists/' + a_uri, headers=headers)
        s = s.json()
        dict_genre[t_uri]['genres'] = s['genres']

    # convert dictionary into dataframe with track_uri as the first column
    df_genre = pd.DataFrame.from_dict(dict_genre, orient='index')
    df_genre.insert(0, 'track_uri', df_genre.index)
    df_genre.reset_index(inplace=True, drop=True)

    df_genre.head()

    df_genre_expanded = df_genre.explode('genres')
    df_genre_expanded.head()

    # save df_tableau and df_genre_expanded as csv files that we can load into Tableau
    df_tableau["minPlayed"] = df_tableau["msPlayed"] / 60000

    df_tableau.to_csv('MySpotifyDataTable.csv')
    df_genre_expanded.to_csv('GenresExpandedTable.csv')

    print('done')
    return


def playlistsAnalyzer():
    client_id, client_secret = getClientIdAndSecret()
    client_credentials_manager = SpotifyClientCredentials(client_id, client_secret)
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

    user = '1176897543'

    if len(sys.argv) > 1:
        user = sys.argv[1]

    playlists = sp.user_playlists(user, 8)

    while playlists:
        for i, playlist in enumerate(playlists['items']):
            print(
                "%4d %s %s" %
                (i +
                 1 +
                 playlists['offset'],
                 playlist['uri'],
                 playlist['name']))

            playlist_id = playlist['uri']  # insert your playlist id
            results = sp.playlist(playlist_id)

            # create a list of song ids
            ids = []

            for item in results['tracks']['items']:
                track = item['track']['id']
                ids.append(track)

            song_meta = {'id': [], 'album': [], 'name': [],
                         'artist': [], 'explicit': [], 'popularity': []}

            for song_id in ids:
                # get song's meta data
                meta = sp.track(song_id)

                # song id
                song_meta['id'].append(song_id)

                # album name
                album = meta['album']['name']
                song_meta['album'] += [album]

                # song name
                song = meta['name']
                song_meta['name'] += [song]

                # artists name
                s = ', '
                artist = s.join([singer_name['name'] for singer_name in meta['artists']])
                song_meta['artist'] += [artist]

                # explicit: lyrics could be considered offensive or unsuitable for children
                explicit = meta['explicit']
                song_meta['explicit'].append(explicit)

                # song popularity
                popularity = meta['popularity']
                song_meta['popularity'].append(popularity)

            song_meta_df = pd.DataFrame.from_dict(song_meta)

            # check the song feature
            features = sp.audio_features(song_meta['id'])
            # change dictionary to dataframe
            features_df = pd.DataFrame.from_dict(features)

            # convert milliseconds to mins
            # duration_ms: The duration of the track in milliseconds.
            # 1 minute = 60 seconds = 60 × 1000 milliseconds = 60,000 ms
            features_df['duration_ms'] = features_df['duration_ms'] / 60000

            # combine two dataframe
            final_df = song_meta_df.merge(features_df)

            print(final_df)

            music_feature = features_df[
                ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness',
                 'valence', 'tempo', 'duration_ms']]
            min_max_scaler = MinMaxScaler()
            music_feature.loc[:] = min_max_scaler.fit_transform(music_feature.loc[:])

            # plot size
            fig = plt.figure(figsize=(12, 8))
            fig.suptitle(playlist['name'], fontsize=14, fontweight='bold')
            # convert column names into a list
            categories = list(music_feature.columns)
            # number of categories
            N = len(categories)

            # create a list with the average of all features
            value = list(music_feature.mean())

            # repeat first value to close the circle
            # the plot is a circle, so we need to "complete the loop"
            # and append the start value to the end.
            value += value[:1]
            # calculate angle for each category
            angles = [n / float(N) * 2 * pi for n in range(N)]
            angles += angles[:1]

            # plot
            plt.polar(angles, value)
            plt.fill(angles, value, alpha=0.3)

            # plt.title('Discovery Weekly Songs Audio Features', size=35)

            plt.xticks(angles[:-1], categories, size=15)
            plt.yticks(color='grey', size=15)
            plt.show()

        if playlists['next']:
            playlists = sp.next(playlists)
        else:
            playlists = False


def getClientIdAndSecret():
    stream = open("secrets.txt", "r")
    info = stream.readlines()
    client_id = info[0].split(" ")[1].rstrip('\n')  # insert your client id
    client_secret = info[1].split(" ")[1].rstrip('\n')  # insert your client secret id here
    return client_id, client_secret


if __name__ == "__main__":
    # playlistsAnalyzer()
    datasAnalyzer()
