import pandas as pd
import google_streetview.api
import google_streetview.helpers

# get your API key: https://developers.google.com/maps/documentation/streetview/get-api-key
my_key = '####' # put your API here
my_size = '350x350'

df = pd.read_csv("data/boston_test_list.csv")
#df = df.sample(5)

df['location'] = df.latitude.astype(str).str.cat(df.longitude.astype(str), sep=',')

loc_list = ""
for item in df['location']:
    loc_list += item
    loc_list += ';'
loc_list = loc_list[:-1]

apiargs = {
    'location': loc_list,
    'size': my_size,
    'key': my_key
}

api_list = google_streetview.helpers.api_list(apiargs)
results = google_streetview.api.results(api_list)

results.preview()
results.save_links('boston_test/links/links.txt')
results.download_links('boston_test/image')
results.save_metadata('boston_test/metadata/metadata.json')



