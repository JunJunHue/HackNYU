import requests
from collections import Counter
import numpy as np
import json


def get_common_plants(latitude, longitude, delta=0.01, limit=300):
    """
    Query the GBIF API for plant occurrences around a given coordinate.
    
    Parameters:
        latitude (float): Latitude of the target location.
        longitude (float): Longitude of the target location.
        delta (float): The degree buffer to create a bounding box (default 0.01).
        limit (int): The maximum number of occurrence records to retrieve.
        
    Returns:
        List of tuples (species_name, count) sorted by frequency, or None if error.
    """
    # Create a bounding box polygon (WKT format)
    # The polygon is defined as: POLYGON((min_lon min_lat, max_lon min_lat, max_lon max_lat, min_lon max_lat, min_lon min_lat))
    min_lon = longitude - delta
    min_lat = latitude - delta
    max_lon = longitude + delta
    max_lat = latitude + delta
    wkt = f"POLYGON(({min_lon} {min_lat}, {max_lon} {min_lat}, {max_lon} {max_lat}, {min_lon} {max_lat}, {min_lon} {min_lat}))"
    
    url = "https://api.gbif.org/v1/occurrence/search"
    params = {
        "geometry": wkt,
        "taxonKey": 6,  # taxonKey 6 corresponds to Plantae (plants)
        "limit": limit
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
    except requests.RequestException as e:
        print("Error fetching data from GBIF:", e)
        return None

    data = response.json()
    occurrences = data.get("results", [])
    
    if not occurrences:
        return None
    
    species_counter = Counter()
    
    # Iterate over the occurrence records and count the species.
    for occ in occurrences:
        # Some records may have a "species" field, others might only have "scientificName".
        species = occ.get("species") or occ.get("scientificName")
        if species:
            species_counter[species] += 1
            
    # Return the species sorted by occurrence count (most common first)
    return species_counter.most_common()

def main():
    print("Enter coordinates to find the most common plants in the area using GBIF data.")
    try:
        latitude =  40.78#5#091 #float(input("Enter latitude: ").strip())
        longitude = -73.96#8#285 # float(input("Enter longitude: ").strip())
    except ValueError:
        print("Invalid input. Please enter numeric values for coordinates.")
        return
    lat_start = 19
    lat_end = 71.5
    long_start = -179
    long_end = -67
    all_res = list()
    for i in np.arange(19, 71.5, 0.5):
        for j in np.arange(-179, -67, 0.5):
            cur_res = get_common_plants(i, j)
            if cur_res:
                toAppend = [latitude, longitude, cur_res]
                all_res.append(toAppend)
            if j% 5 == 0:
                print(j)
        print(i)
    json_string = json.dumps(all_res, indent=4)
    with open('output.json', 'w') as json_file:
        json_file.write(json_string)


if __name__ == '__main__':
    main()
