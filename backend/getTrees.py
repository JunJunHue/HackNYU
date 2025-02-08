import requests
from collections import Counter

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
        print("No occurrence records found for this area.")
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
        latitude = float(input("Enter latitude: ").strip())
        longitude = float(input("Enter longitude: ").strip())
    except ValueError:
        print("Invalid input. Please enter numeric values for coordinates.")
        return
    
    results = get_common_plants(latitude, longitude)
    if not results:
        return
    
    print("\nMost common plant species in the area:")
    for species, count in results:
        print(f"{species}: {count} occurrence{'s' if count > 1 else ''}")

if __name__ == '__main__':
    main()