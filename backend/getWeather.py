import requests
import json

def get_nasa_power_monthly_data(latitude, longitude, start_date, end_date):
    # List of supported parameters for the monthly "point" product in the ag community.
    parameters = (
        "T2M,T2M_MAX,T2M_MIN,T2MDEW,RH2M,PRECTOT,WS2M,"
        "PS,QV2M,ALLSKY_SFC_SW_DWN,ALLSKY_SFC_LW_DWN,ALLSKY_KT,"
        "CLRSKY_SFC_SW_DWN"
    )
    
    # Use the monthly endpoint instead of daily.
    url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point?"
        f"parameters={parameters}"
        "&community=ag"
        f"&longitude={longitude}"
        f"&latitude={latitude}"
        f"&start={start_date}"
        f"&end={end_date}"
        "&format=JSON"
    )
    
    # print("Requesting data from:")
    # print(url)
    
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print("Error fetching data:", response.status_code, response.text)
        return None

if __name__ == "__main__":
    # Define your point location and date range (format: YYYYMM)
    latitude = 36.83
    longitude = -86.78
    start_date = "20250101"  # January 2023 in YYYYMM format
    end_date = "20250101"    # For a single month of data
    with open("output.json", 'r') as file:
        data = json.load(file)
    output_data = list()
    count = 0
    for v in data:
        if count % 2 == 0:
            weath_data = get_nasa_power_monthly_data(v[0], v[1], start_date, end_date)
            parameters_data = weath_data["properties"]["parameter"]
            toAppend = list()
            toAppend.append(v[0])
            toAppend.append(v[1])
            for param, values in parameters_data.items():
                for date, value in sorted(values.items()):
                    toAppend.append([param, value])
            toAppend.append(v[-1])
            output_data.append(toAppend)
        count+= 1
        if count % 10 == 0:
            print(v)
            print(count)
    json_string = json.dumps(output_data, indent=4)
    with open('weatherOutput.json', 'w') as json_file:
        json_file.write(json_string)
    # data = get_nasa_power_monthly_data(latitude, longitude, start_date, end_date)
    
    # if data:
    #     # Optionally, print the full JSON response
    #     print("\nFull JSON response:")
    #     print(data)
        
    #     # Extract and display each parameter's data by month
    #     try:
    #         parameters_data = data["properties"]["parameter"]
    #         for param, values in parameters_data.items():
    #             print(f"\nParameter: {param}")
    #             for date, value in sorted(values.items()):
    #                 print(f"  {date}: {value}")
    #     except KeyError as e:
    #         print("Error processing JSON response:", e)
