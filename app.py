import re
import pandas as pd
import numpy as np
import folium
import time
import psutil
import argparse
from folium.plugins import MarkerCluster
from scipy.stats import zscore
# from memory_profiler import profile


start_time = time.time()

def parse_amenities(amenities_string):
    amenities_string = amenities_string.strip('{}').strip('"').strip(':')
    amenities_list = re.split(r',\s*', amenities_string)
    return amenities_list

def load_data(path, columns=None, chunk_size=None):
    return pd.read_csv(path, usecols=columns, dtype=str, chunksize=chunk_size)

# @profile # enable profiler only in case for memory deep analysis.
def main():
    parser = argparse.ArgumentParser(description='Display total memory used.')
    parser.add_argument('-m', '--memory', action='store_true', help='Show total memory used')
    args = parser.parse_args()

    details_path = "./data/Details_Data.csv"
    price_path = "./data/Price_AV_Itapema.csv"
    vivareal_path = "./data/VivaReal_Itapema.csv"

    chunk_size = 1000000  # Batch Size

    details_df = load_data(details_path, columns=["ad_id", "aquisition_date", "amenities", "latitude", "longitude", "number_of_reviews","guest_satisfaction_overall", "star_rating", "number_of_bedrooms", "number_of_bathrooms", "number_of_guests"], chunk_size=chunk_size)
    price_df = load_data(price_path, columns=["airbnb_listing_id", "price", "aquisition_date", "minimum_stay"], chunk_size=chunk_size)
    vivareal_df = load_data(vivareal_path, columns=["listing_id", "sale_price","monthly_condo_fee", "total_area", "suites", "parking_spaces", "bathrooms","bedrooms", "usable_area", "address_neighborhood"], chunk_size=20000)

    m = folium.Map(location=[-27.0909, -48.6151], zoom_start=13)
    marker_cluster = MarkerCluster().add_to(m)

    for details_chunk, price_chunk, vivareal_chunk in zip(details_df, price_df , vivareal_df):
        price_chunk["price"] = pd.to_numeric(price_chunk["price"], errors='coerce').round(2)
        details_chunk['number_of_reviews'] = pd.to_numeric(details_chunk["number_of_reviews"], errors='coerce').round(2)
        price_chunk["total_revenue"] = price_chunk["price"] * details_chunk["number_of_reviews"]
        price_chunk["total_revenue"] = price_chunk["total_revenue"].round(2)
        details_chunk['guest_satisfaction_overall'] = pd.to_numeric(details_chunk['guest_satisfaction_overall'].str.rstrip('%'), errors='coerce').fillna(0).round(2)
        details_chunk['star_rating'] = pd.to_numeric(details_chunk["star_rating"], errors='coerce').round(2)
        price_chunk["minimum_stay"] = pd.to_numeric(price_chunk["minimum_stay"], errors='coerce').round(2)
        details_chunk['number_of_bedrooms'] = pd.to_numeric(details_chunk["number_of_bedrooms"], errors='coerce').round(2)
        details_chunk['number_of_bathrooms'] = pd.to_numeric(details_chunk["number_of_bedrooms"], errors='coerce').round(2)
        details_chunk['number_of_guests'] = pd.to_numeric(details_chunk["number_of_guests"], errors='coerce').round(2)

        price_with_location = price_chunk.merge(details_chunk, left_on="airbnb_listing_id", right_on="ad_id", how="inner")

        # Apply z-score outlier detection to the "price" column
        z_scores = zscore(price_with_location["price"])
        abs_z_scores = np.abs(z_scores)
        outliers = (abs_z_scores > 4)  # Adjust the threshold as needed

        # Remove outliers from the DataFrame
        price_with_location = price_with_location[~outliers]
        
        avg_revenue_per_location = price_with_location.groupby("ad_id").agg({
        "price": "mean",
        "number_of_reviews": "mean",
        "total_revenue": "mean",
        "guest_satisfaction_overall" : "mean",
        "star_rating" : "mean",
        "minimum_stay" : "mean",
        "number_of_bedrooms" : "mean",
        "number_of_bathrooms" : "mean",
        "number_of_guests" :"mean"
        }).reset_index()
        
        # another form to identify outliers, but in some datasets or batches it's doens't work very well. 
        # price_with_location["price"] = np.log1p(price_with_location["price"])

        # Calculate the average of total_revenue after summing
        avg_revenue_per_location["total_revenue"] = (avg_revenue_per_location['price'] * avg_revenue_per_location['number_of_reviews']).round(2)
        avg_revenue_per_location = avg_revenue_per_location.sort_values(by=["total_revenue"], ascending=[False])

        print(avg_revenue_per_location.head(10))
        # export to csv
        avg_revenue_per_location.head(10).to_csv('./output/top10-revenues.csv', index=False)

        #get vivareal information for comparison
        vivareal_chunk['bathrooms'] = pd.to_numeric(vivareal_chunk["bathrooms"], errors='coerce')
        vivareal_chunk['total_area'] = pd.to_numeric(vivareal_chunk["total_area"], errors='coerce')
        vivareal_chunk['bedrooms'] = pd.to_numeric(vivareal_chunk["bedrooms"], errors='coerce')
        vivareal_chunk['suites'] = pd.to_numeric(vivareal_chunk["suites"], errors='coerce')
        vivareal_chunk['usable_area'] = pd.to_numeric(vivareal_chunk["usable_area"], errors='coerce')
        vivareal_chunk['sale_price'] = pd.to_numeric(vivareal_chunk["sale_price"], errors='coerce')

        vivareal_comparison = vivareal_chunk.merge(details_chunk, left_on="listing_id", right_on="ad_id", how="left")
        
        #example just to filter by Meia Praia neighborhood with 3 bedrooms to get the average of total_area and usable_area
        vivareal_comparison = vivareal_comparison[
            (vivareal_comparison["address_neighborhood"] == "Meia Praia") & (vivareal_comparison["bedrooms"] == 2)
        ]
        vivareal_comparison = vivareal_comparison.agg({
            #  "bathrooms" : "mean",
            #  "bedrooms" : "mean",
            #  "suites" : "mean",
             "usable_area" : "mean",
             "total_area" : "mean",
            #  "sale_price" : "mean"
        }).reset_index()
        # vivareal_comparison = vivareal_comparison.sort_values(by=["total_area"], ascending=[False])
        print(vivareal_comparison)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time taken to finish: {elapsed_time} seconds")

        #show in the map the top 10 best places to rent in Itapema
        for idx, row in avg_revenue_per_location.head(10).iterrows():
            latitude = price_with_location.loc[price_with_location["ad_id"] == row["ad_id"], "latitude"].values[0]
            longitude = price_with_location.loc[price_with_location["ad_id"] == row["ad_id"], "longitude"].values[0]
            reviews = price_with_location.loc[price_with_location["ad_id"] == row["ad_id"], "number_of_reviews"].values[0]
            folium.Marker(
                location=[latitude, longitude],
                popup=f"Listing ID: {row['ad_id']}, Avg. Revenue per Day: R${row['price']:.2f} , Reviews: {reviews}",
            ).add_to(marker_cluster)

        m.save(f"./output/map.html")
        
    print('Reports generated in the ./output/ directory')

    # Print the sum of memory usage
    if args.memory:
        process = psutil.Process()
        memory_info = process.memory_info()
        print(f"Total memory used: {memory_info.rss / (1024 ** 2):.2f} MB")

if __name__ == "__main__":
    main()