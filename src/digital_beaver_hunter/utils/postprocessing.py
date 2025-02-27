from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from digital_beaver_hunter.utils.geo import get_global_coords_from_yolo_output

def process_stats_footprints(
    name: str, data_dir: str, base_dir_vectors: str, vector_suffix: str
):
    # setup paths
    ds_name = name
    save_dir = Path(data_dir) / ds_name

    # load inference results
    
    # class counts
    df_class_count = pd.read_csv(save_dir / "detected_image_summary.csv")
    # single features
    print((save_dir / "detected_features.csv").exists())
    df_features = pd.read_csv(save_dir / "detected_features.csv")

    # load vectors
    vector_dir = Path(base_dir_vectors)
    vector_file = vector_dir / ds_name / f"{ds_name}_{vector_suffix}"
    gdf = gpd.read_file(vector_file)

    # extract basename
    if "Basename" in gdf.columns:
        gdf["image_id"] = gdf["Basename"].str.replace(".macs", "")
    elif "Name" in gdf.columns:
        gdf["image_id"] = gdf["Name"].str.replace(".macs", "")
    # join (left)
    joined = gdf.set_index("image_id").join(df_class_count.set_index("image_id"))

    # setup output columns
    cols = list(df_class_count.columns.values)
    cols.append("geometry")

    gdf_out = joined.reset_index(drop=False)[cols].replace(np.nan, 0)

    # save files
    outfile = save_dir / (ds_name + "_vector.gpkg")
    gdf_out.to_file(outfile)

    # calculate centroids and save
    gdf_out_centroid = gdf_out.copy()
    gdf_out_centroid["geometry"] = gdf_out.centroid

    outfile_centroid = save_dir / (ds_name + "_vector_centroid.gpkg")
    print(outfile_centroid)
    gdf_out_centroid.to_file(outfile_centroid)

    # make local boxes
    
    # image ids with content
    image_ids = df_features['image_id'].unique()
    # filter to relevant footprints
    gdf_filtered_projected = gdf[gdf['image_id'].isin(image_ids)].to_crs(32608)
    # convert local to global coords
    global_geoms = [get_global_coords_from_yolo_output(row, gdf_filtered_projected) for i, row in df_features.iterrows()]
    
    # create output feature gdf
    gdf_features = gpd.GeoDataFrame(
    df_features,
    geometry=global_geoms,
    crs="EPSG:32608"
    ).to_crs(4326)

    features_outfile = save_dir / (ds_name + "_feature_locations.gpkg")
    gdf_features.to_file(features_outfile)