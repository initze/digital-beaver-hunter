import geopandas as gpd
from pathlib import Path
import pandas as pd
import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, required=True)
parser.add_argument('--data_dir', type=str, required=True, default='output')
parser.add_argument('--base_dir_vectors', type=str, required=True, default='/isipd/projects/Response/Restricted_Airborne/MACS/2021_Perma-X_Alaska/01_raw_data')
parser.add_argument('--vector_suffix', type=str, required=True, default='footprints_full.shp')
#parser.add_argument('--footprints_file', type=str, required=False, default=None)
args = parser.parse_args()

def main():
    # setup paths
    ds_name = args.name
    save_dir = Path(args.data_dir) / ds_name

    # load inference results
    df_class_count = pd.read_csv(save_dir / 'detected_image_summary.csv')
    
    # load vectors
    vector_dir = Path(args.base_dir_vectors)
    vector_file = vector_dir / ds_name / f'{ds_name}_{args.vector_suffix}'
    gdf = gpd.read_file(vector_file)

    # extract basename
    gdf['image_id'] = gdf['Basename'].str.replace('.macs', '')
    # join (left)
    joined = gdf.set_index('image_id').join(df_class_count.set_index('image_id'))

    # setup output columns
    cols = list(df_class_count.columns.values)
    cols.append('geometry')
    
    gdf_out = joined.reset_index(drop=False)[cols].replace(np.nan, 0)

    # save files
    outfile = save_dir / (ds_name+ '_vector.gpkg')
    gdf_out.to_file(outfile)

    # calculate centroids and save
    gdf_out_centroid = gdf_out.copy()
    gdf_out_centroid['geometry'] = gdf_out.centroid

    outfile_centroid = save_dir / (ds_name+ '_vector_centroid.gpkg')
    print(outfile_centroid)
    gdf_out_centroid.to_file(outfile_centroid)

if __name__=='__main__':
    main()