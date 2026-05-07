# -*- coding: utf-8 -*-
import numpy as np
import ee
import rasterio

import pickle
from tiler import Tiler, Merger

import torch
import paper_models as mods
import rasterio.enums
import rioxarray as rxr

from pyproj import Transformer

import utils as ut

# Gee Authentication
ee.Authenticate()
ee.Initialize(
    project='tanithsoc-489107', # replace with your own
    opt_url='https://earthengine-highvolume.googleapis.com')

# Loading Normalization Values Dictionary
with open('statistics_subset_2019-2020-v4.pkl', 'rb') as file:
    Norm_Values = pickle.load(file)

s2_10m_path = "S2_10m.tif"

# Sicily
# AOIcrs = "EPSG:32633"
# AOI = 12.83139213,37.63523419,12.85789976,37.65396832

# Greece
S2_tile = "34TGL"
AOIcrs = "EPSG:2100"
# list of test regions used 
# AOI = 24.00551587,40.94915108,24.03336070,40.97226228
AOI = 23.98311359,40.90969646,24.00233295,40.92263750

# slightly enlarge the download area to minimize border effects
# in the final AGBD estimate
incr_fact = 0.01

topY = AOI[3] + incr_fact
bottomY = AOI[1] - incr_fact
rightX = AOI[2] + incr_fact
leftX = AOI[0] - incr_fact      

region = ee.Geometry.BBox(leftX, bottomY, rightX, topY)

# Africa - to test with the region in ghana provided by the authors
# AOIcrs = "EPSG:32630"
# AOI = -1.58834601,6.00660534,-1.56756681,6.02313820

# ---------------------------------------------------------------------------#
# Sentinel 2 Processing -----------------------------------------------------#
# ---------------------------------------------------------------------------#

# s2_time_range can be either two dates or a list of years.
# function automatically switches between using a single s2 image
# or computing a multi-year growing seasons (04/01 to 11/01) median accordingly.

# to switch between the two, comment\uncomment accordingly

# s2_time_range = [2018, 2019, 2020]
s2_time_range = ["2020-10-22", "2020-10-23"]

bands10 = ["B2", "B3", "B4", "B8"]
bands20 = ["B5", "B6", "B7", "B8A", "B11", "B12"]
bands60 = ["B1", "B9"]

# 1. Export 10 m stack as reference grid
ut.sentinel2_processing(
    band_names=bands10,
    export_scale=10,
    final_tif="S2_10m.tif",
    aoi_crs=AOIcrs,
    region=region,
    norm_values=Norm_Values,
    s2_tile=S2_tile,
    s2_time_range=s2_time_range,
)

# 2. Open the saved 10 m raster as alignment template
with rxr.open_rasterio(s2_10m_path) as match_10m:
    # 3. Export 20 m and align to 10 m
    ut.sentinel2_processing(
        band_names=bands20,
        export_scale=20,
        final_tif="S2_20m.tif",
        aoi_crs=AOIcrs,
        region=region,
        norm_values=Norm_Values,
        s2_tile=S2_tile,
        s2_time_range=s2_time_range,
        match_raster=match_10m
    )

    # 4. Export 60 m and align to 10 m
    ut.sentinel2_processing(
        band_names=bands60,
        export_scale=60,
        final_tif="S2_60m.tif",
        aoi_crs=AOIcrs,
        region=region,
        norm_values=Norm_Values,
        s2_tile=S2_tile,
        s2_time_range=s2_time_range,        
        match_raster=match_10m
    )

# ---------------------------------------------------------------------------#
# Lat\Lon Sin\Cos Layers Calculation ----------------------------------------#
# ---------------------------------------------------------------------------#

with rxr.open_rasterio(s2_10m_path) as s2_10m:
    ref_lon = s2_10m.x.values
    ref_lat = s2_10m.y.values

    xx, yy = np.meshgrid(ref_lon, ref_lat)

    transformer = Transformer.from_crs(
        s2_10m.rio.crs,
        "EPSG:4326",
        always_xy=True
    )

    lon, lat = transformer.transform(xx, yy)

lat_cos = (np.cos((np.pi * lat) / 90) + 1) / 2
lat_sin = (np.sin((np.pi * lat) / 90) + 1) / 2
lon_cos = (np.cos((np.pi * lon) / 180) + 1) / 2
lon_sin = (np.sin((np.pi * lon) / 180) + 1) / 2

ut.array_to_raster("LatCos.tif", 
                   lat_cos, AOIcrs, ref_lat, ref_lon)

ut.array_to_raster("LatSin.tif", 
                   lat_sin, AOIcrs, ref_lat, ref_lon)

ut.array_to_raster("LonCos.tif", 
                   lon_cos, AOIcrs, ref_lat, ref_lon)

ut.array_to_raster("LonSin.tif", 
                   lon_sin, AOIcrs, ref_lat, ref_lon)

# ---------------------------------------------------------------------------#
# DTM Download --------------------------------------------------------------#
# ---------------------------------------------------------------------------#

dataset = ee.ImageCollection("JAXA/ALOS/AW3D30/V4_1").select("DSM")
filtered_collection = dataset.filterBounds(region)

mosaicked_dsm = filtered_collection.mosaic()

with rxr.open_rasterio(s2_10m_path) as s2_10m:
    ut.dsm_processing(
        img=mosaicked_dsm,
        final_tif="ALOS_DSM.tif",
        match_raster=s2_10m,
        aoi_crs=AOIcrs,
        region=region,
        norm_values=Norm_Values
    )
  
# Alos-Palsar 2 -------------------------------------------------------------#

alos_bands = ["HH", "HV"]

palsar_years = [2020]
# in case multi-year median is a better approach, uncomment below and comment above.
# palsar_years = [2018, 2019, 2020]

with rxr.open_rasterio(s2_10m_path) as s2_10m:
    ut.palsar_processing(
        region,
        AOIcrs,
        s2_10m,
        years=palsar_years,
        norm_values=Norm_Values,
        s2_10m_path=s2_10m_path
    )

# ---------------------------------------------------------------------------#
# Land Cover Processing -----------------------------------------------------#
# ---------------------------------------------------------------------------#

with rxr.open_rasterio(s2_10m_path) as s2_10m:
    ut.landcover_processing(
        region=region,
        match_raster=s2_10m,
        aoi_crs=AOIcrs
    )

# ---------------------------------------------------------------------------#
# Stacked Array Construction ------------------------------------------------#
# ---------------------------------------------------------------------------#

# Read grouped S2 rasters
with rasterio.open("S2_10m.tif") as src:
    s2_10 = src.read()

with rasterio.open("S2_20m.tif") as src:
    s2_20 = src.read()   

with rasterio.open("S2_60m.tif") as src:
    s2_60 = src.read()   

# Map arrays to actual band names
s2_dict = {
    "B2":  s2_10[0],
    "B3":  s2_10[1],
    "B4":  s2_10[2],
    "B8":  s2_10[3],
    "B5":  s2_20[0],
    "B6":  s2_20[1],
    "B7":  s2_20[2],
    "B8A": s2_20[3],
    "B11": s2_20[4],
    "B12": s2_20[5],
    "B1":  s2_60[0],
    "B9":  s2_60[1],
}

# Desired Sentinel-2 order
s2_order = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]

# Build ordered S2 stack
s2_stack = np.stack([s2_dict[b] for b in s2_order], axis=0)

raster_files = [
    "LatCos", "LatSin", "LonCos", "LonSin",
    "Palsar_HH", "Palsar_HV",
    "LC_Cos","LC_Sin","LC_Prob", "ALOS_DSM"
]

stacked_layers = []

# Legge tutti i raster e li impila nella lista
for file in raster_files:
    with rasterio.open(file + '.tif') as src:
        data = src.read()  # Legge tutte le bande del raster
        stacked_layers.append(data)

# Converte la lista in un array NumPy
stacked_vars = np.concatenate(stacked_layers, axis=0)

stacked_array = np.concatenate([s2_stack, stacked_vars])

# ---------------------------------------------------------------------------#
# Stacked Array Tiling ------------------------------------------------------#
# ---------------------------------------------------------------------------#

ovlap = 0.9

tiler = Tiler(data_shape=stacked_array.shape, 
              tile_shape=(22,25,25), overlap=ovlap,
              channel_dimension=0) 

# Calcolo del padding necessario per evitare problemi di dimensione delle tiles
new_shape, up_padding = tiler.calculate_padding()
tiler.recalculate(data_shape=new_shape)

tiles = []
tiles_ids = []
   
padded_image = np.pad(stacked_array, up_padding, mode="reflect")

for tile_id, tile in tiler(padded_image, progress_bar=True):        
    tile = np.expand_dims(tile, axis = 0) 
    tiles.append(tile)
    tiles_ids.append(tile_id)

tiles = np.concatenate(tiles)

# --------------------------------------------------------------------------#
# Inference ----------------------------------------------------------------#
# --------------------------------------------------------------------------#

# installation of triton also required, pip install triton-windows

device = torch.device("cuda") 

# greece 
# yes 0 
# maybe 4 5
# no 1 2 3

weights_list = ['60688111-1_best.ckpt', '18693595-1_best.ckpt',
                '18693595-2_best.ckpt', '18693595-3_best.ckpt',
                '18693595-4_best.ckpt', '18693595-5_best.ckpt']

state_dict = torch.load(weights_list[0], 
                            map_location=torch.device('cuda'))['state_dict']

# check 
state_dict = {k.replace('model.model.','model.'):v for k,v in state_dict.items()}

model = mods.Net('nico').to(device)  
model.load_state_dict(state_dict)

batch_size = 64

# Conversione delle patches in tensore Torch e creazione del dataloader di ingestione dati
tiles_tensor = torch.tensor(tiles, dtype=torch.float32).to(device)
dataset = torch.utils.data.TensorDataset(tiles_tensor)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

torch.set_float32_matmul_precision('high')
model = torch.compile(model, backend="inductor", 
                      mode="default", dynamic=False)

model.eval()

predictions = []

with torch.no_grad():
    for batch in dataloader:
        batch = batch[0].to(device)  
        output = model(batch)
        predictions.append(output)  

predictions = torch.cat(predictions, dim=0).cpu().numpy()

recon_arr = np.expand_dims(stacked_array[0], axis=0)

reconstr_tiler = Tiler(data_shape=recon_arr.shape, 
              tile_shape=(1,25,25), overlap=ovlap,
              channel_dimension=0) 

new_shape, up_padding = reconstr_tiler.calculate_padding()
reconstr_tiler.recalculate(data_shape=new_shape)
padded_image = np.pad(recon_arr, up_padding, mode="reflect")

# up_pad aggiunge un padding riflettente ai bordi per evitare problemi di dimensione

merger = Merger(reconstr_tiler, window="overlap-tile")

for tile_id in range(len(tiles_ids)):
   merger.add(tiles_ids[tile_id], predictions[tile_id])

Pred_Arr = merger.merge(extra_padding=up_padding).squeeze()

# Negative Values are indeed possible!
Pred_Arr[Pred_Arr < 0] = 0

with rxr.open_rasterio(s2_10m_path) as s2_10m:
    ref_lon = s2_10m.x.values
    ref_lat = s2_10m.y.values

ut.array_to_raster( "AGBD.tif", 
                   Pred_Arr, AOIcrs, ref_lat, ref_lon)

# rescaling and rewriting
# add clipping the borders to original AOI

with rxr.open_rasterio('AGBD.tif', masked=True) as dataset:
      AGBD = dataset.squeeze()
      AGBD = AGBD.rio.reproject(
              AGBD.rio.crs,
              resolution=30, # "average" between best(s2 10m bands) and worst (land cover 100m) actual resolution
              resampling=rasterio.enums.Resampling.average)
      
      AGBD = AGBD.rio.clip_box(*AOI, crs="EPSG:4326")
      
      AGBD_crs = AGBD.rio.crs
      AGBD_lat = AGBD.coords['y'].values
      AGBD_long = AGBD.coords['x'].values
      
      del dataset

ut.array_to_raster("AGBD.tif", AGBD.data, AGBD_crs, 
             AGBD_lat, AGBD_long)

# clipping the predictions for comparison

with rxr.open_rasterio("34TGL.tif", masked=True) as src:
    AGBD_REF = src.squeeze()
    AGBD_REF = AGBD_REF.rio.clip_box(*AOI, crs="EPSG:4326")
    AGBD_REF = AGBD_REF.rio.reproject_match(
        AGBD,
        resampling=rasterio.enums.Resampling.average
    )

AGBD_REF.rio.to_raster("AGBD_REF.tif")