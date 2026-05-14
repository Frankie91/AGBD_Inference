# -*- coding: utf-8 -*-
import os
import tempfile

import ee
import geemap
import numpy as np
import rioxarray as rxr
import xarray as xr
from pyproj import Transformer
from rasterio.enums import Resampling
from sklearn.preprocessing import MinMaxScaler


def array_to_raster(outpath, datarr, crs, yarr, xarr, band_names=None):
    
    """Write a 2D or 3D NumPy array to a georeferenced raster file using xarray/rioxarray 
    coordinates and CRS metadata."""
    
    datarr = np.asarray(datarr).astype("float32").copy()

    if datarr.ndim == 2:
        da = xr.DataArray(
            datarr,
            coords={"y": yarr, "x": xarr},
            dims=("y", "x")
        )
    elif datarr.ndim == 3:
        da = xr.DataArray(
            datarr,
            coords={"band": band_names, "y": yarr, "x": xarr},
            dims=("band", "y", "x")
        )
    else:
        raise ValueError("datarr must be 2D or 3D")

    da.rio.write_crs(crs, inplace=True)
    da.rio.to_raster(outpath)
    da.close()
    del da

def normalize_data(data, norm_values, norm_strat, nodata_value = None) :
    """
    Normalize the data, according to various strategies:
    - mean_std: subtract the mean and divide by the standard deviation
    - pct: subtract the 1st percentile and divide by the 99th percentile
    - min_max: subtract the minimum and divide by the maximum

    Args:
    - data (np.array): the data to normalize
    - norm_values (dict): the normalization values
    - norm_strat (str): the normalization strategy

    Returns:
    - normalized_data (np.array): the normalized data
    """

    if norm_strat == 'mean_std':
        mean, std = norm_values['mean'], norm_values['std']
        if nodata_value is not None :
            data = np.where(data == nodata_value, 0, (data - mean) / std)
        else : data = (data - mean) / std

    elif norm_strat == 'pct':
        p1, p99 = norm_values['p1'], norm_values['p99']
        if nodata_value is not None :
            data = np.where(data == nodata_value, 0, (data - p1) / (p99 - p1))
        else :
            data = (data - p1) / (p99 - p1)
        data = np.clip(data, 0, 1)

    elif norm_strat == 'min_max' :
        min_val, max_val = norm_values['min'], norm_values['max']
        if nodata_value is not None :
            data = np.where(data == nodata_value, 0, (data - min_val) / (max_val - min_val))
        else:
            data = (data - min_val) / (max_val - min_val)
    
    else: 
        raise ValueError(f'Normalization strategy `{norm_strat}` is not valid.')

    return data

# sentinel 2 download functions

def mask_s2(image, selected_bands, cld_prb_thresh=20):
    
    """Mask Sentinel-2 clouds and invalid scene classes, 
    then keep the requested bands and convert reflectance to surface reflectance units."""
    
    cld_prb = image.select("MSK_CLDPRB")
    scl = image.select("SCL")

    cloud_mask = cld_prb.lt(cld_prb_thresh)
    scl_mask = (
        scl.neq(3)
        .And(scl.neq(8))
        .And(scl.neq(9))
        .And(scl.neq(10))
        .And(scl.neq(11))
    )
    mask = cloud_mask.And(scl_mask)
    return image.updateMask(mask).select(selected_bands).divide(10000)

def yearly_group_median(time_range, selected_bands, region, s2_tile):
    
    """Build a yearly median Sentinel-2 composite for the target region 
    and MGRS tile after cloud filtering and masking."""

    return (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(f"{time_range[0]}", f"{time_range[1]}")
        .filter(ee.Filter.eq("MGRS_TILE", s2_tile))
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", 30))
        .map(lambda img: mask_s2(img, selected_bands, cld_prb_thresh=20))
        .median()
        .clip(region)
    )


def sentinel2_processing(
    band_names,
    export_scale,
    final_tif,
    aoi_crs,
    region,
    norm_values,
    s2_tile,
    s2_time_range,
    match_raster=None,
):
    """Export Sentinel-2 bands from Earth Engine, 
    automatically using either a date range or a list of years, 
    then normalize and optionally align the raster."""

    s2_norm_keys = {
        "B1": "B01",
        "B2": "B02",
        "B3": "B03",
        "B4": "B04",
        "B5": "B05",
        "B6": "B06",
        "B7": "B07",
        "B8": "B08",
        "B8A": "B8A",
        "B9": "B09",
        "B11": "B11",
        "B12": "B12",
    }


    imgs = [yearly_group_median(year, band_names, region, s2_tile) for year in s2_time_range]
    img = ee.ImageCollection(imgs).median().clip(region)

    start_date, end_date = s2_time_range
    img = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.eq("MGRS_TILE", s2_tile))
        .map(lambda image: mask_s2(image, band_names, cld_prb_thresh=20))
        .median()
        .clip(region)
    )


    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp_tif = tmp.name

    try:
        geemap.ee_export_image(
            img,
            filename=tmp_tif,
            scale=export_scale,
            crs=aoi_crs,
            region=region,
            file_per_band=False,
        )

        with rxr.open_rasterio(tmp_tif, masked=True) as ds:
            src_crs = ds.rio.crs
            src_x = ds.x.values
            src_y = ds.y.values
            native_rescaled_bands = []

            for i, band_name in enumerate(band_names, start=1):
                band_da = ds.sel(band=i).astype("float32")
                arr = band_da.values.astype("float32")

                if ds.rio.nodata is not None:
                    arr = np.where(arr == ds.rio.nodata, np.nan, arr)
                
                norm_key = s2_norm_keys[band_name]
                norm_vals= norm_values["S2_bands"][norm_key]
                
                arr_rescaled = normalize_data(arr, norm_vals, 'pct')
                
                native_rescaled_bands.append(arr_rescaled)

            native_stack = np.stack(native_rescaled_bands, axis=0).astype("float32")

            native_da = xr.DataArray(
                native_stack,
                dims=("band", "y", "x"),
                coords={
                    "band": np.arange(1, len(band_names) + 1),
                    "y": src_y,
                    "x": src_x,
                },
                attrs=ds.attrs,
            ).rio.write_crs(src_crs)

            if match_raster is not None:
                ds_proc = native_da.rio.reproject_match(
                    match_raster,
                    resampling=Resampling.bilinear,
                )
            else:
                ds_proc = native_da

            out_arr = np.where(np.isnan(ds_proc.values), 
                               0.0, ds_proc.values).astype("float32")

            array_to_raster(
                final_tif,
                out_arr,
                ds_proc.rio.crs,
                ds_proc.y.values,
                ds_proc.x.values,
                band_names=band_names,
            )

    finally:
        if os.path.exists(tmp_tif):
            os.remove(tmp_tif)

# Alos Dtm Processing

def dsm_processing(img, final_tif, match_raster, aoi_crs, region, norm_values):
    
    """Export the ALOS DSM from Earth Engine, normalize elevation values, 
    match the target raster grid, and save the processed raster."""
    
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp_tif = tmp.name

    try:
        geemap.ee_export_image(
            img,
            filename=tmp_tif,
            scale=30,
            crs=aoi_crs,
            region=region,
            file_per_band=False
        )

        with rxr.open_rasterio(tmp_tif, masked=True) as src:
            dem = src.squeeze("band", drop=True).astype("float32")
            dem_vals = dem.values.astype("float32")
            
            dem_rescaled_native = normalize_data(
                dem_vals, norm_values["DEM"], 'pct').astype("float32")

            
            dem_rescaled_da = xr.DataArray(
                dem_rescaled_native,
                dims=("y", "x"),
                coords={"y": dem.y.values, "x": dem.x.values},
                attrs=dem.attrs,
            ).rio.write_crs(dem.rio.crs)
            
            dem_match = dem_rescaled_da.rio.reproject_match(match_raster, resampling=Resampling.bilinear)
            dem_out = np.where(np.isnan(dem_match.values), 0.0, dem_match.values).astype("float32")
            array_to_raster(final_tif, dem_out, match_raster.rio.crs, match_raster.y.values, match_raster.x.values)
    finally:
        if os.path.exists(tmp_tif):
            os.remove(tmp_tif)

def palsar_processing(region, ref_crs, match_raster, years, norm_values, s2_10m_path):
    
    """Download yearly ALOS PALSAR HH and HV backscatter, convert to gamma nought, 
    aggregate across years, normalize, align to the reference grid, and write output rasters."""
    
    hh_list = []
    hv_list = []

    for year in years:
        end = year + 1
        ic = (
            ee.ImageCollection("JAXA/ALOS/PALSAR/YEARLY/SAR_EPOCH")
            .filterDate(f"{year}-01-01", f"{end}-01-01")
            .filterBounds(region)
            .select(["HH", "HV"])
        )
        img = ic.toBands().clip(region)

        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            tmp_tif = tmp.name

        try:
            geemap.ee_export_image(
                img,
                filename=tmp_tif,
                region=region,
                crs=ref_crs,
                scale=25,
                file_per_band=False
            )

            with rxr.open_rasterio(tmp_tif, masked=True) as alos:
                hh_dn = alos.sel(band=1).astype("float32")
                hv_dn = alos.sel(band=2).astype("float32")

                hh_gamma = xr.where(hh_dn == 0, -9999.0, (10.0 * (np.log10(hh_dn ** 2))) - 83.0).astype("float32")
                hv_gamma = xr.where(hv_dn == 0, -9999.0, (10.0 * (np.log10(hv_dn ** 2))) - 83.0).astype("float32")

                hh_list.append(hh_gamma.values)
                hv_list.append(hv_gamma.values)
                alos_y = alos.y.values
                alos_x = alos.x.values
                alos_crs = alos.rio.crs
                hh_attrs = alos.sel(band=1).attrs
                hv_attrs = alos.sel(band=2).attrs
        finally:
            if os.path.exists(tmp_tif):
                os.remove(tmp_tif)

    hh_stack = np.stack(hh_list, axis=0).astype("float32")
    hv_stack = np.stack(hv_list, axis=0).astype("float32")
    
    hh_stack = np.where(hh_stack == -9999.0, np.nan, hh_stack)
    hv_stack = np.where(hv_stack == -9999.0, np.nan, hv_stack)

    hh_gamma_med = np.nanmedian(hh_stack, axis=0).astype("float32")
    hv_gamma_med = np.nanmedian(hv_stack, axis=0).astype("float32")
    
    hh_gamma_med = np.where(np.isnan(hh_gamma_med), -9999.0, hh_gamma_med).astype("float32")
    hv_gamma_med = np.where(np.isnan(hv_gamma_med), -9999.0, hv_gamma_med).astype("float32")

    hh_da = xr.DataArray(hh_gamma_med, dims=("y", "x"), 
                         coords={"y": alos_y, "x": alos_x}, 
                         attrs=hh_attrs).rio.write_crs(alos_crs)
    
    hv_da = xr.DataArray(hv_gamma_med, dims=("y", "x"), 
                         coords={"y": alos_y, "x": alos_x}, 
                         attrs=hv_attrs).rio.write_crs(alos_crs)

    hh_match = hh_da.rio.reproject_match(match_raster, 
                                         resampling=Resampling.bilinear)
    
    hv_match = hv_da.rio.reproject_match(match_raster, 
                                         resampling=Resampling.bilinear)

    hh_rescaled =  normalize_data(hh_match.values, 
                                  norm_values["ALOS_bands"]["HH"], 
                                  'pct',nodata_value=-9999.0).astype("float32")
    
    hv_rescaled =  normalize_data(hv_match.values, 
                                  norm_values["ALOS_bands"]["HV"], 
                                  'pct',nodata_value=-9999.0).astype("float32")

    array_to_raster("Palsar_HH.tif", hh_rescaled, match_raster.rio.crs, 
                    match_raster.y.values, match_raster.x.values, band_names=["HH"])
    
    array_to_raster("Palsar_HV.tif", hv_rescaled, match_raster.rio.crs, 
                    match_raster.y.values, match_raster.x.values, band_names=["HV"])


def landcover_processing(region, match_raster, aoi_crs):
    
    """Export land-cover class and probability layers, 
    encode the class layer into sine/cosine cyclic features, 
    align outputs to the reference grid, and save them as rasters."""
    
    img = (
        ee.ImageCollection("COPERNICUS/Landcover/100m/Proba-V-C3/Global")
        .filterDate("2019-01-01", "2020-01-01")
        .first()
        .clip(region)
    )

    lc_class = img.select("discrete_classification")
    lc_prob = img.select("discrete_classification-proba")

    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp1:
        class_tmp = tmp1.name
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp2:
        prob_tmp = tmp2.name

    try:
        geemap.ee_export_image(lc_class.unmask(255), filename=class_tmp, 
                               scale=100, region=region, crs=aoi_crs, file_per_band=False)
        
        geemap.ee_export_image(lc_prob.unmask(255), filename=prob_tmp, 
                               scale=100, region=region, crs=aoi_crs, file_per_band=False)

        with rxr.open_rasterio(class_tmp, masked=True) as src:
            lc = src.squeeze("band", drop=True).astype("float32")
            class_vals = lc.values.astype("float32")
            class_vals = np.where(class_vals == 255, np.nan, class_vals)

            lc_cos_native = np.where(np.isnan(class_vals), np.nan, 
                                     (np.cos((2 * np.pi) * (class_vals / 100.0)) + 1.0) / 2.0).astype("float32")
            
            lc_sin_native = np.where(np.isnan(class_vals), np.nan, 
                                     (np.sin((2 * np.pi) * (class_vals / 100.0)) + 1.0) / 2.0).astype("float32")

            lc_cos_da = xr.DataArray(lc_cos_native, dims=("y", "x"), 
                                     coords={"y": lc.y.values, "x": lc.x.values}, 
                                     attrs=lc.attrs).rio.write_crs(lc.rio.crs)
            
            lc_sin_da = xr.DataArray(lc_sin_native, dims=("y", "x"), 
                                     coords={"y": lc.y.values, "x": lc.x.values}, 
                                     attrs=lc.attrs).rio.write_crs(lc.rio.crs)

            lc_cos_match = lc_cos_da.rio.reproject_match(match_raster, 
                                                         resampling=Resampling.nearest)
            lc_sin_match = lc_sin_da.rio.reproject_match(match_raster, 
                                                         resampling=Resampling.nearest)

            lc_cos_out = np.where(np.isnan(lc_cos_match.values), 0.0, 
                                  lc_cos_match.values).astype("float32")
            
            lc_sin_out = np.where(np.isnan(lc_sin_match.values), 0.0, 
                                  lc_sin_match.values).astype("float32")

            array_to_raster("LC_Cos.tif", lc_cos_out, lc_cos_match.rio.crs, 
                            lc_cos_match.y.values, lc_cos_match.x.values)
            
            array_to_raster("LC_Sin.tif", lc_sin_out, lc_sin_match.rio.crs, 
                            lc_sin_match.y.values, lc_sin_match.x.values)

        with rxr.open_rasterio(prob_tmp, masked=True) as src:
            lc_prob_da = src.squeeze("band", drop=True).astype("float32")
            prob_vals = lc_prob_da.values.astype("float32")
            
            prob_vals = np.where(prob_vals == 255, np.nan, prob_vals)
            prob_native = (prob_vals / 100.0).astype("float32")
            
            prob_native_da = xr.DataArray(prob_native, dims=("y", "x"), 
                                          coords={"y": lc_prob_da.y.values, 
                                                  "x": lc_prob_da.x.values}, 
                                          attrs=lc_prob_da.attrs).rio.write_crs(lc_prob_da.rio.crs)
            
            lc_prob_match = prob_native_da.rio.reproject_match(match_raster, 
                                                               resampling=Resampling.bilinear)
            
            prob_out = np.where(np.isnan(lc_prob_match.values), 
                                0.0, lc_prob_match.values).astype("float32")
            array_to_raster("LC_Prob.tif", prob_out, 
                            lc_prob_match.rio.crs, lc_prob_match.y.values, 
                            lc_prob_match.x.values)
    
    finally:
        if os.path.exists(class_tmp):
            os.remove(class_tmp)
        if os.path.exists(prob_tmp):
            os.remove(prob_tmp)           