# ERA5 hourly data on single levels from 1959 to the present [2016-2022] (precipitation only)

ERA5 hourly data on single levels from 1959 to the present ([link](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview)) was used in this work to gather precipitation data over Europe and its extended regions. The dataset consists of precipitation maps in 1-hour intervals from January 2016 to December 2022 over the selected regions. Each .nc file contains one month of precipitation data from the whole Europe and its extended regions.

The files used for training are stored in the ```./train/``` folder, and the testing files are stored in the ```./test/``` folder.

## Summary of a .nc file
```
Dimensions:    (longitude: 525, latitude: 269, time: 744)
Coordinates:
  * longitude  (longitude) float32 -31.0 -30.75 -30.5 ... 99.5 99.75 100.0
  * latitude   (latitude) float32 82.0 81.75 81.5 81.25 ... 15.5 15.25 15.0
  * time       (time) datetime64[ns] 2022-01-01 ... 2022-01-31T23:00:00
Data variables:
    tp         (time, latitude, longitude) float32 ...
Attributes:
    Conventions:  CF-1.6
    history:      2022-04-19 09:45:39 GMT by grib_to_netcdf-2.25.1: /opt/ecmw...
```

## Read file example

```
import xarray as xr


# Read .nc file
nc = xr.open_dataset('test/whole-europe_2022_01_82_-31_15_100.nc')

print('File summary')
print(nc)

# Get 'tp' variable
precipitation_map = nc.variables['tp'][:]

print(f'precipitation_map shape: {precipitation_map.shape}')

# ...
```
