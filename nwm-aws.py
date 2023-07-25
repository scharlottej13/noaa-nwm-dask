import coiled
import flox.xarray
import fsspec
import numpy as np
import rioxarray
import xarray as xr

cluster = coiled.Cluster(
    name="nwm-2001-2020",
    region="us-east-1",
    n_workers=10,
    tags={"project": "nwm"},
    scheduler_vm_types="r7g.xlarge",
    worker_vm_types="c7g.xlarge",
    compute_purchase_option="spot_with_fallback"
)

client = cluster.get_client()

cluster.adapt(minimum=10, maximum=200)

ds = xr.open_zarr(
    fsspec.get_mapper("s3://noaa-nwm-retrospective-2-1-zarr-pds/rtout.zarr", anon=True),
    consolidated=True,
)

subset = ds.zwattablrt.sel(time=slice("2001-01-01", "2020-12-31"))

fs = fsspec.filesystem("s3", requester_pays=True)

counties = rioxarray.open_rasterio(
    fs.open("s3://nwm-250m-us-counties/Counties_on_250m_grid.tif"), chunks="auto"
).squeeze()

# remove any small floating point error in coordinate locations
_, counties_aligned = xr.align(ds, counties, join="override")

county_id = np.unique(counties_aligned.data).compute()
county_id = county_id[county_id != 0]
print(f"There are {len(county_id)} counties!")

county_mean = flox.xarray.xarray_reduce(
    subset,
    counties_aligned.rename("county"),
    func="mean",
    expected_groups=(county_id,),
)

county_mean.load()
print("Saving")
county_mean.to_netcdf("mean_zwattablrt_nwm.nc")
cluster.shutdown()


def make_plot():
    # ## Visualize yearly mean
    # Read county shapefile, combo of state FIPS code and county FIPS code as multi-index
    import geopandas as gpd
    import hvplot.pandas

    counties = gpd.read_file(
        "https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_county_20m.zip"
    ).to_crs("EPSG:3395")
    counties["STATEFP"] = counties.STATEFP.astype(int)
    counties["COUNTYFP"] = counties.COUNTYFP.astype(int)
    continental = counties.loc[~counties["STATEFP"].isin([2, 15, 72])]
    continental = continental.set_index(["STATEFP", "COUNTYFP"])

    # Interpret `county` as combo of state FIPS code and county FIPS code. Set multi-index
    county_mean = xr.open_dataset("mean_zwattablrt_nwm.nc")
    yearly_mean = county_mean.mean("time")
    yearly_mean.coords["STATEFP"] = (yearly_mean.county // 1000).astype(int)
    yearly_mean.coords["COUNTYFP"] = np.mod(yearly_mean.county, 1000).astype(int)
    yearly_mean = yearly_mean.drop_vars("county").set_index(county=["STATEFP", "COUNTYFP"])
    yearly_mean

    # Join
    continental["zwattablrt"] = yearly_mean.to_dataframe()["zwattablrt"]

    continental.hvplot(
        c="zwattablrt",
        cmap='turbo_r',
        title="Average Water Table Depth in 2001 by US County (meters)",
        xaxis=None,
        yaxis=None
    )