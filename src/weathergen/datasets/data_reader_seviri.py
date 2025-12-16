# (C) Copyright 2024 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from pathlib import Path
from typing import override

import numpy as np
import xarray as xr
from numpy.typing import NDArray

# for interactive debugging
import code 
import pdb 

# for reading the parquet files
import pandas as pd

from weathergen.datasets.data_reader_base import (
    DataReaderTimestep,
    ReaderData,
    TimeWindowHandler,
    TIndex,
    check_reader_data,
)

_logger = logging.getLogger(__name__)



class DataReaderSeviri(DataReaderTimestep):
    """Data reader for SEVIRI satellite data."""

    def __init__(
        self,
        tw_handler: TimeWindowHandler,
        filename: Path,
        stream_info: dict,
    ) -> None:

        """Initialize the SEVIRI data reader."""
        np32 = np.float32

        # open the dataset the way we want it
        time_ds = xr.open_zarr(filename, group= "era5")
        ds = xr.open_zarr(filename, group= "seviri")
        
        #code.interact(local=locals())
        #pdb.breakpoint()
        print("Max time: ", time_ds.time.max().values)

        # check if the data overlaps with the time window, otherwise initialises as empty datareader
        if tw_handler.t_start >= time_ds.time.max() or tw_handler.t_end <= time_ds.time.min():
            name = stream_info["name"]
            _logger.warning(f"{name} is not supported over data loader window. Stream is skipped.")
            super().__init__(tw_handler, stream_info)
            self.init_empty()
            return
        
        if "frequency" in stream_info:
            assert False, "Frequency sub-sampling currently not supported"

        # checks length of time in dataset
        idx_start = 0
        idx_end = 120 # len(time_ds.time) - 1
        data_start_time = time_ds.time[idx_start].values
        data_end_time = time_ds.time[idx_end].values

        period = (data_end_time - data_start_time)
        print("Data period: ", period)

        assert data_start_time is not None and data_end_time is not None, (
            data_start_time,
            data_end_time,
        )

        # sets the time window handler and stream info in the base class
        super().__init__(
            tw_handler,
            stream_info,
            data_start_time,
            data_end_time,
            period,
        )

        # If there is no overlap with the time range, no need to keep the dataset.
        if tw_handler.t_start >= data_end_time or tw_handler.t_end <= data_start_time:
            self.init_empty()
            return
        else:
            self.ds = ds
            self.len = idx_end - idx_start #len(ds)

        self.exclude = {"LWMASK", "_indices", "quality_flag"} # exclude these from channels because we don't have a statistics for them
        self.channels_file = [k for k in self.ds.keys()] 

        # caches lats and lons
        # if you want a spatial subset, do it here
        lat_name = stream_info.get("latitude_name", "latitude")
        self.latitudes = _clip_lat(np.array(ds[lat_name], dtype=np32))
        lon_name = stream_info.get("longitude_name", "longitude")
        self.longitudes = _clip_lon(np.array(ds[lon_name], dtype=np32))


        self.geoinfo_channels = stream_info.get("geoinfos", [])
        self.geoinfo_idx = [self.channels_file.index(ch) for ch in self.geoinfo_channels]
        # cache geoinfos
        #self.geoinfo_data = np.stack([np.array(ds[ch], dtype=np32) for ch in self.geoinfo_channels])
        #self.geoinfo_data = self.geoinfo_data.transpose()

        # select/filter requested target channels
        # this will access the stream info, hence make sure to set it.
        self.target_idx, self.target_channels = self.select_channels(ds, "target")
        #self.target_channels = [self.channels_file[i] for i in self.target_idx]

        self.source_idx, self.source_channels = self.select_channels(ds, "source")
        #self.source_channels = [self.channels_file[i] for i in self.source_idx]
        #print("Source channels:", self.source_channels)

        ds_name = stream_info["name"]
        _logger.info(f"{ds_name}: target channels: {self.target_channels}")

        # what is this doing?
        self.properties = {
            "stream_id": 0,
        }

        # or your function to load or compute the statistics
        self.mean, self.stdev = self._create_statistics()

        self.mean_geoinfo, self.stdev_geoinfo = self.mean[self.geoinfo_idx], self.stdev[self.geoinfo_idx]

    def _create_statistics(self):
        statistics = Path(self.stream_info["metadata"]) / self.stream_info["experiment"] / "seviri_statistics.parquet"
        df_stats = pd.read_parquet(statistics)
        mean_lookup = df_stats.set_index('variable')["mean"]
        std_lookup = df_stats.set_index('variable')["std"]

        mean, stdev = [], []

        for ch in self.channels_file:
            if ch in self.exclude:
                mean.append(0.0) # placeholder for those we don't have statistics for
                stdev.append(1.0)
            else:
                mean.append(mean_lookup[ch].astype(np.float32))
                stdev.append(std_lookup[ch].astype(np.float32))
        
        mean = np.array(mean)
        stdev = np.array(stdev)

        print("Mean shape", mean.shape)
        print("Means", mean)

        return mean, stdev

    @override
    def init_empty(self) -> None:
        super().init_empty()
        self.ds = None
        self.len = 0
    
    @override
    def length(self) -> int:
        return self.len

    @override
    def _get(self, idx: TIndex, channels_idx: list[int]) -> ReaderData:
        """
        Get data for window (for either source or target, through public interface)
        Parameters
        ----------
        idx : int
            Index of temporal window
        channels_idx : list[int]
            Selection of channels
        Returns
        -------
        ReaderData providing coords, geoinfos, data, datetimes
        """

        (t_idxs, dtr) = self._get_dataset_idxs(idx)

        if self.ds is None or self.len == 0 or len(t_idxs) == 0:
            return ReaderData.empty(
                num_data_fields=len(channels_idx), num_geo_fields=len(self.geoinfo_idx)
            )

        assert t_idxs[0] >= 0, "index must be non-negative"
        didx_start = t_idxs[0]
        # End is inclusive
        didx_end = t_idxs[-1] + 1

        # extract number of time steps and collapse ensemble dimension
        # ds is a wrapper around zarr with get_coordinate_selection not being exposed since
        # subsetting is pushed to the ctor via frequency argument; this also ensures that no sub-
        # sampling is required here
        sel_channels = [self.channels_file[i] for i in channels_idx]
        data = self.ds[sel_channels].isel(time=slice(didx_start, didx_end)).to_array().values
        # flatten along time dimension
        data = data.transpose([1, 2, 0]).reshape((data.shape[1] * data.shape[2], data.shape[0]))
        # set invalid values to NaN
        mask = data == self.fillvalue
        data[mask] = np.nan

        # construct lat/lon coords
        latlon = np.concatenate(
            [
                np.expand_dims(self.latitudes, 0),
                np.expand_dims(self.longitudes, 0),
            ],
            axis=0,
        ).transpose()

        # repeat len(t_idxs) times
        coords = np.vstack((latlon,) * len(t_idxs))
        geoinfos = np.vstack((self.geoinfo_data,) * len(t_idxs))

        # date time matching #data points of data
        datetimes = np.repeat(self.ds.time[didx_start:didx_end].values, len(data) // len(t_idxs))

        rd = ReaderData(
            coords=coords,
            geoinfos=geoinfos,
            data=data,
            datetimes=datetimes,
        )
        check_reader_data(rd, dtr)

        return rd

    def select_channels(self, ds, ch_type: str) -> NDArray[np.int64]:

        """Select channels based on stream info for either source or target."""

        channels = self.stream_info.get(ch_type)
        assert channels is not None, f"{ch_type} channels need to be specified"
        # sanity check
        is_empty = len(channels) == 0 if channels is not None else False
        if is_empty:
            stream_name = self.stream_info["name"]
            _logger.warning(f"No channel for {stream_name} for {ch_type}.")

        if is_empty:
            _logger.warning(f"No channel selected for {stream_name} for {ch_type}.")
            chs_idx = np.empty(shape=[0], dtype=int)
            channels = []
        else:
            chs_idx = np.sort([self.channels_file.index(ch) for ch in channels])
            channels = [self.channels_file[i] for i in chs_idx]

        return np.array(chs_idx), channels


def _clip_lat(lats: NDArray) -> NDArray[np.float32]:
    """
    Clip latitudes to the range [-90, 90] and ensure periodicity.
    """
    return (2 * np.clip(lats, -90.0, 90.0) - lats).astype(np.float32)


# TODO: move to base class
def _clip_lon(lons: NDArray) -> NDArray[np.float32]:
    """
    Clip longitudes to the range [-180, 180] and ensure periodicity.
    """
    return ((lons + 180.0) % 360.0 - 180.0).astype(np.float32)