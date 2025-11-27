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

        ds = xr.open_dataset(filename, group= "seviri", engine="zarr")

        if tw_handler.t_start >= ds.time.max() or tw_handler.t_end <= ds.time.min():
            name = stream_info["name"]
            _logger.warning(f"{name} is not supported over data loader window. Stream is skipped.")
            super().__init__(tw_handler, stream_info)
            self._init_empty()
            return
        
        if "frequency" in stream_info:
            assert False, "Frequency sub-sampling currently not supported"

        data_start_time = ds.time[0].values
        data_end_time = ds.time[20].values

        period = (data_end_time - data_start_time)
        print(f"Data period: {period}")

        assert data_start_time is not None and data_end_time is not None, (
            data_start_time,
            data_end_time,
        )
        super().__init__(
            tw_handler,
            stream_info,
            data_start_time,
            data_end_time,
            period,
        )
        # If there is no overlap with the time range, no need to keep the dataset.
        if tw_handler.t_start >= data_end_time or tw_handler.t_end <= data_start_time:
            self._init_empty()
            return
        else:
            self.ds = ds
            self.len = len(ds)

        self.channels_file = [k for k in self.ds.keys()]

        # caches lats and lons
        lat_name = stream_info.get("latitude_name", "latitude")
        self.latitudes = _clip_lat(np.array(ds[lat_name], dtype=np32))
        lon_name = stream_info.get("longitude_name", "longitude")
        self.longitudes = _clip_lon(np.array(ds[lon_name], dtype=np32))

        self.geoinfo_channels = stream_info.get("geoinfos", [])
        self.geoinfo_idx = [self.channels_file.index(ch) for ch in self.geoinfo_channels]
        # cache geoinfos
        self.geoinfo_data = np.stack([np.array(ds[ch], dtype=np32) for ch in self.geoinfo_channels])
        self.geoinfo_data = self.geoinfo_data.transpose()

        # select/filter requested source channels
        self.source_idx = self.select_channels(ds, "source")
        self.source_channels = [self.channels_file[i] for i in self.source_idx]

        # select/filter requested target channels
        self.target_idx = self.select_channels(ds, "target")
        self.target_channels = [self.channels_file[i] for i in self.target_idx]


    def _compute_mean_stdev(self):
        """Implement this function if you need to compute mean and std on the fly."""
        pass

    @override
    def _init_empty(self) -> None:
        super()._init_empty()
        self.ds = None
        self.len = 0
    
    @override
    def length(self) -> int:
        return self.len

    @override
    def _get(self, idx: TIndex) -> ReaderData:
        """
        Get data for window (for either source or target, through public interface)
        Parameters
        ----------
        idx : int
            Index of temporal window
        channels_idx : np.array
            Selection of channels
        Returns
        -------
        ReaderData providing coords, geoinfos, data, datetimes
        """
        pass

    def select_channels(self, ds, ch_type: str) -> NDArray[np.int64]:
        pass