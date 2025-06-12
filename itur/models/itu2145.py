import os

import numpy as np
from astropy import units as u

from itur.models.itu1144 import bilinear_2D_interpolator
from itur.models.itu1511 import topographic_altitude
from itur.utils import (
    dataset_dir,
    get_input_type,
    load_data,
    prepare_input_array,
    prepare_output_array,
)


class __ITU2145__:
    """
    Digital maps related to the calculation of gaseous attenuation
    and related effects.

    Available versions include:
        * P.2145-0 (08/22) (Current version)
    """

    # This is an abstract class that contains an instance to a version of the
    # ITU-R P.2145 recommendation.

    def __init__(self):
        self.instance = _ITU2145_0_()

    @property
    def __version__(self):
        return self.instance.__version__

    def mean_surface_total_pressure(self, lat, lon):
        return self.instance.mean_surface_total_pressure(lat, lon)

    def std_dev_surface_total_pressure(self, lat, lon):
        return self.instance.std_dev_surface_total_pressure(lat, lon)

    def surface_total_pressure(self, lat, lon, p):
        return self.instance.surface_total_pressure(lat, lon, p)

    def mean_surface_temperature(self, lat, lon):
        return self.instance.mean_surface_temperature(lat, lon)

    def std_dev_surface_temperature(self, lat, lon):
        return self.instance.std_dev_surface_temperature(lat, lon)

    def surface_temperature(self, lat, lon, p):
        return self.instance.surface_temperature(lat, lon, p)

    def mean_surface_water_vapour_density(self, lat, lon):
        return self.instance.mean_surface_water_vapour_density(lat, lon)

    def std_dev_surface_water_vapour_density(self, lat, lon):
        return self.instance.std_dev_surface_water_vapour_density(lat, lon)

    def surface_water_vapour_density(self, lat, lon, p):
        return self.instance.surface_water_vapour_density(lat, lon, p)

    def mean_integrated_water_vapour_content(self, lat, lon):
        return self.instance.mean_integrated_water_vapour_content(lat, lon)

    def std_dev_integrated_water_vapour_content(self, lat, lon):
        return self.instance.std_dev_integrated_water_vapour_content(lat, lon)

    def integrated_water_vapour_content(self, lat: float, lon: float, p: float) -> float:
        """Calculate the integrated water vapour content at a given location and exceedance probability.

        This method implements the interpolation procedure specified in ITU-R P.2145
        for calculating integrated water vapour content statistics at any desired location.

        Parameters
        ----------
        lat : float
            Latitude in degrees
        lon : float
            Longitude in degrees
        p : float
            Exceedance probability (%)

        Returns
        -------
        float
            Integrated water vapour content at the specified location and probability
        """
        # Determine height above mean sea level of desired location
        point_alt = topographic_altitude(lat, lon)

        # Determine two exceedance probabilities above and below the desired probability, p
        ps = (0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10,
              20, 30, 50, 60, 70, 80, 90, 95, 99)
        idx = np.clip(np.searchsorted(ps, p, side='right') - 1, 0, len(ps) - 1)
        p_below, p_above = ps[idx], ps[min(idx + 1, len(ps) - 1)]
        
        # Determine four surrounding grid points for the given latitude and longitude
        points = _get_bilinear_points(lat, lon)

        # Load integrated water vapour content data for both probabilities
        data = {
            p_val: load_data(os.path.join(dataset_dir, f"2145/v0_V_{p_val}.npz"))
            for p_val in (p_below, p_above)
        }
        interpolators = {
            p_val: bilinear_2D_interpolator(__lats, __lons, data[p_val])
            for p_val in (p_below, p_above)
        }
        
        # Calculate integrated water vapour contents for each point using the specified formula
        # X_i = X_i' * exp(-(alt - alt_i)/vsch_i)
        total_contents = []
        for point in points:
            scale_height = self.surface_water_vapour_density_scale_height(point[0], point[1])
            topographic_height = _zground(point[0], point[1])
            
            # Apply the integrated water vapour content scaling formula
            height_factor = np.exp(-(point_alt - topographic_height) / scale_height)
            content_below = interpolators[p_below](point[0], point[1]) * height_factor
            content_above = interpolators[p_above](point[0], point[1]) * height_factor
            total_contents.append((content_below, content_above))
            
        # Perform bilinear interpolation for both probabilities
        point_interpolator = bilinear_2D_interpolator(
            np.array([pt[0] for pt in points]),
            np.array([pt[1] for pt in points]),
            np.array(total_contents)
        )
        
        # Get interpolated values for both probabilities
        content_below, content_above = point_interpolator(lat, lon)
        
        # Final interpolation on log scale
        if p_below == p_above:
            return content_below
            
        # Linear interpolation on log scale
        log_p = np.log10(p)
        log_p_below = np.log10(p_below)
        log_p_above = np.log10(p_above)
        
        return content_below + (content_above - content_below) * (log_p - log_p_below) / (log_p_above - log_p_below)

    def surface_pressure_scale_height(self, lat, lon):
        return self.instance.surface_pressure_scale_height(lat, lon)

    def surface_temperature_scale_height(self, lat, lon):
        return self.instance.surface_temperature_scale_height(lat, lon)

    def surface_water_vapour_density_scale_height(self, lat, lon):
        return self.instance.surface_water_vapour_density_scale_height(lat, lon)

    def shape_parameter(self, lat, lon):
        return self.instance.shape_parameter(lat, lon)

    def scale_parameter(self, lat, lon):
        return self.instance.scale_parameter(lat, lon)


class _ITU2145_0_:
    def __init__(self):
        self.__version__ = 0
        self.year = 2022
        self.month = 8
        self.link = "https://www.itu.int/rec/R-REC-P.2145-0-202208-I/en"

        # Initialize cache for data files
        self._cache = {
            'pressure': {},  # For P_mean, P_std, P_{p}
            'temperature': {},  # For T_mean, T_std, T_{p}
            'water_vapour': {},  # For W_mean, W_std, W_{p}
            'integrated_vapour': {},  # For V_mean, V_std, V_{p}
            'scale_heights': {},  # For PSCH, TSCH, WSCH
            'ground': None,  # For Z_ground
        }

    def _get_cached_data(self, data_type: str, key: str) -> np.ndarray:
        """Get data from cache or load it if not present.

        Parameters
        ----------
        data_type : str
            Type of data ('pressure', 'temperature', 'water_vapour', 'integrated_vapour', 'scale_heights')
        key : str
            Key for the specific data file

        Returns
        -------
        np.ndarray
            The requested data array
        """
        if data_type == 'ground':
            if self._cache['ground'] is None:
                self._cache['ground'] = load_data(os.path.join(dataset_dir, "2145/v0_Z_ground.npz"))
            return self._cache['ground']

        if key not in self._cache[data_type]:
            if data_type == 'pressure':
                prefix = 'P'
            elif data_type == 'temperature':
                prefix = 'T'
            elif data_type == 'water_vapour':
                prefix = 'W'
            elif data_type == 'integrated_vapour':
                prefix = 'V'
            elif data_type == 'scale_heights':
                if key == 'PSCH':
                    self._cache[data_type][key] = load_data(os.path.join(dataset_dir, "2145/v0_PSCH.npz"))
                elif key == 'TSCH':
                    self._cache[data_type][key] = load_data(os.path.join(dataset_dir, "2145/v0_TSCH.npz"))
                elif key == 'WSCH':
                    self._cache[data_type][key] = load_data(os.path.join(dataset_dir, "2145/v0_WSCH.npz"))
                return self._cache[data_type][key]
            else:
                raise ValueError(f"Unknown data type: {data_type}")

            self._cache[data_type][key] = load_data(os.path.join(dataset_dir, f"2145/v0_{prefix}_{key}.npz"))
        return self._cache[data_type][key]

    def surface_pressure_scale_height(self, lat: float, lon: float) -> float:
        """Calculate the surface pressure scale height.

        Parameters
        ----------
        lat : float
            Latitude in degrees
        lon : float
            Longitude in degrees

        Returns
        -------
        float
            Surface pressure scale height
        """
        data = self._get_cached_data('scale_heights', 'PSCH')
        interpolator = bilinear_2D_interpolator(__lats, __lons, data)
        return interpolator(lat, lon)

    def surface_temperature_scale_height(self, lat: float, lon: float) -> float:
        """Calculate the surface temperature scale height.

        Parameters
        ----------
        lat : float
            Latitude in degrees
        lon : float
            Longitude in degrees

        Returns
        -------
        float
            Surface temperature scale height
        """
        data = self._get_cached_data('scale_heights', 'TSCH')
        interpolator = bilinear_2D_interpolator(__lats, __lons, data)
        return interpolator(lat, lon)

    def surface_water_vapour_density_scale_height(self, lat: float, lon: float) -> float:
        """Calculate the surface water vapour density scale height.

        Parameters
        ----------
        lat : float
            Latitude in degrees
        lon : float
            Longitude in degrees

        Returns
        -------
        float
            Surface water vapour density scale height
        """
        data = self._get_cached_data('scale_heights', 'WSCH')
        interpolator = bilinear_2D_interpolator(__lats, __lons, data)
        return interpolator(lat, lon)

    def mean_surface_total_pressure(self, lat: float, lon: float) -> float:
        """Calculate the mean surface total pressure at a given location.

        This method implements the interpolation procedure specified in ITU-R P.2145
        for calculating mean surface total pressure statistics at any desired location.

        Parameters
        ----------
        lat : float
            Latitude in degrees
        lon : float
            Longitude in degrees

        Returns
        -------
        float
            Mean surface total pressure at the specified location
        """
        # Determine height above mean sea level of desired location
        point_alt = topographic_altitude(lat, lon)
        
        # Determine four surrounding grid points for the given latitude and longitude
        points = _get_bilinear_points(lat, lon)

        # Load mean pressure data
        data = self._get_cached_data('pressure', 'mean')
        interpolator = bilinear_2D_interpolator(__lats, __lons, data)
        
        # Calculate pressures for each point using the specified formula
        # X_i = X_i' * exp(-(alt - alt_i)/psch_i)
        total_pressures = []
        for point in points:
            scale_height = self.surface_pressure_scale_height(point[0], point[1])
            topographic_height = _zground(point[0], point[1])
            
            # Apply the pressure scaling formula
            height_factor = np.exp(-(point_alt - topographic_height) / scale_height)
            pressure = interpolator(point[0], point[1]) * height_factor
            total_pressures.append(pressure)
            
        # Perform bilinear interpolation
        point_interpolator = bilinear_2D_interpolator(
            np.array([pt[0] for pt in points]),
            np.array([pt[1] for pt in points]),
            np.array(total_pressures)
        )
        
        return point_interpolator(lat, lon)

    def std_dev_surface_total_pressure(self, lat: float, lon: float) -> float:
        """Calculate the standard deviation of surface total pressure at a given location.

        This method implements the interpolation procedure specified in ITU-R P.2145
        for calculating standard deviation of surface total pressure statistics at any desired location.

        Parameters
        ----------
        lat : float
            Latitude in degrees
        lon : float
            Longitude in degrees

        Returns
        -------
        float
            Standard deviation of surface total pressure at the specified location
        """
        # Determine height above mean sea level of desired location
        point_alt = topographic_altitude(lat, lon)
        
        # Determine four surrounding grid points for the given latitude and longitude
        points = _get_bilinear_points(lat, lon)

        # Load standard deviation pressure data
        data = self._get_cached_data('pressure', 'std')
        interpolator = bilinear_2D_interpolator(__lats, __lons, data)
        
        # Calculate pressures for each point using the specified formula
        # X_i = X_i' * exp(-(alt - alt_i)/psch_i)
        total_pressures = []
        for point in points:
            scale_height = self.surface_pressure_scale_height(point[0], point[1])
            topographic_height = _zground(point[0], point[1])
            
            # Apply the pressure scaling formula
            height_factor = np.exp(-(point_alt - topographic_height) / scale_height)
            pressure = interpolator(point[0], point[1]) * height_factor
            total_pressures.append(pressure)
            
        # Perform bilinear interpolation
        point_interpolator = bilinear_2D_interpolator(
            np.array([pt[0] for pt in points]),
            np.array([pt[1] for pt in points]),
            np.array(total_pressures)
        )
        
        return point_interpolator(lat, lon)

    def surface_total_pressure(self, lat: float, lon: float, p: float) -> float:
        """Calculate the surface total pressure at a given location and exceedance probability.

        This method implements the interpolation procedure specified in ITU-R P.2145
        for calculating surface total pressure statistics at any desired location.

        Parameters
        ----------
        lat : float
            Latitude in degrees
        lon : float
            Longitude in degrees
        p : float
            Exceedance probability (%)

        Returns
        -------
        float
            Surface total pressure at the specified location and probability
        """
        # Determine height above mean sea level of desired location
        point_alt = topographic_altitude(lat, lon)

        # Determine two exceedance probabilities above and below the desired probability, p
        ps = (0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10,
              20, 30, 50, 60, 70, 80, 90, 95, 99)
        idx = np.clip(np.searchsorted(ps, p, side='right') - 1, 0, len(ps) - 1)
        p_below, p_above = ps[idx], ps[min(idx + 1, len(ps) - 1)]
        
        # Determine four surrounding grid points for the given latitude and longitude
        points = _get_bilinear_points(lat, lon)

        # Load pressure data for both probabilities
        data = {
            p_val: self._get_cached_data('pressure', str(p_val))
            for p_val in (p_below, p_above)
        }
        interpolators = {
            p_val: bilinear_2D_interpolator(__lats, __lons, data[p_val])
            for p_val in (p_below, p_above)
        }
        
        # Calculate pressures for each point using the specified formula
        # X_i = X_i' * exp(-(alt - alt_i)/psch_i)
        total_pressures = []
        for point in points:
            scale_height = self.surface_pressure_scale_height(point[0], point[1])
            topographic_height = _zground(point[0], point[1])
            
            # Apply the pressure scaling formula
            height_factor = np.exp(-(point_alt - topographic_height) / scale_height)
            pressure_below = interpolators[p_below](point[0], point[1]) * height_factor
            pressure_above = interpolators[p_above](point[0], point[1]) * height_factor
            total_pressures.append((pressure_below, pressure_above))
            
        # Perform bilinear interpolation for both probabilities
        point_interpolator = bilinear_2D_interpolator(
            np.array([pt[0] for pt in points]),
            np.array([pt[1] for pt in points]),
            np.array(total_pressures)
        )
        
        # Get interpolated values for both probabilities
        pressure_below, pressure_above = point_interpolator(lat, lon)
        
        # Final interpolation on log scale
        if p_below == p_above:
            return pressure_below
            
        # Linear interpolation on log scale
        log_p = np.log10(p)
        log_p_below = np.log10(p_below)
        log_p_above = np.log10(p_above)
        
        return pressure_below + (pressure_above - pressure_below) * (log_p - log_p_below) / (log_p_above - log_p_below)

    def mean_surface_temperature(self, lat: float, lon: float) -> float:
        """Calculate the mean surface temperature at a given location.

        This method implements the interpolation procedure specified in ITU-R P.2145
        for calculating mean surface temperature statistics at any desired location.

        Parameters
        ----------
        lat : float
            Latitude in degrees
        lon : float
            Longitude in degrees

        Returns
        -------
        float
            Mean surface temperature at the specified location
        """
        # Determine height above mean sea level of desired location
        point_alt = topographic_altitude(lat, lon)
        
        # Determine four surrounding grid points for the given latitude and longitude
        points = _get_bilinear_points(lat, lon)

        # Load mean temperature data
        data = self._get_cached_data('temperature', 'mean')
        interpolator = bilinear_2D_interpolator(__lats, __lons, data)
        
        # Calculate temperatures for each point using the specified formula
        # X_i = X_i' + tsch_i * (alt - alt_i)
        total_temperatures = []
        for point in points:
            scale_height = self.surface_temperature_scale_height(point[0], point[1])
            topographic_height = _zground(point[0], point[1])
            
            # Apply the temperature scaling formula
            temperature = (interpolator(point[0], point[1]) + 
                         scale_height * (point_alt - topographic_height))
            total_temperatures.append(temperature)
            
        # Perform bilinear interpolation
        point_interpolator = bilinear_2D_interpolator(
            np.array([pt[0] for pt in points]),
            np.array([pt[1] for pt in points]),
            np.array(total_temperatures)
        )
        
        return point_interpolator(lat, lon)

    def std_dev_surface_temperature(self, lat: float, lon: float) -> float:
        """Calculate the standard deviation of surface temperature at a given location.

        This method implements the interpolation procedure specified in ITU-R P.2145
        for calculating standard deviation of surface temperature statistics at any desired location.

        Parameters
        ----------
        lat : float
            Latitude in degrees
        lon : float
            Longitude in degrees

        Returns
        -------
        float
            Standard deviation of surface temperature at the specified location
        """
        # Determine four surrounding grid points for the given latitude and longitude
        points = _get_bilinear_points(lat, lon)

        # Load standard deviation temperature data
        data = self._get_cached_data('temperature', 'std')
        interpolator = bilinear_2D_interpolator(__lats, __lons, data)
        
        # Calculate temperatures for each point
        # X_i = X_i' (no scaling for standard deviation)
        total_temperatures = [interpolator(point[0], point[1]) for point in points]
            
        # Perform bilinear interpolation
        point_interpolator = bilinear_2D_interpolator(
            np.array([pt[0] for pt in points]),
            np.array([pt[1] for pt in points]),
            np.array(total_temperatures)
        )
        
        return point_interpolator(lat, lon)

    def surface_temperature(self, lat: float, lon: float, p: float) -> float:
        """Calculate the surface temperature at a given location and exceedance probability.

        This method implements the interpolation procedure specified in ITU-R P.2145
        for calculating surface temperature statistics at any desired location.

        Parameters
        ----------
        lat : float
            Latitude in degrees
        lon : float
            Longitude in degrees
        p : float
            Exceedance probability (%)

        Returns
        -------
        float
            Surface temperature at the specified location and probability
        """
        # Determine height above mean sea level of desired location
        point_alt = topographic_altitude(lat, lon)

        # Determine two exceedance probabilities above and below the desired probability, p
        ps = (0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10,
              20, 30, 50, 60, 70, 80, 90, 95, 99)
        idx = np.clip(np.searchsorted(ps, p, side='right') - 1, 0, len(ps) - 1)
        p_below, p_above = ps[idx], ps[min(idx + 1, len(ps) - 1)]
        
        # Determine four surrounding grid points for the given latitude and longitude
        points = _get_bilinear_points(lat, lon)

        # Load temperature data for both probabilities
        data = {
            p_val: self._get_cached_data('temperature', str(p_val))
            for p_val in (p_below, p_above)
        }
        interpolators = {
            p_val: bilinear_2D_interpolator(__lats, __lons, data[p_val])
            for p_val in (p_below, p_above)
        }
        
        # Calculate temperatures for each point using the specified formula
        # X_i = X_i' + tsch_i * (alt - alt_i)
        total_temperatures = []
        for point in points:
            scale_height = self.surface_temperature_scale_height(point[0], point[1])
            topographic_height = _zground(point[0], point[1])
            
            # Apply the temperature scaling formula
            temp_below = (interpolators[p_below](point[0], point[1]) + 
                         scale_height * (point_alt - topographic_height))
            temp_above = (interpolators[p_above](point[0], point[1]) + 
                         scale_height * (point_alt - topographic_height))
            total_temperatures.append((temp_below, temp_above))
            
        # Perform bilinear interpolation for both probabilities
        point_interpolator = bilinear_2D_interpolator(
            np.array([pt[0] for pt in points]),
            np.array([pt[1] for pt in points]),
            np.array(total_temperatures)
        )
        
        # Get interpolated values for both probabilities
        temp_below, temp_above = point_interpolator(lat, lon)
        
        # Final interpolation on log scale
        if p_below == p_above:
            return temp_below
            
        # Linear interpolation on log scale
        log_p = np.log10(p)
        log_p_below = np.log10(p_below)
        log_p_above = np.log10(p_above)
        
        return temp_below + (temp_above - temp_below) * (log_p - log_p_below) / (log_p_above - log_p_below)

    def mean_surface_water_vapour_density(self, lat: float, lon: float) -> float:
        """Calculate the mean surface water vapour density at a given location.

        This method implements the interpolation procedure specified in ITU-R P.2145
        for calculating mean surface water vapour density statistics at any desired location.

        Parameters
        ----------
        lat : float
            Latitude in degrees
        lon : float
            Longitude in degrees

        Returns
        -------
        float
            Mean surface water vapour density at the specified location
        """
        # Determine height above mean sea level of desired location
        point_alt = topographic_altitude(lat, lon)
        
        # Determine four surrounding grid points for the given latitude and longitude
        points = _get_bilinear_points(lat, lon)

        # Load mean water vapour density data
        data = self._get_cached_data('water_vapour', 'mean')
        interpolator = bilinear_2D_interpolator(__lats, __lons, data)
        
        # Calculate water vapour densities for each point using the specified formula
        # X_i = X_i' * exp(-(alt - alt_i)/vsch_i)
        total_densities = []
        for point in points:
            scale_height = self.surface_water_vapour_density_scale_height(point[0], point[1])
            topographic_height = _zground(point[0], point[1])
            
            # Apply the water vapour density scaling formula
            height_factor = np.exp(-(point_alt - topographic_height) / scale_height)
            density = interpolator(point[0], point[1]) * height_factor
            total_densities.append(density)
            
        # Perform bilinear interpolation
        point_interpolator = bilinear_2D_interpolator(
            np.array([pt[0] for pt in points]),
            np.array([pt[1] for pt in points]),
            np.array(total_densities)
        )
        
        return point_interpolator(lat, lon)

    def std_dev_surface_water_vapour_density(self, lat: float, lon: float) -> float:
        """Calculate the standard deviation of surface water vapour density at a given location.

        This method implements the interpolation procedure specified in ITU-R P.2145
        for calculating standard deviation of surface water vapour density statistics at any desired location.

        Parameters
        ----------
        lat : float
            Latitude in degrees
        lon : float
            Longitude in degrees

        Returns
        -------
        float
            Standard deviation of surface water vapour density at the specified location
        """
        # Determine height above mean sea level of desired location
        point_alt = topographic_altitude(lat, lon)
        
        # Determine four surrounding grid points for the given latitude and longitude
        points = _get_bilinear_points(lat, lon)

        # Load standard deviation water vapour density data
        data = self._get_cached_data('water_vapour', 'std')
        interpolator = bilinear_2D_interpolator(__lats, __lons, data)
        
        # Calculate water vapour densities for each point using the specified formula
        # X_i = X_i' * exp(-(alt - alt_i)/vsch_i)
        total_densities = []
        for point in points:
            scale_height = self.surface_water_vapour_density_scale_height(point[0], point[1])
            topographic_height = _zground(point[0], point[1])
            
            # Apply the water vapour density scaling formula
            height_factor = np.exp(-(point_alt - topographic_height) / scale_height)
            density = interpolator(point[0], point[1]) * height_factor
            total_densities.append(density)
            
        # Perform bilinear interpolation
        point_interpolator = bilinear_2D_interpolator(
            np.array([pt[0] for pt in points]),
            np.array([pt[1] for pt in points]),
            np.array(total_densities)
        )
        
        return point_interpolator(lat, lon)

    def surface_water_vapour_density(self, lat: float, lon: float, p: float) -> float:
        """Calculate the surface water vapour density at a given location and exceedance probability.

        This method implements the interpolation procedure specified in ITU-R P.2145
        for calculating surface water vapour density statistics at any desired location.

        Parameters
        ----------
        lat : float
            Latitude in degrees
        lon : float
            Longitude in degrees
        p : float
            Exceedance probability (%)

        Returns
        -------
        float
            Surface water vapour density at the specified location and probability
        """
        # Determine height above mean sea level of desired location
        point_alt = topographic_altitude(lat, lon)

        # Determine two exceedance probabilities above and below the desired probability, p
        ps = (0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10,
              20, 30, 50, 60, 70, 80, 90, 95, 99)
        idx = np.clip(np.searchsorted(ps, p, side='right') - 1, 0, len(ps) - 1)
        p_below, p_above = ps[idx], ps[min(idx + 1, len(ps) - 1)]
        
        # Determine four surrounding grid points for the given latitude and longitude
        points = _get_bilinear_points(lat, lon)

        # Load water vapour density data for both probabilities
        data = {
            p_val: self._get_cached_data('water_vapour', str(p_val))
            for p_val in (p_below, p_above)
        }
        interpolators = {
            p_val: bilinear_2D_interpolator(__lats, __lons, data[p_val])
            for p_val in (p_below, p_above)
        }
        
        # Calculate water vapour densities for each point using the specified formula
        # X_i = X_i' * exp(-(alt - alt_i)/vsch_i)
        total_densities = []
        for point in points:
            scale_height = self.surface_water_vapour_density_scale_height(point[0], point[1])
            topographic_height = _zground(point[0], point[1])
            
            # Apply the water vapour density scaling formula
            height_factor = np.exp(-(point_alt - topographic_height) / scale_height)
            density_below = interpolators[p_below](point[0], point[1]) * height_factor
            density_above = interpolators[p_above](point[0], point[1]) * height_factor
            total_densities.append((density_below, density_above))
            
        # Perform bilinear interpolation for both probabilities
        point_interpolator = bilinear_2D_interpolator(
            np.array([pt[0] for pt in points]),
            np.array([pt[1] for pt in points]),
            np.array(total_densities)
        )
        
        # Get interpolated values for both probabilities
        density_below, density_above = point_interpolator(lat, lon)
        
        # Final interpolation on log scale
        if p_below == p_above:
            return density_below
            
        # Linear interpolation on log scale
        log_p = np.log10(p)
        log_p_below = np.log10(p_below)
        log_p_above = np.log10(p_above)
        
        return density_below + (density_above - density_below) * (log_p - log_p_below) / (log_p_above - log_p_below)

    def mean_integrated_water_vapour_content(self, lat: float, lon: float) -> float:
        """Calculate the mean integrated water vapour content at a given location.

        This method implements the interpolation procedure specified in ITU-R P.2145
        for calculating mean integrated water vapour content statistics at any desired location.

        Parameters
        ----------
        lat : float
            Latitude in degrees
        lon : float
            Longitude in degrees

        Returns
        -------
        float
            Mean integrated water vapour content at the specified location
        """
        # Determine height above mean sea level of desired location
        point_alt = topographic_altitude(lat, lon)
        
        # Determine four surrounding grid points for the given latitude and longitude
        points = _get_bilinear_points(lat, lon)

        # Load mean integrated water vapour content data
        data = self._get_cached_data('integrated_vapour', 'mean')
        interpolator = bilinear_2D_interpolator(__lats, __lons, data)
        
        # Calculate integrated water vapour contents for each point using the specified formula
        # X_i = X_i' * exp(-(alt - alt_i)/vsch_i)
        total_contents = []
        for point in points:
            scale_height = self.surface_water_vapour_density_scale_height(point[0], point[1])
            topographic_height = _zground(point[0], point[1])
            
            # Apply the integrated water vapour content scaling formula
            height_factor = np.exp(-(point_alt - topographic_height) / scale_height)
            content = interpolator(point[0], point[1]) * height_factor
            total_contents.append(content)
            
        # Perform bilinear interpolation
        point_interpolator = bilinear_2D_interpolator(
            np.array([pt[0] for pt in points]),
            np.array([pt[1] for pt in points]),
            np.array(total_contents)
        )
        
        return point_interpolator(lat, lon)

    def std_dev_integrated_water_vapour_content(self, lat: float, lon: float) -> float:
        """Calculate the standard deviation of integrated water vapour content at a given location.

        This method implements the interpolation procedure specified in ITU-R P.2145
        for calculating standard deviation of integrated water vapour content statistics at any desired location.

        Parameters
        ----------
        lat : float
            Latitude in degrees
        lon : float
            Longitude in degrees

        Returns
        -------
        float
            Standard deviation of integrated water vapour content at the specified location
        """
        # Determine height above mean sea level of desired location
        point_alt = topographic_altitude(lat, lon)
        
        # Determine four surrounding grid points for the given latitude and longitude
        points = _get_bilinear_points(lat, lon)

        # Load standard deviation integrated water vapour content data
        data = self._get_cached_data('integrated_vapour', 'std')
        interpolator = bilinear_2D_interpolator(__lats, __lons, data)
        
        # Calculate integrated water vapour contents for each point using the specified formula
        # X_i = X_i' * exp(-(alt - alt_i)/vsch_i)
        total_contents = []
        for point in points:
            scale_height = self.surface_water_vapour_density_scale_height(point[0], point[1])
            topographic_height = _zground(point[0], point[1])
            
            # Apply the integrated water vapour content scaling formula
            height_factor = np.exp(-(point_alt - topographic_height) / scale_height)
            content = interpolator(point[0], point[1]) * height_factor
            total_contents.append(content)
            
        # Perform bilinear interpolation
        point_interpolator = bilinear_2D_interpolator(
            np.array([pt[0] for pt in points]),
            np.array([pt[1] for pt in points]),
            np.array(total_contents)
        )
        
        return point_interpolator(lat, lon)

    def integrated_water_vapour_content(self, lat: float, lon: float, p: float) -> float:
        """Calculate the integrated water vapour content at a given location and exceedance probability.

        This method implements the interpolation procedure specified in ITU-R P.2145
        for calculating integrated water vapour content statistics at any desired location.

        Parameters
        ----------
        lat : float
            Latitude in degrees
        lon : float
            Longitude in degrees
        p : float
            Exceedance probability (%)

        Returns
        -------
        float
            Integrated water vapour content at the specified location and probability
        """
        # Determine height above mean sea level of desired location
        point_alt = topographic_altitude(lat, lon)

        # Determine two exceedance probabilities above and below the desired probability, p
        ps = (0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10,
              20, 30, 50, 60, 70, 80, 90, 95, 99)
        idx = np.clip(np.searchsorted(ps, p, side='right') - 1, 0, len(ps) - 1)
        p_below, p_above = ps[idx], ps[min(idx + 1, len(ps) - 1)]
        
        # Determine four surrounding grid points for the given latitude and longitude
        points = _get_bilinear_points(lat, lon)

        # Load integrated water vapour content data for both probabilities
        data = {
            p_val: self._get_cached_data('integrated_vapour', str(p_val))
            for p_val in (p_below, p_above)
        }
        interpolators = {
            p_val: bilinear_2D_interpolator(__lats, __lons, data[p_val])
            for p_val in (p_below, p_above)
        }
        
        # Calculate integrated water vapour contents for each point using the specified formula
        # X_i = X_i' * exp(-(alt - alt_i)/vsch_i)
        total_contents = []
        for point in points:
            scale_height = self.surface_water_vapour_density_scale_height(point[0], point[1])
            topographic_height = _zground(point[0], point[1])
            
            # Apply the integrated water vapour content scaling formula
            height_factor = np.exp(-(point_alt - topographic_height) / scale_height)
            content_below = interpolators[p_below](point[0], point[1]) * height_factor
            content_above = interpolators[p_above](point[0], point[1]) * height_factor
            total_contents.append((content_below, content_above))
            
        # Perform bilinear interpolation for both probabilities
        point_interpolator = bilinear_2D_interpolator(
            np.array([pt[0] for pt in points]),
            np.array([pt[1] for pt in points]),
            np.array(total_contents)
        )
        
        # Get interpolated values for both probabilities
        content_below, content_above = point_interpolator(lat, lon)
        
        # Final interpolation on log scale
        if p_below == p_above:
            return content_below
            
        # Linear interpolation on log scale
        log_p = np.log10(p)
        log_p_below = np.log10(p_below)
        log_p_above = np.log10(p_above)
        
        return content_below + (content_above - content_below) * (log_p - log_p_below) / (log_p_above - log_p_below)


__model = __ITU2145__()
__lats = np.arange(-90.0, 90.25, 0.25)
__lons = np.arange(-180.0, 180.25, 0.25)


def get_version():
    return __model.__version__


def mean_surface_total_pressure(lat, lon, p):
    type_output = get_input_type(lat)
    lat = prepare_input_array(lat)
    lon = prepare_input_array(lon)
    lon = np.mod(lon, 360)
    val = __model.mean_surface_total_pressure(lat, lon)
    return prepare_output_array(val, type_output) * u.hPa


def std_dev_surface_total_pressure(lat, lon, p):
    type_output = get_input_type(lat)
    lat = prepare_input_array(lat)
    lon = prepare_input_array(lon)
    lon = np.mod(lon, 360)
    val = __model.std_dev_surface_total_pressure(lat, lon)
    return prepare_output_array(val, type_output) * u.hPa


def surface_total_pressure(lat, lon, p):
    type_output = get_input_type(lat)
    lat = prepare_input_array(lat)
    lon = prepare_input_array(lon)
    lon = np.mod(lon, 360)
    val = __model.surface_total_pressure(lat, lon, p)
    return prepare_output_array(val, type_output) * u.hPa


def mean_surface_temperature(lat, lon, p):
    type_output = get_input_type(lat)
    lat = prepare_input_array(lat)
    lon = prepare_input_array(lon)
    lon = np.mod(lon, 360)
    val = __model.mean_surface_temperature(lat, lon)
    return prepare_output_array(val, type_output) * u.K


def std_dev_surface_temperature(lat, lon, p):
    type_output = get_input_type(lat)
    lat = prepare_input_array(lat)
    lon = prepare_input_array(lon)
    lon = np.mod(lon, 360)
    val = __model.std_dev_surface_temperature(lat, lon)
    return prepare_output_array(val, type_output) * u.K


def surface_temperature(lat, lon, p):
    type_output = get_input_type(lat)
    lat = prepare_input_array(lat)
    lon = prepare_input_array(lon)
    lon = np.mod(lon, 360)
    val = __model.surface_temperature(lat, lon, p)
    return prepare_output_array(val, type_output) * u.K


def mean_surface_water_vapour_density(lat, lon, p):
    type_output = get_input_type(lat)
    lat = prepare_input_array(lat)
    lon = prepare_input_array(lon)
    lon = np.mod(lon, 360)
    val = __model.mean_surface_water_vapour_density(lat, lon)
    return prepare_output_array(val, type_output) * u.g / u.m**3


def std_dev_surface_water_vapour_density(lat, lon, p):
    type_output = get_input_type(lat)
    lat = prepare_input_array(lat)
    lon = prepare_input_array(lon)
    lon = np.mod(lon, 360)
    val = __model.std_dev_surface_water_vapour_density(lat, lon)
    return prepare_output_array(val, type_output) * u.g / u.m**3


def surface_water_vapour_density(lat, lon, p):
    type_output = get_input_type(lat)
    lat = prepare_input_array(lat)
    lon = prepare_input_array(lon)
    lon = np.mod(lon, 360)
    val = __model.surface_water_vapour_density(lat, lon, p)
    return prepare_output_array(val, type_output) * u.g / u.m**3


def mean_integrated_water_vapour_content(lat, lon, p):
    type_output = get_input_type(lat)
    lat = prepare_input_array(lat)
    lon = prepare_input_array(lon)
    lon = np.mod(lon, 360)
    val = __model.mean_integrated_water_vapour_content(lat, lon)
    return prepare_output_array(val, type_output) * u.g / u.m**3


def std_dev_integrated_water_vapour_content(lat, lon, p):
    type_output = get_input_type(lat)
    lat = prepare_input_array(lat)
    lon = prepare_input_array(lon)
    lon = np.mod(lon, 360)
    val = __model.std_dev_integrated_water_vapour_content(lat, lon)
    return prepare_output_array(val, type_output) * u.g / u.m**3


def integrated_water_vapour_content(lat, lon, p):
    type_output = get_input_type(lat)
    lat = prepare_input_array(lat)
    lon = prepare_input_array(lon)
    lon = np.mod(lon, 360)
    val = __model.integrated_water_vapour_content(lat, lon, p)
    return prepare_output_array(val, type_output) * u.g / u.m**3


def surface_pressure_scale_height(lat, lon):
    type_output = get_input_type(lat)
    lat = prepare_input_array(lat)
    lon = prepare_input_array(lon)
    lon = np.mod(lon, 360)
    val = __model.surface_pressure_scale_height(lat, lon)
    return prepare_output_array(val, type_output) * u.km


def surface_temperature_scale_height(lat, lon):
    type_output = get_input_type(lat)
    lat = prepare_input_array(lat)
    lon = prepare_input_array(lon)
    lon = np.mod(lon, 360)
    val = __model.surface_temperature_scale_height(lat, lon)
    return prepare_output_array(val, type_output) * u.km


def surface_water_vapour_density_scale_height(lat, lon):
    """Calculate the surface water vapour density scale height.

    Parameters
    ----------
    lat : number or Quantity
        Latitude (deg)
    lon : number or Quantity
        Longitude (deg)

    Returns
    -------
    h : Quantity
        Surface water vapour density scale height (km)

    References
    ----------
    [1] ITU-R P.2145-0, Digital maps related to the calculation of gaseous attenuation and related effects, 2022
    """
    type_output = get_input_type(lat)
    lat = prepare_input_array(lat)
    lon = prepare_input_array(lon)
    lon = np.mod(lon, 360)
    val = __model.surface_water_vapour_density_scale_height(lat, lon)
    return prepare_output_array(val, type_output) * u.km


def shape_parameter(self, lat, lon):
    """Calculate the Weibull shape parameter.

    Parameters
    ----------
    lat : number or Quantity
        Latitude (deg)
    lon : number or Quantity
        Longitude (deg)

    Returns
    -------
    k : Quantity
        Weibull shape parameter (-)

    References
    ----------
    [1] ITU-R P.2145-0, Digital maps related to the calculation of gaseous attenuation and related effects, 2022
    """
    type_output = get_input_type(lat)
    lat = prepare_input_array(lat)
    lon = prepare_input_array(lon)
    lon = np.mod(lon, 360)
    val = __model.shape_parameter(lat, lon)
    return prepare_output_array(val, type_output) * u.dimensionless_unscaled


def scale_parameter(self, lat, lon):
    """Calculate the Weibull scale parameter.

    Parameters
    ----------
    lat : number or Quantity
        Latitude (deg)
    lon : number or Quantity
        Longitude (deg)

    Returns
    -------
    k : Quantity
        Weibull scale parameter (-)

    References
    ----------
    [1] ITU-R P.2145-0, Digital maps related to the calculation of gaseous attenuation and related effects, 2022
    """

    type_output = get_input_type(lat)
    lat = prepare_input_array(lat)
    lon = prepare_input_array(lon)
    lon = np.mod(lon, 360)
    val = __model.scale_parameter(lat, lon)
    return prepare_output_array(val, type_output) * u.dimensionless_unscaled


def _get_bilinear_points(
    lat: float, lon: float, lats_o: np.ndarray = __lats, lons_o: np.ndarray = __lons
) -> tuple:
    """
    Return the four grid points (lat, lon) used for bilinear interpolation
    at the given (lat, lon) for a regular grid.

    Parameters
    ----------
    lats_o : np.ndarray
        2D array of latitude grid points.
    lons_o : np.ndarray
        2D array of longitude grid points.
    lat : float
        Query latitude.
    lon : float
        Query longitude.

    Returns
    -------
    tuple
        ((lat00, lon00), (lat01, lon01), (lat10, lon10), (lat11, lon11))
        The four grid points surrounding (lat, lon).
    """
    lats = np.flipud(lats_o[:, 0])
    lons = lons_o[0, :]

    i = np.searchsorted(lats, lat) - 1
    j = np.searchsorted(lons, lon) - 1

    i = np.clip(i, 0, len(lats) - 2)
    j = np.clip(j, 0, len(lons) - 2)

    lat00, lon00 = lats[i], lons[j]
    lat01, lon01 = lats[i], lons[j + 1]
    lat10, lon10 = lats[i + 1], lons[j]
    lat11, lon11 = lats[i + 1], lons[j + 1]

    return ((lat00, lon00), (lat01, lon01), (lat10, lon10), (lat11, lon11))

def _zground(lat: float, lon: float) -> float:
    """
    Calculate the height above mean sea level (zground) for a given latitude and longitude.

    Parameters:
        lat: float
            Latitude in degrees
        lon: float
            Longitude in degrees

    Returns:
        zground: float
            Height above mean sea level (km)
    """
    data = __model._get_cached_data('ground', '')
    interpolator = bilinear_2D_interpolator(__lats, __lons, data)
    return interpolator(lat, lon)
