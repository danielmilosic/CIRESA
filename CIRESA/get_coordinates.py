import numpy as np
from astropy.time import Time
# Import the necessary function for retrieving spacecraft coordinates.
from sunpy.coordinates import get_horizons_coord
import sunpy.coordinates
import pandas as pd
from astropy.coordinates import SkyCoord
import sunpy.coordinates as c
import astropy.units as u


def get_carrington_longitude(date):

    # Ensure that input is a NumPy array
    date = np.asarray(date)
    
    # Set the reference date for the start of Carrington rotations (November 9, 1853, 16:00:00).
    reference_date = Time('1853-11-09 16:00:00')
    
    # Calculate the number of days since the reference date.
    days_since_reference = (Time(date) - reference_date).jd

    # Calculate the current Carrington rotation number.
    #carrington_rotation2 = 1 + (days_since_reference / 27.2753)
    carrington_rotation = sunpy.coordinates.sun.carrington_rotation_number(t=date)
    #print(carrington_rotation-carrington_rotation2)
    
    # Calculate the Carrington longitude by converting the fractional part of the rotation.
    

    carrington_rotation = np.asarray(carrington_rotation)
  
    if np.isscalar(carrington_rotation) or carrington_rotation.size == 1:
        # Handle the scalar case
        carr_lon = (1 + int(carrington_rotation) - carrington_rotation) * 360
        carr_lons = np.array([carr_lon % 360])
    else:
        # Handle the array case
        carr_lons = []
        for rot in carrington_rotation:
            carr_lon = (1 + int(rot) - rot) * 360
            carr_lons.append(carr_lon)

        carr_lons = np.asarray(carr_lons) % 360

    
    return carr_lons

def get_coordinates(data, spcrft):

    # Initialize lists to store Carrington longitudes, distances, latitudes, and longitudes.
    carr_lons = []
    distances = []
    lats = []
    lons = []
    
    # Loop over each time point in the data.
    for i in range(len(data)):
        # Print the progress of the loop to track which iteration is being processed.
        print(spcrft, ': ', i, 'out of', len(data))
        
        # Get the spacecraft coordinates at the given time using the Horizons system.
        stony_coord = get_horizons_coord(spcrft, pd.to_datetime(data.index[i]))
        
        # Calculate the Carrington longitude for the given time.
        carrington_longitude = get_carrington_longitude(data.index[i])

        # Calculate the spacecraft's Carrington longitude by adding the computed longitude and the spacecraft's longitude.
        spacecraft_carrington_phi = carrington_longitude + stony_coord.lon.value
        carr_lons.append(spacecraft_carrington_phi)

        # Store the spacecraft's heliocentric inertial longitude.
        lon = stony_coord.heliocentricinertial.lon.value
        lons.append(lon)

        # Store the spacecraft's radial distance from the Sun.
        distance = stony_coord.radius.value
        distances.append(distance)

        # Store the spacecraft's heliographic latitude.
        lat = stony_coord.lat.value
        lats.append(lat)

    # Return the lists of Carrington longitudes, distances, latitudes, and longitudes.
    return carr_lons, distances, lats, lons

def calculate_carrington_longitude_from_lon(date, lon):
    
    # Ensure that input is a NumPy array
    date = np.asarray(date)
    lon = np.asarray(lon)

    start_frame = c.HeliocentricInertial(obstime=date)
    end_frame = c.HeliographicStonyhurst(obstime=date)
    sun_center = SkyCoord(lon*u.deg, 0*u.deg, 1*u.AU, frame=start_frame)
    HGS_lon = sun_center.transform_to(end_frame)

    carr_lons = get_carrington_longitude(date)
    carr_lons = np.asarray(carr_lons) + HGS_lon.lon.value
    carr_lons = carr_lons % 360

    # Return the calculated Carrington longitude.
    return carr_lons

