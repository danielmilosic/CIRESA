def calculate_carrington_longitude(date):
    # Import the necessary module for handling astronomical time.
    from astropy.time import Time

    # Set the reference date for the start of Carrington rotations (November 9, 1853, 16:00:00).
    reference_date = Time('1853-11-09 16:00:00')
    
    # Calculate the number of days since the reference date.
    days_since_reference = (Time(date) - reference_date).jd

    # Calculate the current Carrington rotation number.
    carrington_rotation = 1 + (days_since_reference / 27.2753)
    
    # Calculate the Carrington longitude by converting the fractional part of the rotation.
    carrington_longitude = (1 + int(carrington_rotation) - carrington_rotation) * 360

    # Return the calculated Carrington longitude.
    return carrington_longitude

def get_coordinates(data, spcrft):
    # Import the necessary function for retrieving spacecraft coordinates.
    from sunpy.coordinates import get_horizons_coord
    from CIRESA.get_coordinates import calculate_carrington_longitude
    import pandas as pd

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
        carrington_longitude = calculate_carrington_longitude(data.index[i])

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
