import pandas as pd
import numpy as np

def inelastic_radial(spacecraft, cadence, COR=0):
    """
    Generate a new NumPy array with a simulated propagation of the spacecraft data
    with momentum conservation, no energy conservation. Only radial velocity taken into account

    Parameters:
    - input_data: 
            
            spacecraft:

            pd dataframe of a spacecraft's in-situ signature
            Spacecraft data from which to generate the simulation has to contain the following columns:
            'N': density
            'V': velocity
            'R': distance
            'CARR_LON_RAD': carrington longitude in radians

            cadence: the cadence with which the model runs

            COR: Coefficient of Resitution
                0: perfectly inelastic
                0<COR<1 : real inelastic
                1: perfectly elastic
    Returns:
    - sim: np.ndarray
    
        A simulated NumPy array containing all the steps in one long column
        size: ( n*(n+1)/2 ,  5 )
        columns: (N, V, R, CARR_LON_RAD, ITERATION)
        ITERATION denotes the number of steps as well as the number of data points


    Examples:
        sim = inelastic_propagation(solo, '1H')
    """

    import numpy as np
    import pandas
    
    #import numba
    #@numba.jit(nopython=True, parallel = True)

    spacecraft = spacecraft.resample(rule=cadence).median()
    L = pd.Timedelta(cadence).total_seconds() * 600 / 1.5e8 /10 # CHARACTERISTIC DISTANCE /10

    hours = pd.Timedelta(cadence).total_seconds() / 3600

    if 'Region' in spacecraft:
        input_data = spacecraft[['N', 'V', 'R', 'CARR_LON_RAD', 'Region']].to_numpy()
    else:
        input_data = spacecraft[['N', 'V', 'R', 'CARR_LON_RAD']].to_numpy()

    n = input_data.shape[0]

    # Pre-allocate array with NaN values for the entire structure
    sim = np.empty((n * (n + 1) // 2, 5))  # 5 columns for 'N', 'V', 'R', 'L', 'ITERATION'

    # Iterate over n steps for simulation
    for i in range(n):

        if i == 0:
            # Initial values for the first step
            sim[0, 4] = 0
            sim[0, 1] = input_data[0, 1]  # 'V' column
            sim[0, 0] = input_data[0, 0]  # 'N' column
            sim[0, 2] = input_data[0, 2]  # 'R' column
            sim[0, 3] = input_data[0, 3]# - 0.0096 * hours  # 'CARR_LON_RAD' column
        else:
            # Update values based on previous step and input data
            first = i * (i + 1) // 2
            last = first + i
            first_previous = i * (i - 1) // 2
            
            #ITERATION
            sim[first : last + 1, 2] = i


            #V
            sim[first : last + 1, 1] = sim[first_previous : first_previous + i + 1, 1]
            sim[last, 1] = input_data[i, 1]


            #N
            sim[first : last + 1, 0] = sim[first_previous : first_previous + i + 1, 0]
            sim[last, 0] = input_data[i, 0]


            #R
            sim[first : last + 1, 2] = sim[first_previous : first_previous + i + 1, 2] + \
                                               sim[first_previous : first_previous + i + 1, 1] / 1.4959787 / 10**8 * hours*3600.
            sim[last, 2] = input_data[i, 2]


            #CARR_LON_RAD
            sim[first : last + 1, 3] = sim[first_previous : first_previous + i + 1, 3] - 0.0096 * hours
            sim[last, 3] = input_data[i, 3]
            
            #REGION
            if 'Region' in spacecraft:
                sim[first : last + 1, 4] = sim[first_previous : first_previous + i + 1, 4]
                sim[last, 4] = input_data[i, 4]



            # Iterate over previous steps for momentum conservation
            for j in range(first, last + 1):
                delta_R = abs(sim[j, 2] - sim[:j, 2]) # IN AU
                delta_L = abs(sim[j, 3] - sim[:j, 3]) # IN RAD

                mask = (delta_R < L) & (delta_L * sim[j, 2] < L)  # Adjust the conditions as needed
                
                if np.any(mask):
                    
                    # p_b = u_b * m_b (SUM)
                    pastsum = np.sum(sim[0 : j][mask, 1] * sim[0 : j][mask, 0])

                    # v_a = p_b + p_a / m_a + m_b(SUM)

                    # v_a = (1+COR)p_b + p_a - u_a*m_b(SUM)*COR  / (m_a + m_b(SUM))
                    sim[j, 1] = (((COR+1.)*pastsum 
                                         + sim[j, 1] * sim[j, 0] 
                                         - sim[j, 1]*np.sum(sim[0 : j][mask, 0])*(COR)) /
                                        (sim[j, 0] + np.sum(sim[0 : j][mask, 0])))

                    # p_b = u_b * m_b (VECTOR)
                    past = sim[0 : j][mask, 1] * sim[0 : j][mask, 0]

                    # v_b = p_b + (1+COR)p_a - u_b(VECTOR)*m_a*COR / m_a + m_b(VECTOR)
                    sim[0 : j][mask, 1] = ((past 
                                                   + sim[j, 1] * sim[j, 0]*(COR+1.)
                                                   - sim[0 : j][mask, 1] * sim[j, 0] * COR) /
                                                              (sim[j, 0] + sim[0 : j][mask, 0]))

    output_data = pd.DataFrame({
        'CARR_LON_RAD': sim[:, 3],
        'R': sim[:, 2],
        'V': sim[:, 1],
        'N': sim[:, 0],
        'ITERATION': sim[:, 4]
    }, index=spacecraft.index[0] + sim[:, 4] * pd.Timedelta(cadence))

    return output_data

import pandas as pd


def ballistic(spacecraft, cadence):
    """
    Generate a new NumPy array with a simulated propagation of the spacecraft data
    Only radial velocity taken into account

    Parameters:
    - input_data: 
            
            spacecraft:

            pd dataframe of a spacecraft's in-situ signature
            Spacecraft data from which to generate the simulation has to contain the following columns:
            'N': density
            'V': velocity
            'R': distance
            'CARR_LON_RAD': carrington longitude in radians

            L: interaction radius in AU

    Returns:
    - sim: np.ndarray
    
        A simulated NumPy array containing all the steps in one long column
        size: ( n*(n+1)/2 ,  5 )
        columns: (N, V, R, CARR_LON_RAD, ITERATION)
        ITERATION denotes the number of steps as well as the number of data points


    Examples:
        sim = ballstic_propagation(solo, '1H')
    """

    import numpy as np
    import pandas
    
    #import numba
    #@numba.jit(nopython=True, parallel = True)

    spacecraft = spacecraft.resample(rule=cadence).first()
 
    if 'Region' in spacecraft:
        input_data = spacecraft[['N', 'V', 'R', 'CARR_LON_RAD', 'Region']].to_numpy()
    else:
        input_data = spacecraft[['N', 'V', 'R', 'CARR_LON_RAD']].to_numpy()

    hours = pd.Timedelta(cadence).total_seconds() / 3600

    n = input_data.shape[0]

    # Pre-allocate array with NaN values for the entire structure
    sim = np.empty((n * (n + 1) // 2, 5))  # 5 columns for 'N', 'V', 'R', 'L', 'ITERATION'

    # Iterate over n steps for simulation
    for i in range(n):

        if i == 0:
            # Initial values for the first step
            sim[0, 4] = 0
            sim[0, 1] = input_data[0, 1]  # 'V' column
            sim[0, 0] = input_data[0, 0]  # 'N' column
            sim[0, 2] = input_data[0, 2]  # 'R' column
            sim[0, 3] = input_data[0, 3] - 0.0096 * hours  # 'CARR_LON_RAD' column
        else:
            # Update values based on previous step and input data
            first = i * (i + 1) // 2
            last = first + i
            first_previous = i * (i - 1) // 2
            
            #ITERATION
            sim[first : last + 1, 2] = i


            #V
            sim[first : last + 1, 1] = sim[first_previous : first_previous + i + 1, 1]
            sim[last, 1] = input_data[i, 1]


            #N
            sim[first : last + 1, 0] = sim[first_previous : first_previous + i + 1, 0]
            sim[last, 0] = input_data[i, 0]


            #R
            sim[first : last + 1, 2] = sim[first_previous : first_previous + i + 1, 2] + \
                                               sim[first_previous : first_previous + i + 1, 1] / 1.4959787 / 10**8 * hours * 3600.
            sim[last, 2] = input_data[i, 2]


            #CARR_LON_RAD
            sim[first : last + 1, 3] = sim[first_previous : first_previous + i + 1, 3] - 0.0096 * hours
            sim[last, 3] = input_data[i, 3]
            
            #REGION
            if 'Region' in spacecraft:
                sim[first : last + 1, 4] = sim[first_previous : first_previous + i + 1, 4]
                sim[last, 4] = input_data[i, 4]


    output_data = pd.DataFrame({
        'CARR_LON_RAD': sim[:, 3],
        'R': sim[:, 2],
        'V': sim[:, 1],
        'N': sim[:, 0],
        'ITERATION': sim[:, 4]
    }, index=spacecraft.index[0] + sim[:, 4] * pd.Timedelta(cadence))

    return output_data


def ballistic_reverse(spacecraft, cadence):
    """
    Generate a new NumPy array with a simulated propagation of the spacecraft data
    IN REVERSE

    Only radial velocity taken into account

    Parameters:
    - input_data: 
            
            spacecraft:

            pd dataframe of a spacecraft's in-situ signature
            Spacecraft data from which to generate the simulation has to contain the following columns:
            'N': density
            'V': velocity
            'R': distance
            'CARR_LON_RAD': carrington longitude in radians
            
            cadence: the cadence with which the model runs
            
    Returns:
    - sim: np.ndarray
    
        A simulated NumPy array containing all the steps in one long column
        size: ( n*(n+1)/2 ,  5 )
        columns: (N, V, R, CARR_LON_RAD, ITERATION)
        ITERATION denotes the number of steps as well as the number of data points


    Examples:
        sim = ballstic_propagation(solo, '1H')
    """

    import numpy as np
    import pandas
    
    #import numba
    #@numba.jit(nopython=True, parallel = True)

    spacecraft = spacecraft.resample(rule=cadence).first()
 
    if 'Region' in spacecraft:
        input_data = spacecraft[['N', 'V', 'R', 'CARR_LON_RAD', 'Region']].to_numpy()
    else:
        input_data = spacecraft[['N', 'V', 'R', 'CARR_LON_RAD']].to_numpy()

    hours = pd.Timedelta(cadence).total_seconds() / 3600

    n = input_data.shape[0]

    # Pre-allocate array with NaN values for the entire structure
    sim = np.empty((n * (n + 1) // 2, 5))  # 5 columns for 'N', 'V', 'R', 'L', 'ITERATION'

    # Iterate over n steps for simulation
    for i in range(n):

        if i == 0:
            # Initial values for the first step
            sim[0, 4] = 0
            sim[0, 1] = input_data[0, 1]  # 'V' column
            sim[0, 0] = input_data[0, 0]  # 'N' column
            sim[0, 2] = input_data[0, 2]  # 'R' column
            sim[0, 3] = input_data[0, 3] + 0.0096 * hours  # 'CARR_LON_RAD' column
        else:
            # Update values based on previous step and input data
            first = i * (i + 1) // 2
            last = first + i
            first_previous = i * (i - 1) // 2
            
            #ITERATION
            sim[first : last + 1, 2] = i


            #V
            sim[first : last + 1, 1] = sim[first_previous : first_previous + i + 1, 1]
            sim[last, 1] = input_data[i, 1]


            #N
            sim[first : last + 1, 0] = sim[first_previous : first_previous + i + 1, 0]
            sim[last, 0] = input_data[i, 0]


            #R
            sim[first : last + 1, 2] = sim[first_previous : first_previous + i + 1, 2] - \
                                               sim[first_previous : first_previous + i + 1, 1] / 1.4959787 / 10**8 * hours * 3600.
            sim[last, 2] = input_data[i, 2]


            #CARR_LON_RAD
            sim[first : last + 1, 3] = sim[first_previous : first_previous + i + 1, 3] + 0.0096 * hours
            sim[last, 3] = input_data[i, 3]
            
            #REGION
            if 'Region' in spacecraft:
                sim[first : last + 1, 4] = sim[first_previous : first_previous + i + 1, 4]
                sim[last, 4] = input_data[i, 4]


    output_data = pd.DataFrame({
        'CARR_LON_RAD': sim[:, 3],
        'R': sim[:, 2],
        'V': sim[:, 1],
        'N': sim[:, 0],
        'ITERATION': sim[:, 4]
    }, index=spacecraft.index[0] - sim[:, 4] * pd.Timedelta(cadence))

    output_data = output_data[output_data['R']> 0.005]

    return output_data


def cut_from_sim(sim, spacecraft=None):
    """
    Extracts the values of 'V' from the sim DataFrame that correspond to the closest 'CARR_LON_RAD'
    and 'R' values in the spacecraft DataFrame.
    
    Parameters:
    - sim: pd.DataFrame
        Simulation data with columns 'CARR_LON_RAD', 'R', and 'V', 'Region' optional.
    - spacecraft: pd.DataFrame, optional
        Spacecraft data with columns 'CARR_LON_RAD' and 'R'. If None, default values are 1AU.
    
    Returns:
    - result: pd.DataFrame
        DataFrame with the closest values of 'V' and 'Region' from sim based on 'CARR_LON_RAD' and 'R' in spacecraft.
    """
    
    if spacecraft is None:
        spacecraft = pd.DataFrame({
            'CARR_LON_RAD': np.linspace(0, 2*np.pi, 720),
            'R': np.ones(720)  # 1AU
        })
    
    sim = sim.reset_index(drop=True)

    # Create an empty array to store the closest 'V' values
    closest_V = np.empty(len(spacecraft))
    Region = np.zeros(len(spacecraft))

    # Iterate over each row in the spacecraft DataFrame
    for i, row in spacecraft.iterrows():
        # Calculate the Euclidean distance between spacecraft values and sim values for 'CARR_LON_RAD' and 'R'
        distances = np.sqrt((sim['CARR_LON_RAD'] - row['CARR_LON_RAD'])**2 +
                            (sim['R'] - row['R'])**2)
        
        # Find the index of the minimum distance
        closest_idx = distances.idxmin()
        # Store the corresponding 'V' value and Region
        closest_V[i] = sim.loc[closest_idx, 'V']
        if 'Region' in spacecraft:
            Region[i] = sim.loc[closest_idx, 'Region']
        
        if abs(sim.loc[closest_idx, 'CARR_LON_RAD'] - row['CARR_LON_RAD']) > 1 / 180 * np.pi:
            
            closest_V[i] = np.nan
            if 'Region' in spacecraft:
                Region[i] = 0
        
    
    # Create a new DataFrame with the results
    result = pd.DataFrame({
        'CARR_LON_RAD': spacecraft['CARR_LON_RAD'],
        'R': spacecraft['R'],
        'V': closest_V,
        'Region': Region
    })
    
    return result

def inelastic_radial_high_res(spacecraft, cadence, COR=0, res_factor=2):
    """
    Generate a new NumPy array with a simulated propagation of the spacecraft data
    with momentum conservation, no energy conservation. Only radial velocity taken into account

    Parameters:
    - input_data: 
            
            spacecraft:

            pd dataframe of a spacecraft's in-situ signature
            Spacecraft data from which to generate the simulation has to contain the following columns:
            'N': density
            'V': velocity
            'R': distance
            'CARR_LON_RAD': carrington longitude in radians

            cadence: the cadence with which the model runs

            COR: Coefficient of Resitution
                0: perfectly inelastic
                0<COR<1 : real inelastic
                1: perfectly elastic
    Returns:
    - sim: np.ndarray
    
        A simulated NumPy array containing all the steps in one long column
        size: ( n*(n+1)/2 ,  5 )
        columns: (N, V, R, CARR_LON_RAD, ITERATION)
        ITERATION denotes the number of steps as well as the number of data points


    Examples:
        sim = inelastic_propagation(solo, '1H')
    """

    import numpy as np
    import pandas
    
    #import numba
    #@numba.jit(nopython=True, parallel = True)

    spacecraft = spacecraft.resample(rule=cadence).median()
    L = pd.Timedelta(cadence).total_seconds() * 600 / 1.5e8 /10 / res_factor # CHARACTERISTIC DISTANCE /10

    hours = pd.Timedelta(cadence).total_seconds() / 3600 / res_factor

    if 'Region' in spacecraft:
        input_data = spacecraft[['N', 'V', 'R', 'CARR_LON_RAD', 'Region']].to_numpy()
    else:
        input_data = spacecraft[['N', 'V', 'R', 'CARR_LON_RAD']].to_numpy()

    n = input_data.shape[0] * res_factor

    # Pre-allocate array with NaN values for the entire structure
    sim = np.empty((n * (n + 1) // 2, 5))  # 5 columns for 'N', 'V', 'R', 'L', 'ITERATION'

    # Iterate over n steps for simulation
    for i in range(n):

        if i == 0:
            # Initial values for the first step
            sim[0, 4] = 0
            sim[0, 1] = input_data[0, 1]  # 'V' column
            sim[0, 0] = input_data[0, 0]  # 'N' column
            sim[0, 2] = input_data[0, 2]  # 'R' column
            sim[0, 3] = input_data[0, 3]# - 0.0096 * hours  # 'CARR_LON_RAD' column
        else:
            # Update values based on previous step and input data
            first = i * (i + 1) // 2
            last = first + i
            first_previous = i * (i - 1) // 2
            
            #ITERATION
            sim[first : last + 1, 2] = i


            #V
            sim[first : last + 1, 1] = sim[first_previous : first_previous + i + 1, 1]
            

            #N
            sim[first : last + 1, 0] = sim[first_previous : first_previous + i + 1, 0]



            #R
            sim[first : last + 1, 2] = sim[first_previous : first_previous + i + 1, 2] + \
                                               sim[first_previous : first_previous + i + 1, 1] / 1.4959787 / 10**8 * hours*3600.


            #CARR_LON_RAD
            sim[first : last + 1, 3] = sim[first_previous : first_previous + i + 1, 3] - 0.0096 * hours
            
            #REGION
            if 'Region' in spacecraft:
                sim[first : last + 1, 4] = sim[first_previous : first_previous + i + 1, 4]



            #UPDATES

            if i % res_factor == 0:
                #print(i, i%res_factor, np.int16(i/res_factor))
                sim[last, 1] = input_data[np.int16(i / res_factor), 1]
                sim[last, 0] = input_data[np.int16(i/ res_factor), 0]
                sim[last, 2] = input_data[np.int16(i/ res_factor), 2]
                if 'Region' in spacecraft:
                    sim[last, 4] = input_data[np.int16(i/ res_factor), 4]

            # Iterate over previous steps for momentum conservation
            for j in range(first, last + 1):
                delta_R = abs(sim[j, 2] - sim[:j, 2]) # IN AU
                delta_L = abs(sim[j, 3] - sim[:j, 3]) # IN RAD

                mask = (delta_R < L) & (delta_L * sim[j, 2] < L)  # Adjust the conditions as needed
                
                if np.any(mask):
                    
                    # p_b = u_b * m_b (SUM)
                    pastsum = np.sum(sim[0 : j][mask, 1] * sim[0 : j][mask, 0])

                    # v_a = p_b + p_a / m_a + m_b(SUM)

                    # v_a = (1+COR)p_b + p_a - u_a*m_b(SUM)*COR  / (m_a + m_b(SUM))
                    sim[j, 1] = (((COR+1.)*pastsum 
                                         + sim[j, 1] * sim[j, 0] 
                                         - sim[j, 1]*np.sum(sim[0 : j][mask, 0])*(COR)) /
                                        (sim[j, 0] + np.sum(sim[0 : j][mask, 0])))

                    # p_b = u_b * m_b (VECTOR)
                    past = sim[0 : j][mask, 1] * sim[0 : j][mask, 0]

                    # v_b = p_b + (1+COR)p_a - u_b(VECTOR)*m_a*COR / m_a + m_b(VECTOR)
                    sim[0 : j][mask, 1] = ((past 
                                                   + sim[j, 1] * sim[j, 0]*(COR+1.)
                                                   - sim[0 : j][mask, 1] * sim[j, 0] * COR) /
                                                              (sim[j, 0] + sim[0 : j][mask, 0]))

    output_data = pd.DataFrame({
        'CARR_LON_RAD': sim[:, 3],
        'R': sim[:, 2],
        'V': sim[:, 1],
        'N': sim[:, 0],
        'ITERATION': sim[:, 4]
    }, index=spacecraft.index[0] + sim[:, 4] * pd.Timedelta(cadence))

    return output_data
