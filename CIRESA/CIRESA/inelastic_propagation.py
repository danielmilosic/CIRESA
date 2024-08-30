def inelastic_propagation(spacecraft, L):
    """
    Generate a new NumPy array with a simulated propagation of the spacecraft data
    with momentum conservation, no energy conservation.

    Parameters:
    - input_data: pd dataframe of a spacecraft's in-situ signature
        Spacecraft data from which to generate the simulation has to contain the following columns:
            'N': density
            'V': velocity
            'R': distance
            'CARR_LON_RAD': carrington longitude in radians

    Returns:
    - sim_column: np.ndarray
        A simulated NumPy array containing all the steps

    Examples:
        sim = inelastic_propagation(solo, 0.01)
    """

    import numpy as np
    import numba
    import pandas as pd
    @numba.jit(nopython=True, parallel = True)

    input_data = spacecraft[['N','V', 'R', 'CARR_LON_RAD']].to_numpy()

    n = input_data.shape[0]

    # Pre-allocate array with NaN values for the entire structure
    sim_column = np.empty((n * (n + 1) // 2, 5))  # 5 columns for 'N', 'V', 'R', 'L', 'ITERATION'

    # Iterate over n steps for simulation
    for i in range(n):
        #start_time = time.time()  # Record the start time for performance tracking

        if i == 0:
            # Initial values for the first step
            sim_column[0, 4] = 0
            sim_column[0, 1] = input_data[0, 1]  # 'V_BULK' column
            sim_column[0, 0] = input_data[0, 0]  # 'N' column
            sim_column[0, 2] = input_data[0, 2]  # 'SUN_DIST' column
            sim_column[0, 3] = input_data[0, 3] - 0.0096/2  # 'CARR_LON_RAD' column
        else:
            # Update values based on previous step and input data
            first = i * (i + 1) // 2
            last = first + i
            first_previous = i * (i - 1) // 2

            sim_column[first : last + 1, 2] = i

            sim_column[first : last + 1, 1] = sim_column[first_previous : first_previous + i + 1, 1]
            sim_column[last, 1] = input_data[i, 1]

            sim_column[first : last + 1, 0] = sim_column[first_previous : first_previous + i + 1, 0]
            sim_column[last, 0] = input_data[i, 0]

            sim_column[first : last + 1, 2] = sim_column[first_previous : first_previous + i + 1, 2] + \
                                               sim_column[first_previous : first_previous + i + 1, 1] / 1.4959787 / 10**8 * 1800.
            sim_column[last, 2] = input_data[i, 2]

            sim_column[first : last + 1, 3] = sim_column[first_previous : first_previous + i + 1, 3] - 0.0096/2
            sim_column[last, 3] = input_data[i, 3]

            # Iterate over previous steps for momentum conservation
            for j in range(first, last + 1):
                delta_R = sim_column[j, 2] - sim_column[:j, 2]
                delta_L = sim_column[j, 3] - sim_column[:j, 3]

                mask = (delta_R > L) & (delta_L * sim_column[j, 2] > L)  # Adjust the conditions as needed

                if np.any(mask):

                    pastsum = np.sum(sim_column[0 : j][mask, 1] * sim_column[0 : j][mask, 0])
                    sim_column[j, 1] = ((pastsum 
                                         + sim_column[j, 1] * sim_column[j, 0]) /
                                        (sim_column[j, 0] + np.sum(sim_column[0 : j][mask, 0])))
                    sim_column[j, 5] = 1

                    past = sim_column[0 : j][mask, 1] * sim_column[0 : j][mask, 0]
                    sim_column[0 : j][mask, 1] = ((past 
                                                   + sim_column[j, 1] * sim_column[j, 0]) /
                                                              (sim_column[j, 0] + sim_column[0 : j][mask, 0]))

    return sim_column