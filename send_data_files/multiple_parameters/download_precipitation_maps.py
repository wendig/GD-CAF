"""
    File for downloading era5 data about whole europe

    download it in small chunks (one month) but big area

    => Size of the grid can be a variable
"""

import cdsapi
import os

# Europe and the extended regions border
DEFAULT_X_0, DEFAULT_X_1 = -31, 100  # W E
DEFAULT_Y_0, DEFAULT_Y_1 = 15, 82  # S N

year_from = 2016
year_to = 2022

each_month = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
folder_name = 'db-whole-multi-param'


def download_month(year, month, multi_parameters=True):
    """
        Call ERA5 API to get one month of data from the whole map
    """
    c = cdsapi.Client()

    name = f'{folder_name}\\whole-europe_{year}_{month}_{DEFAULT_Y_1}_{DEFAULT_X_0}_{DEFAULT_Y_0}_{DEFAULT_X_1}.nc'

    variable = ['10m_u_component_of_wind', '10m_v_component_of_wind', '2m_dewpoint_temperature', '2m_temperature',
                'surface_pressure', 'total_precipitation'] if multi_parameters else ['total_precipitation']

    if needToDownload(name):
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'variable': variable,
                'year': year,
                'month': month,
                'day': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15',
                        '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30',
                        '31'],
                'time': ['00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00', '08:00', '09:00',
                         '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00', '17:00', '18:00', '19:00',
                         '20:00', '21:00', '22:00', '23:00'],
                'area': [DEFAULT_Y_1, DEFAULT_X_0, DEFAULT_Y_0, DEFAULT_X_1],  # n w s e
                'format': 'netcdf',
            },
            name)


def needToDownload(looking_for):
    """
        Test if file is already downloaded

    :param looking_for: file name that we are looking for
    :return: true if file is not in the folder, thus we need to download it
    """
    filenames = next(os.walk(folder_name), (None, None, []))[2]  # [] if no file
    looking_for = looking_for.split('\\')[-1]
    print(filenames)
    print('looking for: ' + looking_for)

    for file in filenames:
        if file == looking_for:
            print('found ' + file)
            return False
    return True


# Download every selected year, and month
for y in range(year_from, year_to + 1):
    print('Downloading year: {}'.format(y))

    for m in each_month:
        print('\tMonth: {}'.format(m))
        download_month(y, m)
