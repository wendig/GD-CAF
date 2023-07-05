"""
Europe is downloaded as one picture, this dataset assign grids to it

"""

import os
import xarray as xr
import numpy as np
from torch.utils.data import Dataset


class DatasetGridOverEurope(Dataset):
    def __init__(self, dataset_path, time_steps, future_look, delimiter='/', fast_dev_run=False, cell_path='', cell_cutoff=100):
        self.tile_positions = None
        self.pics_width = None  # Total x length of input data (whole europe)
        self.pics_height = None
        self.grid_data = None
        self.total_length = None
        self.cell_shapes = None
        self.graph_size = None

        self.dataset_path = dataset_path  # path to the netCDF files
        self.past_steps = time_steps  # Number of past observations
        self.future_look = future_look  # Number of future observations
        self.delimiter = delimiter
        self.cell_path = cell_path  # Randomly placed cells, position file
        self.cell_cutoff = cell_cutoff  # At this number stop reading in more cells
        self.fast_dev_run = fast_dev_run  # When testing do not load every year
        self.vocal = True

        # Read files
        self.count_data_rows()

    def count_data_rows(self):
        """
            Read .nc datafiles to count the total number of rows
        :return:
        """
        filenames = next(os.walk(self.dataset_path), (None, None, []))[2]  # [] if no file
        filenames = [filename for filename in filenames if filename.endswith('.nc')]
        n = 0

        if filenames:
            nc = xr.open_dataset(self.dataset_path + self.delimiter + filenames[0])
            self.pics_height = nc.variables['tp'][:].shape[1]
            self.pics_width = nc.variables['tp'][:].shape[2]

        # Count number of pictures
        for file in filenames:
            nc = xr.open_dataset(self.dataset_path + self.delimiter + file)
            n += nc.variables['tp'][:].shape[0]

            # Fast stop for
            month = file.split('_')[2]
            if month == '01' and self.fast_dev_run:
                break

        self.total_length = n

        self.load_random_tile()

        print('Total number of images: {}'.format(n))
        print('Nr. Images/grids: {}'.format(n / self.graph_size))

    def load_random_tile(self):
        """
            Load cell position file
        """
        self.tile_positions = []

        with open(self.cell_path) as file:
            lines = [line.rstrip() for line in file]

            for line in lines:
                # Example: 178 91 80 40 : x, y , edge_x, edge_y
                content = line.split(' ')

                square = {
                    'x': int(content[0]),
                    'y': int(content[1]),
                    'edge_x': int(content[2]),
                    'edge_y': int(content[3])
                }
                self.tile_positions.append(square)

                if len(self.tile_positions) == self.cell_cutoff:
                    print(f'Cut off at {self.cell_cutoff}')
                    break

            # Set number of grids
            self.graph_size = len(self.tile_positions)
            # Set cell shapes
            self.cell_shapes = (self.tile_positions[0]['edge_y'], self.tile_positions[0]['edge_x'])


    def iterate_random_cells(self, nc, grid_data, from_counter):
        for i in range(len(self.tile_positions)):
            raw_data = nc.variables['tp'][:]
            # Get random tile position
            square = self.tile_positions[i]
            # Location indexing
            grid_data[from_counter: (from_counter + raw_data.shape[0]), i, :, :] = \
                raw_data[:, square['y']:(square['y'] + square['edge_y']), square['x']:(square['x'] + square['edge_x'])]

        return grid_data

    def load_grid(self):
        """
            Load all .nc file, get disjoint regions and convert it to a numpy array
        """
        # Read files
        filenames = next(os.walk(self.dataset_path), (None, None, []))[2]  # [] if no file
        filenames = [filename for filename in filenames if filename.endswith('.nc')]
        # Keep track
        from_counter = 0  # Starting index of the current batch
        grid_y, grid_x = self.cell_shapes  # Grid cell height and width
        # Round
        self.pics_width = self.pics_width - self.pics_width % 4
        self.pics_height = self.pics_height - self.pics_height % 4

        grid_data = np.zeros((self.total_length, self.graph_size, grid_y, grid_x), dtype='float32')
        print(f'grid data: {grid_data.shape}')
        for file in filenames:
            # Extract parameters from file names
            month = file.split('_')[2]

            nc = xr.open_dataset(self.dataset_path + self.delimiter + file)
            number_of_images = nc.variables['tp'][:].shape[0]

            print(f'Loading: {file}')

            if self.vocal:
                print('TP parameter: {}'.format(nc.variables['tp'][:].shape))
                self.vocal = False

            grid_data = self.iterate_random_cells(nc, grid_data, from_counter)
            # Time pointer
            from_counter += number_of_images
            # Fast stop
            if month == '01' and self.fast_dev_run:
                break

        self.grid_data = grid_data

    def __getitem__(self, idx):
        """
        :return: observation and label
        """
        x = self.grid_data[idx: idx + self.past_steps, :, :]
        y = self.grid_data[idx + self.past_steps + self.future_look - 1]
        return x, y

    def __len__(self):
        return self.grid_data.shape[0] - self.past_steps - self.future_look + 1

    def normalize(self, local=False):
        print(f'NORMALIZE: {np.amin(self.grid_data)} {np.amax(self.grid_data)}')
        print(self.grid_data.shape)
        self.grid_data = self._normalize(self.grid_data, local)

    @staticmethod
    def _normalize(input_data, local=False):
        """
            Min-max normalization

            x = (x - min) / max
        """
        if local:
            min_vals, max_vals = np.amin(input_data), np.amax(input_data)
        else:
            min_vals, max_vals = 0, 0.03651024401187897    # Overall max precipitation value

        print(f'Normalization {input_data.shape}: x = (x - {min_vals}) / ({max_vals} - {min_vals})')

        return (input_data - min_vals) / (max_vals - min_vals)
