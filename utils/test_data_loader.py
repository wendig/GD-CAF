from torch.utils.data import DataLoader

from utils.dataset_one_picture_to_grid_europe import DatasetGridOverEurope


def get_test_dataset(dataset_path, past_look, future_look, B=16, fast_dev_run=False, cell_path='', cell_cutoff=100):
    europe_dataset = DatasetGridOverEurope(
        dataset_path=dataset_path,
        time_steps=past_look,
        future_look=future_look,
        fast_dev_run=fast_dev_run,
        cell_path=cell_path,
        cell_cutoff=cell_cutoff,
    )

    europe_dataset.count_data_rows()
    europe_dataset.load_grid()
    europe_dataset.normalize()

    return DataLoader(europe_dataset, batch_size=B, shuffle=False)
