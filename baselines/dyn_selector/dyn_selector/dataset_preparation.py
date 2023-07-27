"""Handle the dataset partitioning and (optionally) complex downloads.

Please add here all the necessary logic to either download, uncompress, pre/post-process
your dataset (or all of the above). If the desired way of running your baseline is to
first download the dataset and partition it and then run the experiments, please
uncomment the lines below and tell us in the README.md (see the "Running the Experiment"
block) that this file should be executed first.
"""
import os
import hydra
import tensorflow as tf
import numpy as np
from hydra.core.hydra_config import HydraConfig
# from hydra.utils import call, instantiate
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="conf", config_name="base", version_base=None)
def download_and_preprocess(cfg: DictConfig) -> None:
    """Does everything needed to get the dataset.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """

    ## 1. print parsed config
    print(OmegaConf.to_yaml(cfg))

    dataset = cfg.dataset.dataset
    total_clients = cfg.num_clients
    num_classes = cfg.dataset.num_classes

    folder = dataset

    # if the folder exists it is deleted and the ds partitions are re-created
    # if the folder does not exist, firstly the folder is created
    # and then the ds partitions are generated
    exist = os.path.exists(folder)
    if not exist:
        os.makedirs(folder)
    
    # Load the MNIST dataset
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()

    # Shuffle the dataset using a given seed for reproducibility
    seed = cfg.seed if cfg.seed is not None else 42
    np.random.seed(seed)
    indices = np.arange(len(x_train))
    np.random.shuffle(indices)
    x_train_shuffled = x_train[indices]
    y_train_shuffled = y_train[indices]

    # Create partition for each client
    partition_size = int(len(x_train_shuffled) / total_clients)

    for cid in range(total_clients):
        idx_from, idx_to = cid * partition_size, (cid + 1) * partition_size
        x_train_cid = x_train_shuffled[idx_from:idx_to] / 255.0
        y_train_cid = y_train_shuffled[idx_from:idx_to]

        ds = tf.data.Dataset.from_tensor_slices((x_train_cid, y_train_cid))
        ds = ds.shuffle(buffer_size=4096)

        # save the dataset partition
        tf.data.experimental.save(ds, os.path.join(folder, str(cid)), compression=None)

    list_of_narrays = []
    for sampled_client in range(0, total_clients):
        loaded_ds = tf.data.experimental.load(
            path=os.path.join(folder, str(sampled_client)), element_spec=None, compression=None, reader_func=None
        )

        print("[Client " + str(sampled_client) + "]")
        print("Cardinality: ", tf.data.experimental.cardinality(loaded_ds).numpy())

        def count_class(counts, batch, num_classes=num_classes):
            _, labels = batch
            for i in range(num_classes):
                cc = tf.cast(labels == i, tf.int32)
                counts[i] += tf.reduce_sum(cc)
            return counts

        initial_state = dict((i, 0) for i in range(num_classes))
        counts = loaded_ds.reduce(initial_state=initial_state, reduce_func=count_class)

        # print([(k, v.numpy()) for k, v in counts.items()])
        new_dict = {k: v.numpy() for k, v in counts.items()}
        # print(new_dict)
        res = np.array([item for item in new_dict.values()])
        # print(res)
        list_of_narrays.append(res)

    distribution = np.stack(list_of_narrays)
    print(distribution)
    # saving the distribution of per-label examples in a numpy file
    # this can be useful also to draw charts about the label distrib.
    path = os.path.join(folder, "distribution_train.npy")
    np.save(path, distribution)

if __name__ == "__main__":

    download_and_preprocess()
