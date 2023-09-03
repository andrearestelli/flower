"""Create and connect the building blocks for your experiments; start the simulation.

It includes processioning the dataset, instantiate strategy, specify how the global
model is going to be evaluated, etc. At the end, this script saves the results.
"""
# these are the basic packages you'll need here
# feel free to remove some if aren't needed
from utils import plot_dloss_from_history
from utils import plot_metric_from_history
from utils import save_results_as_pickle
from client import gen_client_fn
import hydra
import flwr as fl
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the baseline.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """
    # 1. Print parsed config
    print(OmegaConf.to_yaml(cfg))

    # 2. Prepare your dataset
    # here you should call a function in datasets.py that returns whatever is needed to:
    # (1) ensure the server can access the dataset used to evaluate your model after
    # aggregation
    # (2) tell each client what dataset partitions they should use (e.g. a this could
    # be a location in the file system, a list of dataloader, a list of ids to extract
    # from a dataset, it's up to you)

    # 3. Define your clients
    # Define a function that returns another function that will be used during
    # simulation to instantiate each individual client
    client_fn = gen_client_fn((cfg.epochs_min, cfg.epochs_max), (cfg.fraction_samples_min, cfg.fraction_samples_max), (cfg.batch_size_min, cfg.batch_size_max), cfg.num_clients)

    # 4. Define your strategy
    # pass all relevant argument (including the global dataset used after aggregation,
    # if needed by your method.)
    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.08,
        fraction_evaluate=0.08,
        min_available_clients=cfg.num_clients,
    )

    # Initialize ray_init_args
    ray_init_args = {
        "ignore_reinit_error": True,
        "include_dashboard": False,
    }

    # 5. Start Simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        client_resources={"num_cpus": 1},
        config=fl.server.ServerConfig(cfg.num_rounds),
        strategy= strategy,
        ray_init_args=ray_init_args,
    )

    # 6. Save your results
    # Experiment completed. Now we save the results and
    # generate plots using the `history`
    print("................")
    print(history)

    # Hydra automatically creates an output directory
    # Let's retrieve it and save some results there
    save_path = HydraConfig.get().runtime.output_dir

    # save results as a Python pickle using a file_path
    # the directory created by Hydra for each run
    save_results_as_pickle(history, file_path=save_path, extra_results={})

    # plot results and include them in the readme
    strategy_name = strategy.__class__.__name__
    file_suffix: str = (
        f"_{strategy_name}"
        f"_C={cfg.num_clients}"
        f"_MinB={cfg.batch_size_min}"
        f"_MaxB={cfg.batch_size_max}"
        f"_MinE={cfg.epochs_min}"
        f"_MaxE={cfg.epochs_max}"
        f"_R={cfg.num_rounds}"
    )

    plot_dloss_from_history(
        history,
        save_path,
        (file_suffix),
    )

if __name__ == "__main__":
    main()