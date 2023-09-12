"""Optionally define a custom strategy.

Needed only when the strategy is not yet implemented in Flower or because you want to
extend or modify the functionality of an existing strategy.
"""

from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple
from flwr.common.typing import FitIns, GetPropertiesIns, MetricsAggregationFn, NDArrays, Parameters, Scalar
from flwr.server.strategy import FedAvg
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common.logger import log
from flwr.server.strategy.fedavg import WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW

class FedAvgBestRTOptimizer(FedAvg):
    """Federated averaging strategy with Best RT Optimizer."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes, line-too-long
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        max_local_epochs: int = 5,
        batch_size: int = 32,
        fraction_samples: float = 1.0,
    ) -> None:
        super().__init__()

        if (
            min_fit_clients > min_available_clients
            or min_evaluate_clients > min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.max_local_epochs = max_local_epochs
        self.batch_size = batch_size
        self.fraction_samples = fraction_samples

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        
        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Create fit instructions for each client
        fit_ins = []

        ips_clients = []

        # Retrieve IPS of sampled clients
        for client in clients:
            config = {}
            propertiesRes = client.get_properties(GetPropertiesIns(config), None)
            ips = propertiesRes.properties["ips"]
            ips_clients.append((client, ips))

        # Find the maximum IPS among those of the selected clients
        max_ips = max(ips_clients, key=lambda x: x[1])[1]

        for client, ips in ips_clients:
            # Compute scaling factor
            scale_factor = ips / max_ips

            if(ips == max_ips):
                local_epochs = self.max_local_epochs
            else:
                local_epochs = max(1, int(self.max_local_epochs * scale_factor))

            config = {
                "local_epochs": local_epochs,
                "batch_size": self.batch_size,
                "fraction_samples": self.fraction_samples,
            }

            print(f"Client {client.cid} - IPS {ips} - Local epochs {local_epochs}")

            # Create fit instruction
            fit_ins.append((client, FitIns(parameters, config)))


        # Return client/config pairs
        return fit_ins
        