import flwr as fl

class TinyFLStrategy(fl.server.strategy.FedAvg):

    def __init__(self):
        super().__init__(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=1,
            min_evaluate_clients=1,
            min_available_clients=1,
        )


def start_server():

    strategy = TinyFLStrategy()

    print("🚀 Federated Server Started with SINGLE-CLIENT STRATEGY")

    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )


if __name__ == "__main__":
    start_server()
