from typing import Any

import ray


@ray.remote
class Learner:
    def start(self) -> None:
        while True:
            batch = self.fetch_next_batch()
            self.optimize(batch)
            self.publish_weights()

    def fetch_next_batch(self) -> Any:
        pass

    def optimize(self, batch: Any) -> None:
        pass

    def publish_weights(self) -> None:
        pass


@ray.remote
class Actor:
    pass


def main() -> None:
    with ray.init():
        learner = Learner.remote()
        _actor = Actor.remote()
        ray.get(learner.start.remote())


if __name__ == "__main__":
    main()
