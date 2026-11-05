"""Install replacement JAXtari sprites and verify the supported games render."""

import jax
import jax.numpy as jnp
import jaxtari
from jaxtari.install_sprites import download_and_extract


def download_assets() -> None:
    download_and_extract(accepted_ownership=False)
    for game in ("breakout", "spaceinvaders"):
        env = jaxtari.make(game)
        _, state = env.reset(jax.random.key(0))  # native observation/state leaves: (...)
        frame = env.render(state)  # (210, 160, 3)
        assert frame.shape == (210, 160, 3) and frame.dtype == jnp.uint8, game


if __name__ == "__main__":
    download_assets()
