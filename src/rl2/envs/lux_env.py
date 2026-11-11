"""Lux S3 team agents with joint ship commands and memory across matches."""

from typing import Annotated, Any, Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.custom_batching import custom_vmap
from luxai_s3.env import LuxAIS3Env
from luxai_s3.params import EnvParams, env_params_ranges
from luxai_s3.state import EnvObs, EnvState, MapTile, UnitState
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from rl2.actions import HierarchicalMultiDiscreteSpace
from rl2.envs.multi_agent import AgentObservation, BaseMultiAgentEnv, MultiAgentStep


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class LuxConfig:
    id: Literal["luxai_s3"] = "luxai_s3"
    # Square noise grid; leave room for sap offsets in native int16 coordinates.
    map_size: Annotated[int, Field(ge=8, le=32760, multiple_of=8, strict=True)] = 24
    max_units: Annotated[int, Field(ge=1, le=16, strict=True)] = 16
    max_steps_in_match: Annotated[int, Field(ge=2, lt=2**31, strict=True)] = 100
    match_count: Annotated[int, Field(ge=1, le=5, strict=True)] = 5
    randomize_params: bool = True

    def __post_init__(self) -> None:
        if (self.max_steps_in_match + 1) * self.match_count >= 2**31:
            raise ValueError("Game horizon must fit the native int32 counter")


class LuxState(NamedTuple):
    game: EnvState  # Array leaves: (...), full simulator state, never policy input.
    params: EnvParams  # Array leaves: (), sampled once per game.


class _RenderTile(NamedTuple):
    left: int
    top: int
    width: int
    height: int


class LuxTeamObservation[Array](NamedTuple):
    """One team's visible ships and public scores; optional leading axes S.

    units: (*S, U, 2) positions and (*S, U) energy in native units.
    units_mask: (*S, U) visible, existing ship slots.
    points: (*S,) current match points.
    wins: (*S,) cumulative match wins in the current game.
    """

    units: UnitState  # position: (*S, U, 2); energy: (*S, U)
    units_mask: Array  # (*S, U) bool
    points: Array  # (*S,) int32
    wins: Array  # (*S,) int32


class LuxWorldObservation[Array](NamedTuple):
    """Visible world and progress counters; optional leading axes S.

    sensor_mask: (*S, W, H) visible tiles.
    map_features: (*S, W, H) energy and tile-type leaves, with absolute coordinates.
    relic_nodes: (*S, R, 2) visible relic positions.
    relic_nodes_mask: (*S, R) visible relic slots.
    steps: (*S,) game step counter.
    match_steps: (*S,) match step counter.
    """

    sensor_mask: Array  # (*S, W, H) bool
    map_features: MapTile  # leaves: (*S, W, H), native dtypes
    relic_nodes: Array  # (*S, R, 2) int16
    relic_nodes_mask: Array  # (*S, R) bool
    steps: Array  # (*S,) int32
    match_steps: Array  # (*S,) int32


class LuxObservation[Array](NamedTuple):
    """Observer-relative teams and visible world; optional leading axes S.

    self: (*S, ...) own team leaves.
    opponent: (*S, ...) opponent leaves masked by the observer's visibility.
    world: (*S, ...) visible world and progress-counter leaves.
    public_params: (*S, 4) move cost, sap cost, sap range, sensor range, in native units.
    Array is jax.Array for observations or ShapeDtypeStruct for the observation space.
    """

    self: LuxTeamObservation[Array]  # leaves: (*S, ...)
    opponent: LuxTeamObservation[Array]  # leaves: (*S, ...)
    world: LuxWorldObservation[Array]  # leaves: (*S, ...)
    public_params: Array  # (*S, 4), native parameter dtype


class LuxEnv(BaseMultiAgentEnv[LuxState, LuxObservation[jax.Array]]):
    """Two agents; each chooses U categorical ship commands from A=230 choices.

    Observations split native team fields into self/opponent, preserving native
    dtypes and a leading observing-team axis. World fields retain the observer's mask.
    """

    @property
    def supports_selfplay(self) -> bool:
        return True

    def __init__(self, config: LuxConfig) -> None:
        self.config = config
        self.num_agents = 2
        self.params = EnvParams(
            map_width=config.map_size,
            map_height=config.map_size,
            max_units=config.max_units,
            max_steps_in_match=config.max_steps_in_match,
            match_count_per_episode=config.match_count,
        )
        self.env = LuxAIS3Env(auto_reset=False, fixed_env_params=self.params)
        self._autoreset = custom_vmap(self._autoreset)
        self._autoreset.def_vmap(self._autoreset_batch)
        self.sap_range = max(env_params_ranges["unit_sap_range"])
        self.action_space = HierarchicalMultiDiscreteSpace(
            nvec=(5 + (2 * self.sap_range + 1) ** 2,) * config.max_units,
            group_sizes=(1, 4, (2 * self.sap_range + 1) ** 2),  # Stay, move direction, sap target.
        )
        offsets = jnp.arange((2 * self.sap_range + 1) ** 2, dtype=jnp.int16)  # (A-5,)
        self.sap_offsets = (
            jnp.stack((offsets // (2 * self.sap_range + 1), offsets % (2 * self.sap_range + 1)), axis=-1)
            - self.sap_range
        )  # (A-5, 2), dx then dy

    def _decode_action(self, action: jax.Array) -> jax.Array:  # (N, U) -> (N, U, 3)
        sap = action >= 5  # (N, U)
        offsets = self.sap_offsets[jnp.maximum(action - 5, 0)]  # (N, U, 2)
        return jnp.concatenate(
            (jnp.where(sap, 5, action)[..., None], jnp.where(sap[..., None], offsets, 0)), axis=-1
        ).astype(jnp.int16)  # (N, U, 3)

    @property
    def observation_space(self) -> LuxObservation[jax.ShapeDtypeStruct]:
        """Return (...) shape/dtype leaves for one team's observation, excluding masks."""
        observation, _ = jax.eval_shape(self.reset, jax.random.key(0))
        return jax.tree.map(
            lambda leaf: jax.ShapeDtypeStruct(leaf.shape[1:], leaf.dtype), observation.observation
        )  # leaves: (...), without the observing-team axis

    def _observation(
        self,
        observation: dict[str, EnvObs],  # per-team leaves: (...)
        state: LuxState,  # leaves: (...)
    ) -> AgentObservation[LuxObservation[jax.Array]]:
        """Return (N, ...) native observation leaves and (N, U, A) action masks."""
        params, game = state.params, state.game
        public_params = jnp.stack(
            (params.unit_move_cost, params.unit_sap_cost, params.unit_sap_range, params.unit_sensor_range)
        )  # (4,)

        def team_observation(obs: EnvObs, team: int) -> LuxTeamObservation[jax.Array]:  # obs leaves: (...)
            return LuxTeamObservation(
                units=jax.tree.map(lambda leaf: leaf[team], obs.units),  # position: (U, 2); energy: (U,)
                units_mask=obs.units_mask[team],  # (U,)
                points=obs.team_points[team],  # ()
                wins=obs.team_wins[team],  # ()
            )

        views: list[LuxObservation[jax.Array]] = []  # each view has (...) leaves
        for team in range(2):
            obs = observation[f"player_{team}"]  # leaves: (...), this observer's masked view
            views.append(
                LuxObservation(
                    self=team_observation(obs, team),  # leaves: (...)
                    opponent=team_observation(obs, 1 - team),  # leaves: (...), same observer's visibility
                    world=LuxWorldObservation(
                        sensor_mask=obs.sensor_mask,  # (W, H)
                        map_features=obs.map_features,  # leaves: (W, H)
                        relic_nodes=obs.relic_nodes,  # (R, 2)
                        relic_nodes_mask=obs.relic_nodes_mask,  # (R,)
                        steps=obs.steps,  # ()
                        match_steps=obs.match_steps,  # ()
                    ),
                    public_params=public_params,  # (4,)
                )
            )
        features = jax.tree.map(lambda *leaves: jnp.stack(leaves), *views)  # leaves: (N, ...)
        energy = game.units.energy[..., 0]  # (N, U), own ship energy for each team
        active = game.units_mask & (energy >= 0) & (game.match_steps != 0)  # (N, U)
        targets = game.units.position[..., None, :] + self.sap_offsets  # (N, U, A-5, 2)
        in_bounds = ((targets >= 0) & (targets < jnp.array([params.map_width, params.map_height]))).all(-1)
        # in_bounds: (N, U, A-5); no hidden terrain or opponent information is used.
        sap = (
            active[..., None]
            & (energy >= params.unit_sap_cost)[..., None]
            & (jnp.abs(self.sap_offsets).max(-1) <= params.unit_sap_range)
            & in_bounds
        )  # (N, U, A-5)
        movement = jnp.broadcast_to(
            (active & (energy >= params.unit_move_cost))[..., None], (2, self.config.max_units, 4)
        )  # (N, U, 4), failed moves remain available as in native Lux
        masks = jnp.concatenate((jnp.ones((2, self.config.max_units, 1), dtype=bool), movement, sap), axis=-1)
        # masks: (N, U, A); stay is always available, including absent ships/boundary frames.
        return AgentObservation(observation=features, actions_mask=masks)

    def reset(self, key: jax.Array) -> tuple[AgentObservation[LuxObservation[jax.Array]], LuxState]:  # key: ()
        """Return (N, ...)/(N, U, A) observation/masks and (...) state leaves."""
        params_key, map_key = jax.random.split(key)  # each ()
        params = self.params
        if self.config.randomize_params:
            keys = jax.random.split(params_key, len(env_params_ranges))  # (P,), P randomized parameters
            params = params.replace(
                **{
                    # Native ship energy/positions use int16; strong int32 costs promote state leaves.
                    name: jax.random.choice(
                        keys[i], jnp.asarray(values, dtype=jnp.int16 if isinstance(values[0], int) else jnp.float32)
                    )  # ()
                    for i, (name, values) in enumerate(env_params_ranges.items())
                }
            )
        params = jax.tree.map(jnp.asarray, params)  # scalar array leaves
        observation, game = self.env.reset_env(map_key, params)  # observation/game leaves: (...)
        state = LuxState(game=game, params=params)
        return self._observation(observation, state), state

    def _autoreset(
        self,
        done: jax.Array,  # ()
        key: jax.Array,  # () typed PRNG key
        observation: AgentObservation[LuxObservation[jax.Array]],  # (N, ...)/(N, U, A)
        state: LuxState,  # leaves: (...)
    ) -> tuple[AgentObservation[LuxObservation[jax.Array]], LuxState]:
        return jax.lax.cond(done, lambda: self.reset(key), lambda: (observation, state))

    def _autoreset_batch(
        self,
        axis_size: int,
        in_batched: list[Any],  # Boolean pytrees for the four inputs.
        done: jax.Array,  # (E,) if batched, otherwise ()
        key: jax.Array,  # (E,) if batched, otherwise () typed PRNG key
        observation: AgentObservation[LuxObservation[jax.Array]],  # optionally E-prefixed observation/mask leaves
        state: LuxState,  # optionally E-prefixed state leaves
    ) -> tuple[tuple[AgentObservation[LuxObservation[jax.Array]], LuxState], Any]:
        """Reset in parallel only when any game ends; keep unfinished games unchanged."""

        def broadcast(value: jax.Array, batched: bool) -> jax.Array:  # (...) -> (E, ...)
            return value if batched else jnp.broadcast_to(value, (axis_size, *value.shape))  # (E, ...)

        done, key, observation, state = jax.tree.map(
            broadcast, (done, key, observation, state), tuple(in_batched)
        )  # done/key: (E,); observation/state leaves: (E, ...)

        def reset_finished() -> tuple[AgentObservation[LuxObservation[jax.Array]], LuxState]:
            fresh = jax.vmap(self.reset)(key)  # observation/state leaves: (E, ...)

            def select(new: jax.Array, old: jax.Array) -> jax.Array:  # each (E, ...)
                mask = done.reshape((axis_size,) + (1,) * (old.ndim - 1))  # (E, 1, ...)
                return jnp.where(mask, new, old)  # (E, ...)

            return jax.tree.map(select, fresh, (observation, state))  # observation/state leaves: (E, ...)

        # A scalar guard avoids eager map generation without serializing the environment axis.
        result = jax.lax.cond(done.any(), reset_finished, lambda: (observation, state))  # leaves: (E, ...)
        return result, jax.tree.map(lambda _: True, result)

    def step(
        self,
        key: jax.Array,  # () typed PRNG key
        state: LuxState,  # leaves: (...)
        action: jax.Array,  # (N, U) integer command IDs
    ) -> MultiAgentStep[LuxState, LuxObservation[jax.Array]]:
        step_key, reset_key = jax.random.split(key)  # each ()
        commands = self._decode_action(action)  # (N, U, 3)
        observation, game, _, _, game_ended, _ = self.env.step_env(
            step_key, state.game, {f"player_{team}": commands[team] for team in range(2)}, state.params
        )  # observation/game leaves: (...); game_ended: ()
        final_state = LuxState(game=game, params=state.params)
        final_observation = self._observation(observation, final_state)  # (N, ...)/(N, U, A)
        observation, next_state = self._autoreset(
            game_ended, reset_key, final_observation, final_state
        )  # observation: (N, ...)/(N, U, A); state leaves: (...)
        return MultiAgentStep(
            observation=observation,
            state=next_state,
            reward=(game.team_wins - state.game.team_wins).astype(jnp.float32),  # (N,)
            done=game_ended,  # ()
            agent_done=jnp.full(2, game_ended),  # (N,), retain team memory across matches/deaths
            agent_inactive=jnp.zeros(2, dtype=bool),  # (N,), teams always remain active
            terminated=game_ended,  # (), the full game is a task boundary, not a bootstrap timeout
            truncated=jnp.asarray(False),  # ()
            final_observation=final_observation,
            final_state=final_state,
        )

    def render(self, state: LuxState) -> NDArray[np.uint8]:  # leaves: (...) -> (416, 640, 3)
        """Return (416, 640, 3) uint8 privileged overview with visibility outlines and ship energy bars."""
        game, params = jax.device_get(state)
        steps, match_steps = int(game.steps), int(game.match_steps)
        match_count, match_length = int(params.match_count_per_episode), int(params.max_steps_in_match)
        horizon = (match_length + 1) * match_count
        completed = int(game.team_wins.sum())
        boundary = steps > 0 and match_steps == 0
        finished = steps >= horizon
        match = min(completed if boundary else completed + 1, match_count)
        turn = match_length if boundary else max(match_steps - 1, 0)
        status = f"MATCH {match} / {match_count}" + (" COMPLETE" if boundary else "")
        if finished:
            status = (
                f"GAME OVER - TEAM {int(game.team_wins.argmax())} WINS"
                if game.team_wins[0] != game.team_wins[1]
                else "GAME OVER - TIE"
            )
        energy = game.units.energy[..., 0]  # (N, U)
        active = game.units_mask & (energy >= 0)  # (N, U)
        team_colors = ((70, 180, 255), (255, 95, 85))
        scoring_color = (165, 140, 65)
        relic_color = (240, 205, 70)
        neutral_color = (25, 30, 45)
        nebula_color = (160, 100, 210)
        asteroid_color = (105, 105, 115)
        palette = np.array([[205, 115, 45], neutral_color, [45, 170, 155]], dtype=np.uint8)  # (3, RGB)
        minimum, maximum = int(params.min_energy_per_tile), int(params.max_energy_per_tile)

        def energy_colors(values: NDArray[np.number]) -> NDArray[np.uint8]:  # (...) -> (..., RGB)
            return np.stack(
                [np.interp(values, (minimum, 0, maximum), palette[:, channel]) for channel in range(3)], axis=-1
            ).astype(np.uint8)  # (..., RGB)

        tiles = np.asarray(game.map_features.tile_type).T  # (H, W)
        pixels = energy_colors(np.asarray(game.map_features.energy).T)  # (H, W, RGB)
        pixels[tiles == 2] = asteroid_color
        map_size = int(params.map_width)
        cell_size = 288 / map_size
        tile_ids = np.arange(map_size, dtype=np.int32)[None, :]  # (1, W)
        with (
            Image.fromarray(tile_ids) as index_image,
            index_image.resize((288, 1), resample=Image.Resampling.NEAREST) as index_board,
        ):
            # Use Pillow's exact pixel allocation, including fractional scaling and omitted tiles.
            tile_edges = np.searchsorted(np.asarray(index_board)[0], np.arange(map_size + 1))  # (W+1,)

        def tile_bounds(x: int, y: int) -> _RenderTile:
            return _RenderTile(
                left=16 + int(tile_edges[x]),
                top=72 + int(tile_edges[y]),
                width=int(tile_edges[x + 1] - tile_edges[x]),
                height=int(tile_edges[y + 1] - tile_edges[y]),
            )

        scoring = (game.relic_nodes_map_weights > 0) & (
            game.relic_nodes_map_weights <= game.relic_nodes_mask.sum() // 2
        )  # (W, H), same active scoring tiles used by the simulator
        title_font = ImageFont.load_default(size=20)
        font = ImageFont.load_default(size=16)
        small_font = ImageFont.load_default(size=13)
        badge_font = ImageFont.load_default(size=max(6, min(12, int(cell_size * 0.75))))
        with (
            Image.new("RGB", (640, 416), (14, 18, 28)) as frame,
            Image.fromarray(pixels) as tiles_image,
            tiles_image.resize((288, 288), resample=Image.Resampling.NEAREST) as board,
        ):
            frame.paste(board, (16, 72))
            draw = ImageDraw.Draw(frame)
            draw.text((16, 14), f"LUX S3  |  {status}", font=title_font, fill="white")
            draw.text(
                (16, 43),
                f"Turn {turn} / {match_length}   |   Game step {steps} / {horizon}",
                font=font,
                fill=(185, 195, 210),
            )
            for y, x in np.argwhere(tiles == 1):
                left, top, width, height = tile_bounds(int(x), int(y))
                if width and height:
                    draw.rectangle((left, top, left + width - 1, top + height - 1), outline=nebula_color)
            outlines = np.zeros((288, 288, 4), dtype=np.uint8)  # (H, W, RGBA)
            previous_edge = np.zeros((288, 288), dtype=bool)  # (H, W)
            stripes = (np.indices((288, 288)).sum(axis=0) // 3) % 2 == 1  # (H, W), alternate shared edges
            for team, color in enumerate(team_colors):
                with (
                    Image.fromarray(np.asarray(game.sensor_mask[team]).T.astype(np.uint8)) as mask_image,
                    mask_image.resize((288, 288), resample=Image.Resampling.NEAREST) as mask_board,
                ):
                    visible = np.asarray(mask_board, dtype=bool)  # (H, W)
                padded = np.pad(visible, 1)  # (H+2, W+2), outside the board is unseen
                edge = visible & ~(
                    padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
                )  # (H, W), only the perimeter, including holes
                outlines[edge & (~previous_edge | stripes)] = (*color, 255)
                previous_edge = edge  # (H, W), shared boundaries retain both team colors
            with Image.fromarray(outlines) as outline_image:
                frame.paste(outline_image, (16, 72), outline_image)
            for x, y in np.argwhere(scoring):
                left, top, width, height = tile_bounds(int(x), int(y))
                if not width or not height:
                    continue
                draw.rectangle(
                    (left + width / 12, top + height / 12, left + width * 5 / 6, top + height * 5 / 6),
                    outline=scoring_color,
                )
            for position, visible in zip(game.relic_nodes, game.relic_nodes_mask):
                if visible:
                    x, y = map(int, position)
                    left, top, width, height = tile_bounds(x, y)
                    if not width or not height:
                        continue
                    draw.ellipse(
                        (left + width / 4, top + height / 4, left + width * 3 / 4, top + height * 3 / 4),
                        fill=relic_color,
                    )
            for team, color in enumerate(team_colors):
                occupied_scoring: set[tuple[int, int]] = set()
                ship_ids = np.flatnonzero(active[team])  # (S,), existing ships with nonnegative energy
                ships_by_position: dict[tuple[int, int], list[int]] = {}
                for ship_id in ship_ids:
                    x, y = map(int, game.units.position[team, ship_id])
                    ships_by_position.setdefault((x, y), []).append(int(ship_id))
                for (x, y), stacked_ids in ships_by_position.items():
                    if scoring[x, y]:
                        occupied_scoring.add((x, y))
                    left, top, width, height = tile_bounds(x, y)
                    if not width or not height:
                        continue
                    draw.rectangle(
                        (left + width / 3, top + height / 3, left + width * 2 / 3, top + height * 2 / 3),
                        fill=color,
                    )
                    # Omit bars and badges when they cannot fit inside a tile.
                    if min(width, height) >= 10:
                        if len(stacked_ids) > 1:
                            draw.rectangle(
                                (left + 1, top + 1, left + width - 2, top + height - 2),
                                fill=neutral_color,
                                outline=color,
                            )
                            draw.text(
                                (left + width / 2, top + height / 2),
                                str(len(stacked_ids)),
                                font=badge_font,
                                anchor="mm",
                                fill=color,
                            )
                        else:
                            bar_left = int(left + width / 6)
                            bar_top = int(top + height * 2 / 3) + 2  # Leave one row after the inclusive ship edge.
                            bar_right, bar_bottom = int(left + width * 5 / 6) - 1, top + height - 2
                            draw.rectangle((bar_left, bar_top, bar_right, bar_bottom), fill=neutral_color)
                            filled = round(
                                (bar_right - bar_left + 1)
                                * int(energy[team, stacked_ids[0]])
                                / int(params.max_unit_energy)
                            )
                            if filled > 0:
                                draw.rectangle((bar_left, bar_top, bar_left + filled - 1, bar_bottom), fill=color)
                top = 72 + team * 100
                points = "--" if boundary else str(int(game.team_points[team]))
                ships = int(active[team].sum())
                total_energy = int(energy[team][active[team]].sum())
                draw.rounded_rectangle((324, top, 624, top + 94), radius=8, fill=(25, 31, 44))
                draw.text((336, top + 10), f"TEAM {team}", font=title_font, fill=color)
                draw.text(
                    (612, top + 14),
                    f"Scoring tiles: {len(occupied_scoring)}",
                    anchor="rt",
                    font=font,
                    fill="white",
                )
                draw.text(
                    (336, top + 40),
                    f"Relic points {points}   |   Wins {int(game.team_wins[team])}",
                    font=font,
                    fill="white",
                )
                draw.text(
                    (336, top + 66),
                    f"Ships {ships} / {int(params.max_units)}   |   Energy {total_energy}",
                    font=font,
                    fill=(185, 195, 210),
                )
            draw.text((324, 274), f"Active relic nodes: {int(game.relic_nodes_mask.sum())}", font=font, fill="white")
            if boundary:
                draw.text((324, 293), "Relic points reset at match end.", font=small_font, fill=(185, 195, 210))
            draw.text((324, 309), "Tile energy / turn", font=small_font, fill="white")
            scale_values = np.concatenate((np.linspace(minimum, 0, 151), np.linspace(0, maximum, 151)[1:]))  # (301,)
            for offset, rgb in enumerate(energy_colors(scale_values)):  # rgb: (3,)
                draw.line((324 + offset, 329, 324 + offset, 337), fill=tuple(int(channel) for channel in rgb))
            for x, value, anchor in ((324, minimum, "lt"), (474, 0, "mt"), (624, maximum, "rt")):
                draw.text((x, 339), str(value), anchor=anchor, font=small_font, fill=(185, 195, 210))
            draw.text(
                (324, 355),
                f"Nebula drain: {int(params.nebula_tile_energy_reduction)} / turn",
                font=small_font,
                fill=(185, 195, 210),
            )
            for tile, (label, color) in enumerate(
                (("Empty", neutral_color), ("Nebula", nebula_color), ("Asteroid", asteroid_color))
            ):
                left = 16 + tile * 108
                draw.rectangle(
                    (left, 374, left + 12, 386),
                    fill=neutral_color if tile == 1 else color,
                    outline=nebula_color if tile == 1 else (130, 140, 155),
                )
                draw.text((left + 18, 373), label, font=small_font, fill=(185, 195, 210))
            draw.ellipse((343, 377, 349, 383), fill=relic_color)
            draw.text((358, 373), "Relic", font=small_font, fill=(185, 195, 210))
            draw.rectangle((448, 374, 460, 386), outline=scoring_color)
            draw.text((466, 373), "Scoring tile", font=small_font, fill=(185, 195, 210))
            for team, color in enumerate(team_colors):
                left = 16 + team * 108
                draw.rectangle((left + 3, 399, left + 9, 405), fill=color)
                draw.text((left + 18, 395), f"Team {team}", font=small_font, fill=(185, 195, 210))
            draw.text(
                (244, 395),
                "Full map (hidden info); outlines = vision",
                font=small_font,
                fill=(185, 195, 210),
            )
            return np.array(frame, dtype=np.uint8)  # (416, 640, 3)
