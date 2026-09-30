# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""DQN Zoo Atari-57 reference returns, vendored without changing score values.

Adapted from DeepMind's atari_data.py; the pinned source is below. The upstream
scores average at least three episodes with 1..30 initial no-ops and a
108,000-frame cap. They are fixed normalization references, not measurements
of a random/human policy under this project's sticky-action settings.
Use the target paper's reference table when it differs from this one.

Changes: added types, source metadata, and ALE environment ID lookup; omitted
upstream normalization code. Apache-2.0 license: licenses/dqn_zoo.txt.
"""

REFERENCE_SOURCE = (
    "https://github.com/google-deepmind/dqn_zoo/blob/359945dcb6f6a04f16e380b210c4f22b57681905/dqn_zoo/atari_data.py"
)

# Each pair is (random return, human return), in original game-score units.
ATARI_REFERENCE_SCORES: dict[str, tuple[float, float]] = {
    "alien": (227.8, 7127.7),
    "amidar": (5.8, 1719.5),
    "assault": (222.4, 742.0),
    "asterix": (210.0, 8503.3),
    "asteroids": (719.1, 47388.7),
    "atlantis": (12850.0, 29028.1),
    "bank_heist": (14.2, 753.1),
    "battle_zone": (2360.0, 37187.5),
    "beam_rider": (363.9, 16926.5),
    "berzerk": (123.7, 2630.4),
    "bowling": (23.1, 160.7),
    "boxing": (0.1, 12.1),
    "breakout": (1.7, 30.5),
    "centipede": (2090.9, 12017.0),
    "chopper_command": (811.0, 7387.8),
    "crazy_climber": (10780.5, 35829.4),
    "defender": (2874.5, 18688.9),
    "demon_attack": (152.1, 1971.0),
    "double_dunk": (-18.6, -16.4),
    "enduro": (0.0, 860.5),
    "fishing_derby": (-91.7, -38.7),
    "freeway": (0.0, 29.6),
    "frostbite": (65.2, 4334.7),
    "gopher": (257.6, 2412.5),
    "gravitar": (173.0, 3351.4),
    "hero": (1027.0, 30826.4),
    "ice_hockey": (-11.2, 0.9),
    "jamesbond": (29.0, 302.8),
    "kangaroo": (52.0, 3035.0),
    "krull": (1598.0, 2665.5),
    "kung_fu_master": (258.5, 22736.3),
    "montezuma_revenge": (0.0, 4753.3),
    "ms_pacman": (307.3, 6951.6),
    "name_this_game": (2292.3, 8049.0),
    "phoenix": (761.4, 7242.6),
    "pitfall": (-229.4, 6463.7),
    "pong": (-20.7, 14.6),
    "private_eye": (24.9, 69571.3),
    "qbert": (163.9, 13455.0),
    "riverraid": (1338.5, 17118.0),
    "road_runner": (11.5, 7845.0),
    "robotank": (2.2, 11.9),
    "seaquest": (68.4, 42054.7),
    "skiing": (-17098.1, -4336.9),
    "solaris": (1236.3, 12326.7),
    "space_invaders": (148.0, 1668.7),
    "star_gunner": (664.0, 10250.0),
    "surround": (-10.0, 6.5),
    "tennis": (-23.8, -8.3),
    "time_pilot": (3568.0, 5229.2),
    "tutankham": (11.4, 167.6),
    "up_n_down": (533.4, 11693.2),
    "venture": (0.0, 1187.5),
    "video_pinball": (16256.9, 17667.9),
    "wizard_of_wor": (563.5, 4756.5),
    "yars_revenge": (3092.9, 54576.9),
    "zaxxon": (32.5, 9173.3),
}

# ALE uses CamelCase names (SpaceInvaders, MsPacman, UpNDown); source keys use underscores.
_ALE_SCORES = {game.replace("_", ""): scores for game, scores in ATARI_REFERENCE_SCORES.items()}


def get_reference_scores(env_id: str) -> tuple[float, float] | None:
    """Return the published pair for an ALE v5 game, or None outside Atari-57."""
    if not env_id.startswith("ALE/") or not env_id.endswith("-v5"):
        return None
    return _ALE_SCORES.get(env_id[4:-3].lower())
