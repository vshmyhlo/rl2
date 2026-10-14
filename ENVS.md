# JAX environments

Families from the 14 **(jax env)** projects in [README.md](README.md).
Upstream inventory checked 2026-10-14; variants grouped. rl2 support varies.

- [JaxMARL](https://github.com/bold-lab-ai/JaxMARL) — multi-agent ([registry](https://github.com/bold-lab-ai/JaxMARL/blob/main/jaxmarl/registration.py)).
  - MPE: Simple, Tag, World Comm, Spread, Crypto, Speaker Listener, Push,
    Adversary, Reference, and FACMAC predator–prey variants.
    - rl2's [adapter](src/rl2/envs/mpe_env.py): discrete Simple Spread/Reference.
      [Spread preset](configs/mappo/jaxmarl/mpe/simple_spread.yaml)
      (`MPE_simple_spread_v3`): `N = 3` agents, `L = 3` landmarks.
    - Observation: float32 `(4N + 2L,)` per agent (18 in preset), in order:
      - Own velocity `(2,)`, position `(2,)`.
      - Relative landmark positions `(L, 2)`, flattened.
      - Relative agent positions `(N - 1, 2)`, flattened.
      - Other agents' messages `(N - 1, 2)`, flattened; zero (silent agents).
    - Actions: `Discrete(5)` — 0: no force, 1: left, 2: right, 3: down, 4: up.
      Inertia and collisions still apply.
    - Arrays: observations `(N, O)`, int32 actions `(N,)`, all-valid boolean masks
      `(N, A)`. Here `O = 18`, `A = 5`; batching prepends environment axis `E`.
    - Reward: `r_i = -(1 - α)D - αC_i`; `α = local_ratio` (preset: 0.5),
      `D` = sum of landmark-to-nearest-agent distances, `C_i` = agent `i`'s
      overlaps with others. Collisions penalize both participants.
    - [Reference preset](configs/mappo/jaxmarl/mpe/simple_reference.yaml)
      (`MPE_simple_reference_v3`): two communicating agents, three landmarks;
      each sees the partner's target.
      - Observation `(21,)`: own velocity (2), relative landmarks (6), partner's
        target color (3), partner's last message (10).
      - Actions: `Discrete(50)`, `5 * symbol + movement`; symbols 0–9, Spread
        movement controls. Each sends a one-hot message visible next step.
      - Arrays: observations `(2, 21)`, actions `(2,)`, masks `(2, 50)`.
      - Reward: `r_i = -(1 - α)(d_0 + d_1)/2 - αd_j`; `d_i` = agent `i`'s
        target distance, `j` = partner. Preset `α = 0`: shared negative mean
        distance. No collisions or direct communication reward/cost.
    - Both: dense post-step rewards; higher is better. No success bonus/termination;
      truncate at `max_steps`. Episode return sums mean agent rewards.
  - Overcooked and OvercookedV2: cooperative cooking.
  - SMAX: StarCraft-style combat; heuristic/learned enemies.
  - Hanabi: cooperative hidden-information card game.
  - STORM: two-/multi-player spatial matrix games (`storm`, `storm_2p`, `storm_np`).
    - rl2's [adapter](src/rl2/envs/storm_env.py): `storm_np`, shared MLP/LSTM IPPO.
      Collect red/blue coins, then zap another coin holder to play a matrix game.
    - [Basic](configs/mappo/jaxmarl/storm/basic.yaml): two agents, 6×6 board,
      cooperative payoffs, 128 steps. [Advanced](configs/mappo/jaxmarl/storm/advanced.yaml):
      four agents, 8×8 board, Prisoner's Dilemma, 256 steps. Both: six coins/type,
      randomized placement, five-step interaction freeze followed by respawn.
    - Observation: egocentric symbolic `(5, 5, 14)` grid → float32 `(350,)` per
      agent; objects, self/others, orientation, pickup readiness, inventory and freeze features.
    - Actions: `Discrete(8)` — turn left/right, forward, stay, zap forward/ahead/right/left
      (IDs 0–7). All-valid masks; frozen agents stay. Arrays: obs `(N, 350)`, actions `(N,)`.
    - Reward: zero except interactions; coin proportions `p`, `q` give `pᵀ M_i q`.
      Rows/columns are red/blue: basic gives both `M = [[3,1],[1,0]]`;
      advanced gives `(3,3)` for red/red, `(0,5)` red/blue, `(5,0)` blue/red, `(1,1)` blue/blue.
      Payoffs sum over accepted partners; the adapter fixes upstream reward overwrites.
      Individual rewards train each agent; logged return sums their mean.
    - One round/episode, truncated at `max_steps` with final-state bootstrapping;
      memory persists across interactions. Run either preset with
      `uv run mappo --config configs/mappo/jaxmarl/storm/basic.yaml` (or `advanced.yaml`).
  - Coin Game: resource collection/social dilemmas.
  - Switch Riddle: communication puzzle.
  - JaxNav: multi-robot navigation/collision avoidance.
  - Multi-Agent Brax: Ant, HalfCheetah, Hopper, Humanoid, Walker2d; optional Brax dependency.
  - JaxRobotarium: Navigation, Discovery, Material Transport, Warehouse,
    Arctic Transport, Foraging, RWARE, Predator Prey; optional package.

- [Pgx](https://github.com/sotetsuk/pgx) — games ([registry](https://github.com/sotetsuk/pgx/blob/main/pgx/core.py)).
  - Board games: Backgammon, Chess, Gardner Chess, Shogi, Animal Shogi,
    Go (9×9 and 19×19), Hex, Othello, Connect Four, Tic-Tac-Toe.
  - Card/tile games: Kuhn Poker, Leduc Hold'em, Sparrow Mahjong.
  - Bridge Bidding: `BridgeBidding` class.
  - Puzzle: 2048.
  - MinAtar: Asterix, Breakout, Freeway, Seaquest, Space Invaders.

- [XLand-MiniGrid](https://github.com/dunnolab/xland-minigrid#environments-) — grid worlds and meta-RL.
  - XLand-MiniGrid: configurable layouts, sizes, goals, transformation rules; benchmark rulesets.
  - Navigation: Empty, EmptyRandom, FourRooms.
  - Keys and doors: LockedRoom, DoorKey, Unlock, UnlockPickUp, BlockedUnlockPickUp.
  - Memory: match objects to remembered cues.
  - Playground: open-ended object interaction.

- [minimax](https://github.com/facebookresearch/minimax) — maze curricula.
  - Maze: generated training layouts, held-out evaluation layouts.
  - Maze UED: teacher-generated mazes.

- [JaxUED](https://github.com/DramaCow/jaxued#supported-environments) — curriculum learning.
  - Maze: generated/fixed levels, curriculum editing.
  - Craftax: symbolic-observation curriculum example.
  - Gymnax adapters: CartPole, Acrobot, Pendulum; variable physics.

- [Brax](https://github.com/google/brax) — continuous control ([registry](https://github.com/google/brax/blob/main/brax/envs/__init__.py)).
  - Locomotion: Ant, HalfCheetah, Hopper, Humanoid, Swimmer, Walker2d.
  - Posture and balance: HumanoidStandup, InvertedPendulum, InvertedDoublePendulum.
  - Manipulation: Reacher, Pusher.
  - Fast: minimal test environment.

- [Gymnax](https://github.com/RobertTLange/gymnax#implemented-accelerated-environments-) — control, arcade, research tasks.
  - Classic control: Acrobot, CartPole, Pendulum, MountainCar, MountainCarContinuous.
  - MinAtar: Asterix, Breakout, Freeway, Seaquest, SpaceInvaders.
  - bsuite: Catch, DeepSea, MemoryChain, UmbrellaChain, DiscountingChain,
    MNISTBandit, SimpleBandit.
  - Other tasks: FourRooms, MetaMaze, PointRobot, BernoulliBandit,
    GaussianBandit, Reacher, Swimmer.

- [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) — MuJoCo MJX robotics.
  - [DM Control suite](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/dm_control_suite/__init__.py):
    Acrobot swing-up; BallInCup; Cartpole balance/swing-up; Cheetah run;
    Finger spin/turn; Fish swim; Hopper hop/stand; Humanoid stand/walk/run;
    Pendulum swing-up; PointMass; Reacher easy/hard; six-link Swimmer;
    Walker stand/walk/run. Some have sparse rewards.
  - [Locomotion](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/locomotion/__init__.py):
    joystick control for Apollo, Barkour, Berkeley Humanoid, G1, Go1, OP3,
    Spot, T1; selected flat/rough terrain variants;
    Go1 get-up/handstand/footstand; H1 in-place and joystick gait tracking;
    Spot get-up and gait tracking.
  - [Manipulation](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/manipulation/__init__.py):
    Aloha handover and peg insertion; Panda cube picking (standard,
    orientation, Cartesian), cabinet opening, and Robotiq cube pushing;
    LEAP cube reorientation and Z-axis rotation; Aero cube Z-axis rotation.
    - **PandaPickCube**: grasp a cube and move it to a target above the table;
      randomized positions, fixed target orientation.
      [PPO preset](configs/ppo/mujoco_playground/manipulation/panda_pick_cube.yaml):
      MLP + LSTM, bounded continuous policy, MJX JAX backend.
      [Task source](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/manipulation/franka_emika_panda/pick.py) (Playground 0.2.0).
      - Setup/train from repo root:

        ```bash
        uv run python -m rl2.envs.playground_assets
        uv run --env-file nvida.env ppo --config configs/ppo/mujoco_playground/manipulation/panda_pick_cube.yaml
        ```

        Setup caches pinned Panda XML/meshes in the installed package.
        Video/evaluation run every 489 iterations. The NVIDIA preset selects
        headless EGL rendering. The training image includes EGL libraries and
        requests NVIDIA graphics drivers; rebuild it to apply rendering changes.
        Outside the container, install `libegl1 libgl1 libopengl0` (Debian/Ubuntu)
        and the NVIDIA driver's OpenGL/EGL libraries. Hyperparameters are untuned.
      - **Observation:** float32 `(66,)`, unnormalized, in order:
        - Positions (16): arm angles (7), fingers (2), cube XYZ + quaternion (7).
        - Velocities (15): robot joints (9), cube (6).
        - Gripper world position (3).
        - Gripper/cube orientations (6 each): last two rotation-matrix rows.
        - Cube − gripper (3), target − cube (3).
        - Target − cube orientation (6): first two matrix rows.
        - Actuator targets − joint positions (8): arm (7), first finger (1).
      - **Actions:** float32 `(8,)` in `[-1, 1]`: arm (7), coupled gripper (1).
        Targets += `0.04 * action`, clipped to actuator limits; radians for arm,
        metres for finger (`[0, 0.04]`). Gripper + opens, − closes; zero holds.
        Control: 50 Hz, four 5 ms physics steps. Batching prepends `B` to spaces.
      - **Reward:** dense post-step `r = 4G + 8IB + 0.25F + 0.3Q`:
        - `G = 1 - tanh(5 * ||cube - gripper||)` (reaching).
        - `B = 1 - tanh(5 * (0.9p + 0.1e))` (target matching): `p` = position
          distance, `e` = norm of first-two-row rotation-matrix difference.
        - `I` latches to 1 when gripper is <0.012 m from cube (proximity, not grasp).
        - `F` = 1 without finger-pad/hand floor contact, else 0.
        - `Q = 1 - tanh(||arm_angles - initial_arm_angles||)` (pose regularization).
        Clipped to `[-10000, 10000]`; episode return sums rewards.
        Higher is better; no grasp/success bonus or success termination.
      - **Episode:** truncates at 150 steps (3 s); terminates if cube `z < 0`, any cube
        coordinate has `abs > 1 m`, or positions/velocities become NaN.
        Autoreset preserves final observations for timeout bootstrapping.

- [Jumanji](https://github.com/instadeepai/jumanji#environments-) — puzzles/optimization.
  - Logic: Game2048, GraphColoring, Minesweeper, RubiksCube, SlidingTilePuzzle, Sudoku.
  - Packing and scheduling: BinPack, FlatPack, JobShop, Knapsack, Tetris.
  - Routing: Cleaner, Connector, CVRP, MultiCVRP, Maze, RobotWarehouse,
    Snake, TSP, MMST (multiple minimum spanning trees), PacMan, Sokoban.

- [Craftax](https://github.com/MichaelTMatthews/Craftax) — survival/crafting ([factory](https://github.com/MichaelTMatthews/Craftax/blob/main/craftax/craftax_env.py)).
  - Craftax: `Craftax-Symbolic-v1` and `Craftax-Pixels-v1`.
    - **Symbolic:** `(8268,)`; flattened 9×11 tile/item/entity/light features
      plus 51 inventory, equipment, player-state, and status values.
    - **Pixels:** `(130, 110, 3)`; 9×11 local tiles plus inventory/status HUD.
    - **Actions:** `Discrete(43)`; Classic controls plus rest, stairs, diamond
      tools, armour, arrows, spells, torches, potions, books, enchanting, attributes.
    - **Achievements:** 67; first unlock pays 1/3/5/8 by difficulty
      ([weights](https://github.com/MichaelTMatthews/Craftax/blob/main/craftax/craftax/constants.py)).
    - **Episode:** death or defeating the necromancer; default timeout 100,000 steps.
  - Craftax Classic (Crafter-style): `Craftax-Classic-Symbolic-v1`
    and `Craftax-Classic-Pixels-v1`.
    - **Symbolic:** `(1345,)`; flattened 7×9 tile/entity features plus 22
      inventory, vitals, facing, light, and sleep values.
    - **Pixels:** `(63, 63, 3)`; 7×9 local tiles plus inventory/status HUD.
    - **Actions:** `Discrete(17)` — 0: no-op; 1–4: left/right/up/down;
      5: interact (mine/attack/eat/drink); 6: sleep; 7–10: place stone/table/furnace/plant;
      11–13: craft wood/stone/iron pickaxe; 14–16: craft corresponding swords.
      Full Craftax shares IDs 0–16.
    - **Achievements:** 22; +1 per first unlock (collecting, crafting, combat, survival).
    - **Episode:** death/lava; default timeout 10,000 steps.
  - **rl2:** [PPO preset](configs/ppo/craftax/craftax.yaml), symbolic MLP + LSTM;
    `uv run ppo --config configs/ppo/craftax/craftax.yaml`. Set `env.env_id` to either
    symbolic variant; pixels require `model.encoder.type: rgb`. Videos remain RGB.
    Details above use Craftax 1.6.1.
  - **Arrays:** float32 symbolic `(B, F)` with native feature scaling, or RGB
    `(B, H, W, 3)` in `[0, 1]`; both player-centered. Int32 actions `(B,)`,
    no action mask; unmet prerequisites prevent the effect while the world advances.
  - **Reward:** `r = sum(new achievement weights) + 0.1 * (health_after - health_before)`;
    unlocks pay once per episode, healing rewards and damage penalizes.
    Logged return sums raw rewards ([logic](https://github.com/MichaelTMatthews/Craftax/blob/main/craftax/craftax/game_logic.py)).
  - **Reset:** new procedural world per episode. `env.max_steps: null` uses native
    limits; set an integer to override. Autoreset preserves final observations
    for timeout bootstrapping.

- [Navix](https://github.com/epignatelli/navix) — MiniGrid tasks ([environment modules](https://github.com/epignatelli/navix/blob/main/navix/environments/__init__.py)).
  - Navigation: Empty rooms, FourRooms, MultiRoom, Crossings, LavaGap,
    DynamicObstacles, DistShift.
  - Keys and doors: DoorKey, KeyCorridor, LockedRoom, Unlock, UnlockPickup,
    BlockedUnlockPickup, RedBlueDoors, ObstructedMaze.
  - Object tasks: GoToDoor, GoToObject, Fetch, PutNear.
  - Memory and exploration: Memory, Playground.

- [Kinetix](https://github.com/FLAIROx/Kinetix) — 2D physics.
  - Procedural levels: small/medium/large; motor/thruster control, join goal objects, avoid hazards.
  - [Handmade levels](https://github.com/FLAIROx/Kinetix/blob/main/configs/eval/eval_all.yaml):
    cars, unicycles, thruster control, robot arms, grasping, cartpole, balance,
    lunar landing, pinball, Angry Birds, Flappy Bird, mazes, platformers,
    obstacle avoidance, gears, and lever puzzles.
  - Locomotion: HalfCheetah, Hopper, Swimmer, Walker variants.

- [JAXtari](https://github.com/k4ntz/JAXtari) — Atari reimplementations ([game status list](https://github.com/k4ntz/JAXtari/blob/master/games_covered.md)).
  - [Breakout PPO preset](configs/ppo/jaxtari/breakout.yaml): native uint8 RGB
    `(210, 160, 3)`, CNN + LSTM; one frame per action. `Discrete(4)`:
    0 no-op, 1 fire, 2 right, 3 left. FIRE starts each ball; life loss continues
    the episode. Reward is raw score gain (1/4/7 per brick row tier).
    Ends at zero lives, cleared bricks, or 108,000 steps (timeout bootstrapped).
    Install sprites once with `uv run python -m jaxtari.install_sprites`
    (decline ROM ownership to use alternative sprites), then run
    `uv run ppo --config configs/ppo/jaxtari/breakout.yaml`.
  - Atari57 (close to originals): Alien, Amidar, Asterix, Asteroids,
    Atlantis, Bank Heist, Beam Rider, Berzerk, Boxing, Breakout, Centipede,
    Chopper Command, Enduro, Fishing Derby, Freeway, Frostbite, Gopher,
    Gravitar, Kangaroo, Montezuma's Revenge, Ms. Pac-Man, Name This Game,
    Phoenix, Pong, Q*bert, River Raid, Seaquest, Skiing, Space Invaders,
    Surround, Tennis, Time Pilot, Venture, Video Pinball, Yars' Revenge.
  - Substantial differences: Assault, Road Runner, Up N Down, Wizard of Wor.
  - Additional games: Air Raid, Backgammon, Basic Math, Blackjack,
    Casino Blackjack/Five Stud Poker/Poker Solitaire, Donkey Kong, Flag Capture,
    Galaxian, Hangman, Haunted House, Human Cannonball, Journey Escape, Kaboom,
    King Kong, Klax, Laser Gates, Miniature Golf, Othello, Pac-Man,
    Sir Lancelot, Slot Machine, Space War, Tetris, Tron, Turmoil,
    Video Checkers, Video Chess, Video Cube, Word Zapper.
  - Work-in-progress/unimplemented games omitted.

- [continual-foragax](https://pypi.org/project/continual-foragax/#registry-and-included-environments) — changing-reward foraging.
  - `ForagaxBig-v5`: multi-biome, Fourier-modulated rewards.
  - `ForagaxSquareWaveTwoBiome-v11`: two biomes, square-wave rewards.
  - `ForagaxTwoBiomeLarge-v1`: 15×15 Morel/Oyster biomes, Deathcaps.
  - Custom worlds: objects, biomes, respawning, rewards; programmable weather-driven rewards.
