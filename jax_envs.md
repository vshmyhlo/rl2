# JAX-native environments

Libraries and environment families implemented natively in JAX, with environment
dynamics that support JAX transformations such as `jit` and `vmap`.
Includes the 14 **(jax env)** projects in [README.md](README.md).
Upstream registries and bundled task lists checked 2026-11-04. Lists cover built-in
environments and named variants; arbitrary generated worlds, custom registrations,
and parameter combinations are not separate entries. Upstream support does not
imply rl2 integration.

- [JaxMARL](https://github.com/bold-lab-ai/JaxMARL) — multi-agent ([registry](https://github.com/bold-lab-ai/JaxMARL/blob/main/jaxmarl/registration.py)).
  - MPE: Simple, Tag, World Comm, Spread, Crypto, Speaker Listener, Push,
    Adversary, Reference, and FACMAC predator–prey (base, 3a, 6a, 9a).
  - Overcooked (`overcooked`): cooperative cooking. [Layouts](https://github.com/bold-lab-ai/JaxMARL/blob/main/jaxmarl/environments/overcooked/layouts.py):
    `cramped_room`, `asymm_advantages`, `coord_ring`, `forced_coord`, `counter_circuit`.
  - OvercookedV2 (`overcooked_v2`): the same five layouts plus
    [additional layouts](https://github.com/bold-lab-ai/JaxMARL/blob/main/jaxmarl/environments/overcooked_v2/layouts.py):
    `cramped_room_v2`, `asymm_advantages_recipes_center`, `asymm_advantages_recipes_right`,
    `asymm_advantages_recipes_left`, `two_rooms`, `two_rooms_both`, `long_room`,
    `fun_coordination`, `more_fun_coordination`, `fun_symmetries`, `fun_symmetries_plates`,
    `fun_symmetries1`, `grounded_coord_simple`, `grounded_coord_ring`, `test_time_simple`,
    `test_time_wide`, `demo_cook_simple`, `demo_cook_wide`.
  - SMAX: `SMAX`, `HeuristicEnemySMAX`, `LearnedPolicyEnemySMAX`.
    [Scenarios](https://github.com/bold-lab-ai/JaxMARL/blob/main/jaxmarl/environments/smax/smax_env.py):
    `3m`, `2s3z`, `25m`, `3s5z`, `8m`, `5m_vs_6m`, `10m_vs_11m`, `27m_vs_30m`,
    `3s5z_vs_3s6z`, `3s_vs_5z`, `6h_vs_8z`, `smacv2_5_units`, `smacv2_10_units`,
    `smacv2_20_units`.
  - Hanabi: cooperative hidden-information card game.
  - STORM: two-/multi-player spatial matrix games (`storm`, `storm_2p`, `storm_np`).
  - Coin Game: resource collection/social dilemmas.
  - Switch Riddle: communication puzzle.
  - JaxNav: multi-robot navigation/collision avoidance.
    [Named singleton maps](https://github.com/bold-lab-ai/JaxMARL/blob/main/jaxmarl/environments/jaxnav/jaxnav_singletons.py):
    `BlankTest`, `MiddleTest`, `BlankCross2`, `BlankCross4`, `BlankCrossUneven2`, `CircleCross`,
    `SingleNav1`, `SingleNav2`, `SingleNav3`, `Corridor4`, `Corridor8`, `LongCorridor2`,
    `Layby4`, `Corner2`, `Chicane2`, `Chicane4`, `NarrowChicane2a`, `NarrowChicane2b`.
  - Multi-Agent Brax: Ant, HalfCheetah, Hopper, Humanoid, Walker2d (`ant_4x2`, `halfcheetah_6x1`,
    `hopper_3x1`, `humanoid_9|8`, `walker2d_2x3`); optional Brax dependency.
  - [JaxRobotarium](https://github.com/GT-STAR-Lab/JaxRobotarium): Navigation,
    Discovery, Material Transport, Warehouse, Arctic Transport, Foraging, RWARE, Predator Prey;
    requires the optional JaxRobotarium package.

- [Pgx](https://github.com/sotetsuk/pgx) — games ([registry](https://github.com/sotetsuk/pgx/blob/main/pgx/core.py)).
  - Board games: Backgammon, Chess, Gardner Chess, Shogi, Animal Shogi,
    Go (9×9 and 19×19), Hex, Othello, Connect Four, Tic-Tac-Toe.
  - Card/tile games: Kuhn Poker, Leduc Hold'em, Sparrow Mahjong.
  - Bridge Bidding: `bridge_bidding` (`BridgeBidding` class).
  - Puzzle: 2048.
  - MinAtar: Asterix, Breakout, Freeway, Seaquest, Space Invaders.

- [XLand-MiniGrid](https://github.com/dunnolab/xland-minigrid) — grid worlds and meta-RL
  ([registry](https://github.com/dunnolab/xland-minigrid/blob/main/src/xminigrid/__init__.py)).
  - XLand 1-room layouts: `XLand-MiniGrid-R1-9x9`, `XLand-MiniGrid-R1-11x11`,
    `XLand-MiniGrid-R1-13x13`, `XLand-MiniGrid-R1-15x15`, `XLand-MiniGrid-R1-17x17`.
  - XLand 2-room layouts: `XLand-MiniGrid-R2-9x9`, `XLand-MiniGrid-R2-11x11`,
    `XLand-MiniGrid-R2-13x13`, `XLand-MiniGrid-R2-15x15`, `XLand-MiniGrid-R2-17x17`.
  - XLand 4-room layouts: `XLand-MiniGrid-R4-9x9`, `XLand-MiniGrid-R4-11x11`,
    `XLand-MiniGrid-R4-13x13`, `XLand-MiniGrid-R4-15x15`, `XLand-MiniGrid-R4-17x17`.
  - XLand 6-room layouts: `XLand-MiniGrid-R6-13x13`, `XLand-MiniGrid-R6-17x17`,
    `XLand-MiniGrid-R6-19x19`.
  - XLand 9-room layouts: `XLand-MiniGrid-R9-16x16`, `XLand-MiniGrid-R9-19x19`,
    `XLand-MiniGrid-R9-25x25`.
  - Navigation: `MiniGrid-Empty-5x5`, `MiniGrid-Empty-6x6`, `MiniGrid-Empty-8x8`,
    `MiniGrid-Empty-16x16`, `MiniGrid-EmptyRandom-5x5`, `MiniGrid-EmptyRandom-6x6`,
    `MiniGrid-EmptyRandom-8x8`, `MiniGrid-EmptyRandom-16x16`, `MiniGrid-FourRooms`.
  - Keys and doors: `MiniGrid-BlockedUnlockPickUp`, `MiniGrid-DoorKey-5x5`, `MiniGrid-DoorKey-6x6`,
    `MiniGrid-DoorKey-8x8`, `MiniGrid-DoorKey-16x16`, `MiniGrid-LockedRoom`, `MiniGrid-Unlock`,
    `MiniGrid-UnlockPickUp`.
  - Memory and exploration: `MiniGrid-MemoryS8`, `MiniGrid-MemoryS16`, `MiniGrid-MemoryS32`,
    `MiniGrid-MemoryS64`, `MiniGrid-MemoryS128`, `MiniGrid-Playground`.

- [minimax](https://github.com/facebookresearch/minimax) — maze curricula.
  - Training: `Maze`; `UEDMaze` is the teacher environment registered for `Maze`
    ([modules](https://github.com/facebookresearch/minimax/tree/main/src/minimax/envs/maze)).
  - [Evaluation mazes](https://github.com/facebookresearch/minimax/blob/main/src/minimax/envs/maze/maze_ood.py):
    `Maze-SixteenRooms`, `Maze-SixteenRooms2`, `Maze-Labyrinth`, `Maze-Labyrinth2`,
    `Maze-LabyrinthFlipped`, `Maze-StandardMaze`, `Maze-StandardMaze2`, `Maze-StandardMaze3`,
    `Maze-SmallCorridor`, `Maze-LargeCorridor`, `Maze-FourRooms`, `Maze-Crossing`,
    `Maze-PerfectMaze`, `Maze-PerfectMazeMedium`, `Maze-PerfectMazeXL`, `Maze-Memory`.

- [JaxUED](https://github.com/DramaCow/jaxued#supported-environments) — curriculum learning.
  - [Maze, MazeEditor, and MazeSolved](https://github.com/DramaCow/jaxued/blob/main/src/jaxued/environments/maze/__init__.py):
    navigation, curriculum editing, and solved-maze support.
  - [Prefab maze levels](https://github.com/DramaCow/jaxued/blob/main/src/jaxued/environments/maze/level.py):
    `TrivialMaze`, `TrivialMaze2`, `TrivialMaze3`, `SixteenRooms`, `SixteenRooms2`,
    `Labyrinth`, `LabyrinthFlipped`, `Labyrinth2`, `StandardMaze`, `StandardMaze2`,
    `StandardMaze3`.
  - Craftax: `CraftaxSymbolic` curriculum example.
  - Gymnax example adapters: CartPole, Acrobot, Pendulum; variable physics.

- [Brax](https://github.com/google/brax) — continuous control ([registry](https://github.com/google/brax/blob/main/brax/envs/__init__.py)).
  - Locomotion: Ant, HalfCheetah, Hopper, Humanoid, Swimmer, Walker2d.
  - Posture and balance: HumanoidStandup, InvertedPendulum, InvertedDoublePendulum.
  - Manipulation: Reacher, Pusher.
  - Fast: minimal test environment.

- [Gymnax](https://github.com/RobertTLange/gymnax#implemented-accelerated-environments-) — control, arcade, research tasks
  ([registry](https://github.com/RobertTLange/gymnax/blob/main/gymnax/registration.py)).
  - Classic control: Acrobot, CartPole, Pendulum, MountainCar, MountainCarContinuous.
  - MinAtar: Asterix, Breakout, Freeway, Seaquest, SpaceInvaders.
  - bsuite: Catch, DeepSea, MemoryChain, UmbrellaChain, DiscountingChain,
    MNISTBandit, SimpleBandit.
  - Other tasks: FourRooms, MetaMaze, PointRobot, BernoulliBandit,
    GaussianBandit, Reacher, Swimmer, Pong, FrozenLake.

- [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) — MuJoCo MJX robotics.
  - [DM Control suite](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/dm_control_suite/__init__.py):
    `AcrobotSwingup`, `AcrobotSwingupSparse`, `BallInCup`, `CartpoleBalance`,
    `CartpoleBalanceSparse`, `CartpoleSwingup`, `CartpoleSwingupSparse`, `CheetahRun`, `FingerSpin`,
    `FingerTurnEasy`, `FingerTurnHard`, `FishSwim`, `HopperHop`, `HopperStand`, `HumanoidStand`,
    `HumanoidWalk`, `HumanoidRun`, `PendulumSwingup`, `PointMass`, `ReacherEasy`, `ReacherHard`,
    `SwimmerSwimmer6`, `WalkerRun`, `WalkerStand`, `WalkerWalk`.
  - [Locomotion](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/locomotion/__init__.py):
    `ApolloJoystickFlatTerrain`, `BarkourJoystick`, `BerkeleyHumanoidJoystickFlatTerrain`,
    `BerkeleyHumanoidJoystickRoughTerrain`, `G1JoystickFlatTerrain`, `G1JoystickRoughTerrain`,
    `Go1JoystickFlatTerrain`, `Go1JoystickRoughTerrain`, `Go1Getup`, `Go1Handstand`, `Go1Footstand`,
    `H1InplaceGaitTracking`, `H1JoystickGaitTracking`, `Op3Joystick`, `SpotFlatTerrainJoystick`,
    `SpotGetup`, `SpotJoystickGaitTracking`, `T1JoystickFlatTerrain`, `T1JoystickRoughTerrain`.
  - [Manipulation](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/manipulation/__init__.py):
    `AlohaHandOver`, `AlohaSinglePegInsertion`, `PandaPickCube`, `PandaPickCubeOrientation`,
    `PandaPickCubeCartesian`, `PandaOpenCabinet`, `PandaRobotiqPushCube`, `LeapCubeReorient`,
    `LeapCubeRotateZAxis`, `AeroCubeRotateZAxis`.

- [Jumanji](https://github.com/instadeepai/jumanji#environments-) — puzzles/optimization
  ([registry](https://github.com/instadeepai/jumanji/blob/main/jumanji/__init__.py)).
  - Logic: Game2048, GraphColoring, Minesweeper, RubiksCube, SlidingTilePuzzle, Sudoku.
    Named variants: `RubiksCube-partly-scrambled-v0`, `Sudoku-very-easy-v0`.
  - Packing and scheduling: BinPack, FlatPack, JobShop, Knapsack, Tetris.
  - Routing: Cleaner, Connector, CVRP, MultiCVRP, Maze, RobotWarehouse,
    Snake, TSP, MMST (multiple minimum spanning trees), PacMan, Sokoban, LevelBasedForaging.
  - Swarm: SearchAndRescue.

- [Craftax](https://github.com/MichaelTMatthews/Craftax) — survival/crafting ([factory](https://github.com/MichaelTMatthews/Craftax/blob/main/craftax/craftax_env.py)).
  - Craftax: `Craftax-Symbolic-v1` and `Craftax-Pixels-v1`.
  - Craftax Classic (Crafter-style): `Craftax-Classic-Symbolic-v1`
    and `Craftax-Classic-Pixels-v1`.
  - Each has auto-reset and non-auto-reset implementations; the factory also accepts
    the corresponding `-AutoReset-v1` aliases when auto-reset is enabled.

- [Navix](https://github.com/epignatelli/navix) — MiniGrid tasks
  ([environment modules](https://github.com/epignatelli/navix/tree/main/navix/environments)).
  All names below have the prefix `Navix-` and suffix `-v0`.
  - Empty rooms: `Empty-5x5`, `Empty-6x6`, `Empty-8x8`, `Empty-16x16`, `Empty-Random-5x5`,
    `Empty-Random-6x6`, `Empty-Random-8x8`, `Empty-Random-16x16`.
  - Four rooms: `FourRooms`, `FourRooms-7x7`, `FourRooms-9x9`, `FourRooms-11x11`, `FourRooms-13x13`,
    `FourRooms-15x15`, `FourRooms-17x17`.
  - Multiple rooms: `MultiRoom-N2-S4`, `MultiRoom-N4-S5`, `MultiRoom-N6`.
  - Crossings: `SimpleCrossingS9N1`, `SimpleCrossingS9N2`, `SimpleCrossingS9N3`,
    `SimpleCrossingS11N5`, `LavaCrossingS9N1`, `LavaCrossingS9N2`, `LavaCrossingS9N3`,
    `LavaCrossingS11N5`.
  - Lava gaps: `LavaGapS5`, `LavaGapS6`, `LavaGapS7`.
  - Dynamic obstacles: `Dynamic-Obstacles-5x5`, `Dynamic-Obstacles-5x5-Random`,
    `Dynamic-Obstacles-6x6`, `Dynamic-Obstacles-6x6-Random`, `Dynamic-Obstacles-8x8`,
    `Dynamic-Obstacles-16x16`.
  - Distribution shifts: `DistShift1`, `DistShift2`.
  - Door/key: `DoorKey-5x5`, `DoorKey-6x6`, `DoorKey-8x8`, `DoorKey-16x16`, `DoorKey-Random-5x5`,
    `DoorKey-Random-6x6`, `DoorKey-Random-8x8`, `DoorKey-Random-16x16`.
  - Key corridors: `KeyCorridorS3R1`, `KeyCorridorS3R2`, `KeyCorridorS3R3`, `KeyCorridorS4R3`,
    `KeyCorridorS5R3`, `KeyCorridorS6R3`.
  - Locked room: `LockedRoom`.
  - Unlocking: `Unlock`, `UnlockPickup`, `BlockedUnlockPickup`.
  - Red/blue doors: `RedBlueDoors-6x6`, `RedBlueDoors-8x8`.
  - Obstructed mazes: `ObstructedMaze-1Dl`, `ObstructedMaze-1Dlh`, `ObstructedMaze-1Dlhb`,
    `ObstructedMaze-2Dl`, `ObstructedMaze-2Dlh`, `ObstructedMaze-2Dlhb`, `ObstructedMaze-1Q`,
    `ObstructedMaze-2Q`, `ObstructedMaze-Full`.
  - Go to door: `GoToDoor-5x5`, `GoToDoor-6x6`, `GoToDoor-8x8`.
  - Go to object: `GoToObject-6x6-N2`, `GoToObject-8x8-N2`.
  - Fetch: `Fetch-5x5-N2`, `Fetch-6x6-N2`, `Fetch-8x8-N3`.
  - Put near: `PutNear-6x6-N2`, `PutNear-8x8-N3`.
  - Memory: `MemoryS17Random`, `MemoryS13Random`, `MemoryS13`, `MemoryS11`, `MemoryS9`, `MemoryS7`.
  - Playground: `Playground`.

- [Kinetix](https://github.com/FLAIROx/Kinetix) — configurable 2D physics worlds.
  - `KinetixEnv`: procedural small, medium, and large worlds, plus custom levels.
    `KinetixDummyEnv` is a no-op testing variant
    ([factory](https://github.com/FLAIROx/Kinetix/blob/main/kinetix/environment/env.py)).
  - [Bundled levels](https://github.com/FLAIROx/Kinetix/tree/main/kinetix/levels):
    all 74 are also listed in the [evaluation set](https://github.com/FLAIROx/Kinetix/blob/main/configs/eval/eval_all.yaml).
    Names below are relative to each size directory.
  - Small (`s/`): `h0_weak_thrust`, `h1_thrust_over_ball`, `h2_one_wheel_car`,
    `h3_point_the_thruster`, `h4_thrust_aim`, `h5_rotate_fall`, `h6_unicycle_right`,
    `h7_unicycle_left`, `h8_unicycle_balance`, `h9_explode_then_thrust_over`.
  - Medium (`m/`): `arm_hard`, `arm_left`, `arm_right`, `arm_up`, `h0_unicycle`,
    `h10_thrust_right_easy`, `h11_thrust_left_easy`, `h12_thrustfall_left`, `h13_thrustfall_right`,
    `h14_thrustblock`, `h15_thrustshoot`, `h16_thrustcontrol_right`, `h17_thrustcontrol_left`,
    `h18_thrust_right_very_easy`, `h19_thrust_left_very_easy`, `h1_car_left`, `h2_car_right`,
    `h3_car_thrust`, `h4_thrust_the_needle`, `h5_angry_birds`, `h6_thrust_over`, `h7_car_flip`,
    `h8_weird_vehicle`, `h9_spin_the_right_way`.
  - Large (`l/`): `car_launch`, `car_swing_around`, `chain_lander`, `chain_thrust`, `gears`,
    `grasp_easy`, `grasp_hard`, `h0_angrybirds`, `h10_morph_direction_2`, `h11_obstacle_avoidance`,
    `h12_platformer_1`, `h13_platformer_2`, `h14_simple_thruster`, `h15_swing_up`,
    `h16_thruster_goal`, `h17_unicycle`, `h1_car_left`, `h2_car_ramp`, `h3_car_right`,
    `h4_cartpole`, `h5_flappy_bird`, `h6_lorry`, `h7_maze_1`, `h8_maze_2`, `h9_morph_direction`,
    `hard_beam_balance`, `hard_cartpole_thrust`, `hard_cartpole_wheels`, `hard_lunar_lander`,
    `hard_pinball`, `lever_puzzle`, `mjc_half_cheetah`, `mjc_half_cheetah_easy`, `mjc_hopper`,
    `mjc_hopper_easy`, `mjc_swimmer`, `mjc_walker`, `mjc_walker_easy`, `pr`, `rail`.

- [JAXtari](https://github.com/k4ntz/JAXtari) — 70 registered Atari game environments
  ([registry](https://github.com/k4ntz/JAXtari/blob/master/src/jaxtari/core.py), [game status list](https://github.com/k4ntz/JAXtari/blob/master/games_covered.md)).
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
  - `ForagaxSquareWaveTwoBiome-v11`: 24×15 toroidal world, two mushroom biomes and walls.
  - `ForagaxTwoBiomeLarge-v1`: 15×15 Morel (+30)/Oyster (+1) biomes, Deathcaps (−1).
  - Custom worlds: objects, biomes, respawning, rewards; programmable weather-driven rewards.
