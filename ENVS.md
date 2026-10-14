# JAX environments

Environment families offered by the 14 projects marked **(jax env)** in
[README.md](README.md). Upstream documentation and registries checked on
2026-10-14; size, difficulty, and layout variants are grouped together.
Availability here refers to upstream projects; integration into rl2 varies.

- [JaxMARL](https://github.com/bold-lab-ai/JaxMARL) — multi-agent environments ([registry](https://github.com/bold-lab-ai/JaxMARL/blob/main/jaxmarl/registration.py)).
  - MPE: Simple, Tag, World Comm, Spread, Crypto, Speaker Listener, Push,
    Adversary, Reference, and FACMAC predator–prey variants.
  - Overcooked and OvercookedV2: cooperative cooking and coordination.
  - SMAX: simplified StarCraft combat, including heuristic and learned-policy enemies.
  - Hanabi: cooperative card play with hidden information.
  - STORM: spatial matrix games, including two-player and multi-player variants.
  - Coin Game: resource collection and social dilemmas.
  - Switch Riddle: cooperative communication puzzle.
  - JaxNav: navigation and collision avoidance for multiple robots.
  - Multi-Agent Brax: Ant, HalfCheetah, Hopper, Humanoid, Walker2d; requires the optional Brax dependency.
  - JaxRobotarium: Navigation, Discovery, Material Transport, Warehouse,
    Arctic Transport, Foraging, RWARE, Predator Prey; requires its optional package.

- [Pgx](https://github.com/sotetsuk/pgx) — board, card, and arcade games ([registry](https://github.com/sotetsuk/pgx/blob/main/pgx/core.py)).
  - Board games: Backgammon, Chess, Gardner Chess, Shogi, Animal Shogi,
    Go (9×9 and 19×19), Hex, Othello, Connect Four, Tic-Tac-Toe.
  - Card/tile games: Kuhn Poker, Leduc Hold'em, Sparrow Mahjong.
  - Bridge Bidding: available through the `BridgeBidding` class directly.
  - Puzzle: 2048.
  - MinAtar: Asterix, Breakout, Freeway, Seaquest, Space Invaders.

- [XLand-MiniGrid](https://github.com/dunnolab/xland-minigrid#environments-) — grid worlds and meta-RL.
  - XLand-MiniGrid: configurable layouts, sizes, goals, and transformation rules;
    tasks are sampled from ruleset benchmarks.
  - Navigation: Empty, EmptyRandom, FourRooms.
  - Keys and doors: LockedRoom, DoorKey, Unlock, UnlockPickUp, BlockedUnlockPickUp.
  - Memory: remember an initial cue and select the matching object.
  - Playground: open-ended object interaction without a specified goal.

- [minimax](https://github.com/facebookresearch/minimax) — maze environments for automatic curricula.
  - Maze: navigation through configurable grid layouts, with generated training
    mazes and held-out evaluation layouts.
  - Maze UED: a teacher environment that constructs mazes for the navigation agent.

- [JaxUED](https://github.com/DramaCow/jaxued#supported-environments) — environments and adapters for curriculum learning.
  - Maze: generated and fixed navigation levels, with level editing for curriculum generation.
  - Craftax: the supplied curriculum example supports symbolic Craftax observations.
  - Gymnax adapters: CartPole, Acrobot, Pendulum, with varying physical parameters.

- [Brax](https://github.com/google/brax) — continuous control ([registry](https://github.com/google/brax/blob/main/brax/envs/__init__.py)).
  - Locomotion: Ant, HalfCheetah, Hopper, Humanoid, Swimmer, Walker2d.
  - Posture and balance: HumanoidStandup, InvertedPendulum, InvertedDoublePendulum.
  - Manipulation: Reacher, Pusher.
  - Fast: a minimal environment for testing.

- [Gymnax](https://github.com/RobertTLange/gymnax#implemented-accelerated-environments-) — classic control, arcade, and small research tasks.
  - Classic control: Acrobot, CartPole, Pendulum, MountainCar, MountainCarContinuous.
  - MinAtar: Asterix, Breakout, Freeway, Seaquest, SpaceInvaders.
  - bsuite: Catch, DeepSea, MemoryChain, UmbrellaChain, DiscountingChain,
    MNISTBandit, SimpleBandit.
  - Other tasks: FourRooms, MetaMaze, PointRobot, BernoulliBandit,
    GaussianBandit, Reacher, Swimmer.

- [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) — robot control with MuJoCo MJX.
  - [DM Control suite](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/dm_control_suite/__init__.py):
    Acrobot swing-up; BallInCup; Cartpole balance/swing-up; Cheetah run;
    Finger spin/turn; Fish swim; Hopper hop/stand; Humanoid stand/walk/run;
    Pendulum swing-up; PointMass; Reacher easy/hard; six-link Swimmer;
    Walker stand/walk/run. Selected tasks also have sparse-reward variants.
  - [Locomotion](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/locomotion/__init__.py):
    joystick control for Apollo, Barkour, Berkeley Humanoid, G1, Go1, OP3,
    Spot, and T1; flat/rough terrain variants for selected robots;
    Go1 get-up/handstand/footstand; H1 in-place and joystick gait tracking;
    Spot get-up and gait tracking.
  - [Manipulation](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/manipulation/__init__.py):
    Aloha handover and peg insertion; Panda cube picking (standard,
    orientation, Cartesian), cabinet opening, and Robotiq cube pushing;
    LEAP cube reorientation and Z-axis rotation; Aero cube Z-axis rotation.

- [Jumanji](https://github.com/instadeepai/jumanji#environments-) — puzzles and combinatorial optimization.
  - Logic: Game2048, GraphColoring, Minesweeper, RubiksCube, SlidingTilePuzzle, Sudoku.
  - Packing and scheduling: BinPack, FlatPack, JobShop, Knapsack, Tetris.
  - Routing: Cleaner, Connector, CVRP, MultiCVRP, Maze, RobotWarehouse,
    Snake, TSP, MMST (multiple minimum spanning trees), PacMan, Sokoban.

- [Craftax](https://github.com/MichaelTMatthews/Craftax) — procedural survival, crafting, and exploration ([factory](https://github.com/MichaelTMatthews/Craftax/blob/main/craftax/craftax_env.py)).
  - Craftax: expanded roguelike world; `Craftax-Symbolic-v1` and `Craftax-Pixels-v1`.
  - Craftax Classic: Crafter-style world; `Craftax-Classic-Symbolic-v1`
    and `Craftax-Classic-Pixels-v1`.

- [Navix](https://github.com/epignatelli/navix) — MiniGrid task families ([environment modules](https://github.com/epignatelli/navix/blob/main/navix/environments/__init__.py)).
  - Navigation: Empty rooms, FourRooms, MultiRoom, Crossings, LavaGap,
    DynamicObstacles, DistShift.
  - Keys and doors: DoorKey, KeyCorridor, LockedRoom, Unlock, UnlockPickup,
    BlockedUnlockPickup, RedBlueDoors, ObstructedMaze.
  - Object tasks: GoToDoor, GoToObject, Fetch, PutNear.
  - Memory and exploration: Memory, Playground.

- [Kinetix](https://github.com/FLAIROx/Kinetix) — configurable 2D physics tasks.
  - Procedural levels: small, medium, and large worlds controlled through motors
    and thrusters; bring goal objects together while avoiding hazards.
  - [Handmade levels](https://github.com/FLAIROx/Kinetix/blob/main/configs/eval/eval_all.yaml):
    cars, unicycles, thruster control, robot arms, grasping, cartpole, balance,
    lunar landing, pinball, Angry Birds, Flappy Bird, mazes, platformers,
    obstacle avoidance, gears, and lever puzzles.
  - Locomotion levels: HalfCheetah, Hopper, Swimmer, Walker variants within the same physics framework.

- [JAXtari](https://github.com/k4ntz/JAXtari) — Atari reimplementations ([game status list](https://github.com/k4ntz/JAXtari/blob/master/games_covered.md)).
  - Atari57 games rated close to the originals: Alien, Amidar, Asterix, Asteroids,
    Atlantis, Bank Heist, Beam Rider, Berzerk, Boxing, Breakout, Centipede,
    Chopper Command, Enduro, Fishing Derby, Freeway, Frostbite, Gopher,
    Gravitar, Kangaroo, Montezuma's Revenge, Ms. Pac-Man, Name This Game,
    Phoenix, Pong, Q*bert, River Raid, Seaquest, Skiing, Space Invaders,
    Surround, Tennis, Time Pilot, Venture, Video Pinball, Yars' Revenge.
  - Registered with substantial differences: Assault, Road Runner, Up N Down, Wizard of Wor.
  - Additional games: Air Raid, Backgammon, Basic Math, Blackjack,
    Casino Blackjack/Five Stud Poker/Poker Solitaire, Donkey Kong, Flag Capture,
    Galaxian, Hangman, Haunted House, Human Cannonball, Journey Escape, Kaboom,
    King Kong, Klax, Laser Gates, Miniature Golf, Othello, Pac-Man,
    Sir Lancelot, Slot Machine, Space War, Tetris, Tron, Turmoil,
    Video Checkers, Video Chess, Video Cube, Word Zapper.
  - Games marked work-in-progress or unimplemented upstream are omitted.

- [continual-foragax](https://pypi.org/project/continual-foragax/#registry-and-included-environments) — foraging with changing rewards.
  - `ForagaxBig-v5`: large multi-biome world with Fourier-modulated rewards.
  - `ForagaxSquareWaveTwoBiome-v11`: two biomes with square-wave reward changes.
  - `ForagaxTwoBiomeLarge-v1`: 15×15 Morel/Oyster biomes containing Deathcaps.
  - Custom worlds: configurable objects, biomes, respawning, and rewards;
    weather-driven rewards can be composed programmatically.
