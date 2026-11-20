"""Load static collision geometry in a process without cameras or GPU rendering."""

from contextlib import closing

import habitat_sim

from rl2.envs.visfly.catalogues import PlannedCatalogue, PlanningTask, build_catalogue
from rl2.envs.visfly.collision import SceneCollision
from rl2.envs.visfly.sampling import sampling_bounds


def plan_scene(task: PlanningTask) -> PlannedCatalogue:
    settings = habitat_sim.SimulatorConfiguration()
    settings.scene_id = task.scene_path
    settings.scene_dataset_config_file = task.dataset_path
    settings.random_seed = task.scene_seed
    settings.enable_physics = True
    settings.use_semantic_textures = False
    settings.load_semantic_mesh = False
    settings.force_separate_semantic_scene_graph = False
    settings.create_renderer = False
    config = habitat_sim.Configuration(settings, [habitat_sim.agent.AgentConfiguration()])
    with closing(habitat_sim.Simulator(config)) as scene:
        bounds = sampling_bounds(
            scene.get_active_scene_graph().get_root_node().cumulative_bb, task.clearance, task.height
        )
        return build_catalogue(task, bounds, SceneCollision(scene=scene))
