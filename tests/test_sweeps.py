from pathlib import Path

import pytest
from omegaconf import OmegaConf

from rl2.sweeps import load_sweep


@pytest.fixture
def sweep_path(tmp_path: Path) -> Path:
    (tmp_path / "base.yaml").write_text(
        "seed: 1\nlearning_rate: 0.001\nenv: {period: 512}\nrun_id: old_seed${seed}\nckpt_dir: /runs/ppo/${run_id}\n"
    )
    path = tmp_path / "sweep.yaml"
    path.write_text("name: example\nconfig: base.yaml\nparameters:\n  seed: [1, 2]\n  env.period: [512, 1024]\n")
    return path


def test_grid_expansion_and_stable_resume(sweep_path: Path) -> None:
    sweep = load_sweep(path=sweep_path, root=sweep_path.parent)
    assert sweep.name == "example"
    assert {(trial.seed, trial.env.period) for trial in sweep.trials} == {(1, 512), (1, 1024), (2, 512), (2, 1024)}
    identities = {(trial.seed, trial.env.period): trial.run_id for trial in sweep.trials}
    assert len(set(identities.values())) == 4
    for trial in sweep.trials:
        assert trial.run_id.startswith("example_")
        assert trial.ckpt_dir == f"/runs/ppo/{trial.run_id}"
        assert trial.learning_rate == 0.001
    assert load_sweep(path=sweep_path, root=sweep_path.parent) == sweep
    sweep_path.write_text(
        "name: example\nconfig: base.yaml\nparameters:\n  env.period: [1024, 512]\n  seed: [2, 1, 3]\n"
    )
    extended = load_sweep(path=sweep_path, root=sweep_path.parent)
    assert len(extended.trials) == 6
    assert all(identities[(t.seed, t.env.period)] == t.run_id for t in extended.trials if t.seed != 3)
    base = OmegaConf.load(sweep_path.parent / "base.yaml")
    base.learning_rate = 0.002
    OmegaConf.save(base, sweep_path.parent / "base.yaml")
    changed = load_sweep(path=sweep_path, root=sweep_path.parent)
    assert not set(identities.values()) & {trial.run_id for trial in changed.trials}


@pytest.mark.parametrize("directory", [None, "/runs/fixed"], ids=["missing-directory", "literal-directory"])
def test_checkpoint_defaults_and_duplicate_values(sweep_path: Path, directory: str | None) -> None:
    base: dict[str, object] = {"seed": 1}
    if directory is not None:
        base["ckpt_dir"] = directory
    OmegaConf.save(OmegaConf.create(base), sweep_path.parent / "base.yaml")
    sweep_path.write_text("config: base.yaml\nparameters: {seed: [1, 1, 2]}\n")
    sweep = load_sweep(path=sweep_path, root=sweep_path.parent)
    assert sweep.name == "sweep"
    assert len(sweep.trials) == 2
    assert all(t.ckpt_dir == (directory if directory is not None else f"/runs/{t.run_id}") for t in sweep.trials)


def test_checkpoint_override_preserves_interpolation_and_identity(sweep_path: Path) -> None:
    original = load_sweep(path=sweep_path, root=sweep_path.parent)
    template = "gs://bucket/${run_id}/seed${seed}"
    with sweep_path.open("a") as stream:
        stream.write(f"ckpt_dir: {template}\n")
    sweep = load_sweep(path=sweep_path, root=sweep_path.parent)
    assert [t.run_id for t in sweep.trials] == [t.run_id for t in original.trials]
    for trial in sweep.trials:
        assert OmegaConf.to_container(trial, resolve=False)["ckpt_dir"] == template
        assert trial.ckpt_dir == f"gs://bucket/{trial.run_id}/seed{trial.seed}"


@pytest.mark.parametrize("directory", ["null", "''", "42"], ids=["null", "empty", "number"])
def test_invalid_checkpoint_override(sweep_path: Path, directory: str) -> None:
    with sweep_path.open("a") as stream:
        stream.write(f"ckpt_dir: {directory}\n")
    with pytest.raises(ValueError, match="ckpt_dir must be a nonempty string"):
        load_sweep(path=sweep_path, root=sweep_path.parent)


def test_structured_parameter_values_replace_base(sweep_path: Path) -> None:
    sweep_path.write_text("config: base.yaml\nparameters: {env: [{id: replacement}], layers: [[16, 8]]}\n")
    trial = load_sweep(path=sweep_path, root=sweep_path.parent).trials[0]
    assert OmegaConf.to_container(trial.env) == {"id": "replacement"}
    assert trial.layers == [16, 8]


@pytest.mark.parametrize(
    "contents,match",
    [
        ("[]", "mapping"),
        ("config: base.yaml\nparameter: {}", "mapping"),
        ("name: ../invalid\nconfig: base.yaml", "name"),
        ("parameters: {seed: [1]}", "file path"),
        ("config: base.yaml\nparameters: {}", "nonempty mapping"),
        ("config: base.yaml\nparameters: {seed: []}", "nonempty list"),
        ("config: base.yaml\nparameters: {seed: 1}", "nonempty list"),
        ("config: base.yaml\nparameters: {run_id: [a]}", "controls run_id"),
        ("config: base.yaml\nparameters: {ckpt_dir: [/runs]}", "controls run_id"),
        ("config: base.yaml\nparameters: {env: [{}], env.period: [1]}", "Overlapping"),
        ("config: base.yaml\nparameters: {env..period: [1]}", "dotted paths"),
        ("config: base.yaml\nparameters: {seed: [1], 7: [2]}", "dotted paths"),
    ],
)
def test_invalid_sweep(sweep_path: Path, contents: str, match: str) -> None:
    sweep_path.write_text(contents)
    with pytest.raises(ValueError, match=match):
        load_sweep(path=sweep_path, root=sweep_path.parent)


def test_non_mapping_base_config(sweep_path: Path) -> None:
    (sweep_path.parent / "base.yaml").write_text("[]\n")
    with pytest.raises(TypeError, match="Base config must be a mapping"):
        load_sweep(path=sweep_path, root=sweep_path.parent)
