"""Play procedurally generated Reasoning Gym tasks in the terminal."""

import argparse
import json
from dataclasses import asdict

import reasoning_gym
from reasoning_gym.dataset import ProceduralDataset
from reasoning_gym.factory import DATASETS

HELP = "Commands: /skip reveals the answer, /multi starts a multiline answer, /help, /quit."


def read_answer() -> str:
    """Read one answer, preserving whitespace in multiline submissions."""
    while True:
        answer = input("Answer> ")
        if answer.strip() == "/help":
            print(HELP)
        elif answer.strip() == "/multi":
            print("Enter your answer; finish with /submit on its own line. /quit exits.")
            lines: list[str] = []
            while True:
                line = input("... ")
                if line == "/submit":
                    break
                if line == "/quit":
                    return "/quit"
                lines.append(line)
            if lines and any(line.strip() for line in lines):
                return "\n".join(lines)
            print("Answer cannot be empty. Use /skip to reveal the answer.")
        elif answer.strip():
            return answer.strip()
        else:
            print("Answer cannot be empty. Use /skip to reveal the answer.")


def play(dataset: ProceduralDataset) -> None:
    """Score each submitted answer once and summarize the session on exit."""
    attempted = 0
    skipped = 0
    total_score = 0.0
    print(HELP)
    try:
        for index in range(len(dataset)):
            entry = dataset[index]
            print(f"\nQuestion {index + 1}/{len(dataset)}\n\n{entry['question']}\n")
            answer = read_answer()
            if answer == "/quit":
                break
            if answer == "/skip":
                skipped += 1
            else:
                score = dataset.score_answer(answer=answer, entry=entry)
                attempted += 1
                total_score += score
                print(f"Score: {score:.3f}/1")
            print(f"Reference answer:\n{entry['answer']}")
    except (EOFError, KeyboardInterrupt):
        print()
    finally:
        mean = total_score / attempted if attempted else 0.0
        print(f"\nSession: {attempted} answered, {skipped} skipped; mean score {mean:.3f}/1 (answered only).")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", nargs="?", help="Dataset name; prompts when omitted (default: countdown)")
    parser.add_argument("--list", action="store_true", help="List available tasks and exit")
    parser.add_argument("--describe", action="store_true", help="Show task description and default config, then exit")
    parser.add_argument("--seed", type=int, default=42, help="Reproducible question seed (default: 42)")
    parser.add_argument("--size", type=int, default=10, help="Number of questions (default: 10)")
    parser.add_argument("--config", default="{}", help="Task options as a JSON object, e.g. '{\"min_numbers\": 3}'")
    args = parser.parse_args(argv)
    if args.list:
        print("\n".join(sorted(DATASETS)))
        return
    if args.size < 1:
        parser.error("--size must be positive")
    task = args.task
    if task is None:
        print("Try countdown, leg_counting, or use --list to discover all tasks.")
        try:
            task = input("Task [countdown]> ").strip() or "countdown"
        except (EOFError, KeyboardInterrupt):
            print()
            return
    if task not in DATASETS:
        parser.error(f"Unknown task {task!r}; use --list to see available tasks")
    if args.describe:
        dataset_cls, config_cls = DATASETS[task]
        print(dataset_cls.__doc__ or task)
        print(json.dumps(asdict(config_cls()), indent=2, default=str))
        return
    try:
        config = json.loads(args.config)
        if not isinstance(config, dict):
            raise TypeError("--config must be a JSON object")
        if {"seed", "size"} & config.keys():
            raise ValueError("Use --seed and --size instead of setting them in --config")
        dataset = reasoning_gym.create_dataset(task, seed=args.seed, size=args.size, **config)
    except (ValueError, TypeError, AssertionError) as exc:
        parser.error(f"Invalid task configuration: {exc or 'task constraints were not satisfied; see --describe'}")
    print(f"\nTask: {task} | seed: {args.seed} | questions: {args.size}")
    play(dataset)


if __name__ == "__main__":
    main()
