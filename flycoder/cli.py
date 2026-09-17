"""Command-line interface for FLY-CODER."""

import argparse
from pathlib import Path

from flycoder.agent import FlyCoderAgent
from flycoder.state import CodingState


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""

    parser = argparse.ArgumentParser(
        prog="flycoder",
        description="FLY-CODER autonomous coding assistant",
    )

    parser.add_argument(
        "workspace",
        nargs="?",
        default="playground",
        help="Workspace directory to inspect",
    )

    parser.add_argument(
        "--task",
        default="Inspect and test the workspace",
        help="Coding task for the agent",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=20,
        help="Maximum number of agent steps",
    )

    parser.add_argument(
        "--approve",
        action="store_true",
        help="Automatically approve a proposed repair",
    )

    return parser


def print_repair_proposal(result) -> None:
    """Print a repair proposal."""

    data = result.data or {}

    print()
    print("Repair proposal:")
    print(f"  File: {data.get('file', 'Unknown')}")
    print(
        "  Description: "
        f"{data.get('description', 'No description')}"
    )
    print()
    print("Proposed content:")
    print("-" * 60)
    print(data.get("proposed_content", ""))
    print("-" * 60)


def main() -> None:
    """Run the FLY-CODER agent from the terminal."""

    parser = build_parser()
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()

    if not workspace.exists():
        raise SystemExit(
            f"Workspace does not exist: {workspace}"
        )

    agent = FlyCoderAgent(workspace)
    state = CodingState(
        task=args.task,
        task_intent=agent.classify_task(args.task),
    )


    print(f"Workspace: {workspace}")
    print(f"Task: {state.task}")
    print(f"Task Intent: {state.task_intent}")
    print()

    for step in range(args.max_steps):
        result = agent.run_once(state)

        print(
            f"[Step {step}] "
            f"{result.action} | "
            f"success={result.success}"
        )
        print(f"  {result.message}")

        if result.action == "inspect_files" and result.data:
            print("  Files:")
            for file_path in result.data["files"]:
                print(f"    - {file_path}")

        elif result.action == "run_tests" and result.data:
            print("  Test output:")
            print(result.data["output"])

        elif result.action == "propose_repair":
            print_repair_proposal(result)

            if args.approve:
                print()
                print("Approval enabled.")
                print("Applying repair...")

                approval_result = agent.actions.execute(
                    "approve_repair",
                    workspace=agent.workspace,
                    state=state,
                )

                print(
                    f"[Approval] "
                    f"success={approval_result.success}"
                )
                print(
                    f"  {approval_result.message}"
                )

                if not approval_result.success:
                    state.finished = True
                    break

                continue

            print()
            print(
                "Repair requires approval. "
                "Re-run with --approve to apply it."
            )
            break

        if state.finished:
            break

    print()
    print("Final state:")
    print(f"  Task: {state.task}")
    print(f"  Tests run: {state.tests_run}")
    print(f"  Tests passed: {state.tests_passed}")
    print(f"  Repair proposed: {state.repair_proposed}")
    print(f"  Repair approved: {state.repair_approved}")
    print(f"  Repair applied: {state.repair_applied}")
    print(f"  Finished: {state.finished}")


if __name__ == "__main__":
    main()