"""Command-line interface for FLY-CODER."""

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from flycoder.agent import FlyCoderAgent
from flycoder.state import CodingState
from flycoder.tools.completion_evidence import evaluate_completion_evidence
from flycoder.tools.completion_outcome import (
    classify_stopped_run,
    verification_attempt,
)
from flycoder.tools.human_input import awaiting_repair_approval
from flycoder.tools.human_intervention import (
    DECISION_APPROVE,
    apply_human_decision,
    build_intervention_request,
    prompt_for_approval,
)


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
        help=(
            "Review the exact proposed diff and confirm it interactively "
            "before it is applied. Never approves without confirmation."
        ),
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


def main(
    argv: list[str] | None = None,
    *,
    input_fn: Callable[[str], str] = input,
    is_interactive: bool | None = None,
) -> None:
    """Run the FLY-CODER agent from the terminal.

    ``input_fn`` and ``is_interactive`` exist so the interactive review
    can be exercised; by default approval requires a real terminal.
    """

    parser = build_parser()
    args = parser.parse_args(argv)

    interactive = (
        sys.stdin.isatty() if is_interactive is None else is_interactive
    )

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

    verification_attempts = []
    result = None
    budget_exhausted = False

    for step in range(args.max_steps):
        result = agent.run_once(state)

        attempt = verification_attempt(step, result)

        if attempt is not None:
            verification_attempts.append(attempt)

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

        elif (
            result.action == "propose_repair"
            and awaiting_repair_approval(state)
        ):
            # Phase 10.7: the human reviews the exact proposal. Approval
            # is bound to its fingerprint and applied by the next
            # run_once through approve_repair and the guardrails.
            request = build_intervention_request(
                state,
                evidence=evaluate_completion_evidence(state),
            )

            if not args.approve:
                print()
                print(request.render())
                print()
                print(
                    "Repair requires approval. "
                    "Re-run with --approve to review and confirm it."
                )
                break

            print()
            decision = prompt_for_approval(
                request,
                input_fn=input_fn,
                interactive=interactive,
            )

            if decision is None:
                break

            record = apply_human_decision(state, request, decision)

            print(f"[Decision] {record.describe()}")

            if decision.kind != DECISION_APPROVE:
                break

            continue

        if state.finished:
            break
    else:
        # The step budget ran out without the agent concluding the task.
        budget_exhausted = not state.finished

    print()
    print("Final state:")
    print(f"  Task: {state.task}")
    print(f"  Tests run: {state.tests_run}")
    print(f"  Tests passed: {state.tests_passed}")
    print(f"  Repair proposed: {state.repair_proposed}")
    print(f"  Repair approved: {state.repair_approved}")
    print(f"  Repair applied: {state.repair_applied}")
    print(f"  Finished: {state.finished}")

    # Phase 10.8: report how the task ended, from evidence. A stopped or
    # finished agent is not success unless the work is verified.
    outcome = classify_stopped_run(
        state,
        intent=state.task_intent,
        budget_exhausted=budget_exhausted,
        verification_attempts=verification_attempts,
        final_result=result,
    )

    print()
    print(outcome.render())


if __name__ == "__main__":
    main()