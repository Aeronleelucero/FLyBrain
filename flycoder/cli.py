"""Command-line interface for FLY-CODER.

Phase 10.9: both command-line entry points (``main`` here and
``flycoder.agent.cli``) run their task through a ``TaskSession`` using
``run_cli_session``. The session is the single orchestration controller:
execution goes through ``run_autonomous_task`` and ``agent.run_once``
(guardrails, then the action registry), and human decisions go through
``TaskSession.decide``. Each interface keeps its own output format.
"""

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from flycoder.agent import FlyCoderAgent
from flycoder.state import CodingState
from flycoder.tools.autonomous import AutonomousStep
from flycoder.tools.human_intervention import (
    DecisionRecord,
    HumanDecisionError,
    InterventionRequest,
    prompt_for_approval,
)
from flycoder.tools.task_session import TaskSession

# Compatibility re-exports (Phase 10.9): these names were importable from
# flycoder.cli before the CLI moved to TaskSession. They are not used by
# this module's execution flow.
from flycoder.tools.completion_evidence import (  # noqa: F401
    evaluate_completion_evidence,
)
from flycoder.tools.completion_outcome import (  # noqa: F401
    classify_stopped_run,
    verification_attempt,
)
from flycoder.tools.human_input import awaiting_repair_approval  # noqa: F401
from flycoder.tools.human_intervention import (  # noqa: F401
    DECISION_APPROVE,
    apply_human_decision,
    build_intervention_request,
)


CLI_CLOSE_REASON = (
    "The command-line run ended; this interface keeps no session to resume."
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


# ==================================================================
# SHARED SESSION DRIVER (Phase 10.9)
# ==================================================================


def run_cli_session(
    session: TaskSession,
    *,
    approve: bool,
    input_fn: Callable[[str], str],
    interactive: bool,
    print_step: Callable[[AutonomousStep], None],
    print_decision: Callable[[DecisionRecord, int], None],
    finish: Callable[[TaskSession], None],
    on_unapproved: Callable[[InterventionRequest], None] | None = None,
    before_review: Callable[[], None] | None = None,
    reviewable: Callable[[TaskSession], bool] | None = None,
) -> TaskSession:
    """Drive one command-line task through ``session``.

    1. ``session.run`` executes the task; ``print_step`` prints each step
       as it is recorded.
    2. While the session is paused for repair approval (and
       ``reviewable`` allows it): without ``approve`` the run stops after
       ``on_unapproved``; otherwise the human reviews the exact proposal
       with ``prompt_for_approval`` (which fails closed), the decision
       goes through ``session.decide``, and the task resumes through the
       normal autonomous workflow. An approval is applied and verified;
       a rejection discards the proposal and the agent continues safely.
    3. The session is always closed (this interface keeps no session to
       resume) and ``finish`` prints the final report, also when the run
       is interrupted. KeyboardInterrupt/SystemExit then propagate as the
       original objects.

    The caller constructs the session, so the handle stays available
    even when ``run`` is interrupted.
    """

    interrupted = False

    try:
        session.run(on_step=print_step)

        while True:
            request = session.intervention_request()

            if request is None or not request.is_repair_approval:
                break

            if reviewable is not None and not reviewable(session):
                break

            if not approve:
                if on_unapproved is not None:
                    on_unapproved(request)
                break

            if before_review is not None:
                before_review()

            decision = prompt_for_approval(
                request,
                input_fn=input_fn,
                interactive=interactive,
            )

            if decision is None:
                break

            try:
                record = session.decide(decision)
            except HumanDecisionError as error:
                print(f"Decision not accepted: {error}")
                break

            print_decision(record, len(session.steps) - 1)

            if session.cancelled or session.remaining_iterations < 1:
                break

            session.resume(on_step=print_step)

    except (KeyboardInterrupt, SystemExit):
        interrupted = True
        raise

    finally:
        session.close(CLI_CLOSE_REASON)

        try:
            finish(session)
        except Exception:
            # Never let a reporting problem replace an interrupt.
            if not interrupted:
                raise

    return session


def render_session_outcome(session: TaskSession) -> str:
    """Return the session's outcome report for a command-line summary."""

    if session.outcome is not None:
        return session.outcome.render()

    lines = [
        "Outcome: unavailable — no completion outcome could be classified.",
        "Successful: no",
    ]

    error = session.result.error if session.result is not None else None

    if error is not None:
        lines.append(f"Error: {error.describe()}")

    return "\n".join(lines)


# ==================================================================
# flycoder.cli OUTPUT
# ==================================================================


def _print_step(step: AutonomousStep) -> None:
    result = step.result
    data = result.data if isinstance(result.data, dict) else {}

    print(
        f"[Step {step.iteration - 1}] "
        f"{result.action} | "
        f"success={result.success}"
    )
    print(f"  {result.message}")

    if result.action == "inspect_files" and "files" in data:
        print("  Files:")
        for file_path in data["files"]:
            print(f"    - {file_path}")

    elif result.action == "run_tests" and "output" in data:
        print("  Test output:")
        print(data["output"])


def _print_unapproved(request: InterventionRequest) -> None:
    print()
    print(request.render())
    print()
    print(
        "Repair requires approval. "
        "Re-run with --approve to review and confirm it."
    )


def _proposal_just_made(session: TaskSession) -> bool:
    # This interface reviews a proposal right after propose_repair.
    return bool(session.steps) and session.steps[-1].result.action == (
        "propose_repair"
    )


def _print_final_state(session: TaskSession) -> None:
    state = session.state

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
    print()
    print(render_session_outcome(session))


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

    if args.max_steps < 1:
        parser.error("--max-steps must be at least 1")

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

    session = TaskSession(agent, state, max_iterations=args.max_steps)

    run_cli_session(
        session,
        approve=args.approve,
        input_fn=input_fn,
        interactive=interactive,
        print_step=_print_step,
        print_decision=lambda record, _index: print(
            f"[Decision] {record.describe()}"
        ),
        finish=_print_final_state,
        on_unapproved=_print_unapproved,
        before_review=print,
        reviewable=_proposal_just_made,
    )


if __name__ == "__main__":
    main()
