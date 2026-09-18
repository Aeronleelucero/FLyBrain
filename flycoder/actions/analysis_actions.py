"""Code analysis actions for FLY-CODER."""

from flycoder.actions.registry import ActionResult
from flycoder.state import CodingState
from flycoder.tools.filesystem import Workspace


def explain_error_action(
    workspace: Workspace,
    state: CodingState,
) -> ActionResult:
    """Explain the latest test error."""

    if not state.last_error:
        return ActionResult(
            action="explain_error",
            success=True,
            message="No test error is available.",
            data={
                "explanation": (
                    "The latest test run passed. "
                    "There is no failing error to explain."
                ),
            },
        )

    explanation = (
        "The test suite reported a failure.\n\n"
        "Latest error output:\n"
        f"{state.last_error[-3000:]}\n\n"
        "Recommended next steps:\n"
        "1. Identify the failing test.\n"
        "2. Read the affected source file.\n"
        "3. Compare expected and actual behavior.\n"
        "4. Create a repair proposal.\n"
        "5. Run the tests again after approval."
    )

    state.error_inspected = True

    return ActionResult(
        action="explain_error",
        success=True,
        message="Error explanation created.",
        data={
            "explanation": explanation,
        },
    )


def review_code_action(
    workspace: Workspace,
    state: CodingState,
) -> ActionResult:
    """Perform a basic static review of the selected Python file."""

    if not state.current_file:
        return ActionResult(
            action="review_code",
            success=False,
            message="No file is selected for review.",
        )

    if state.current_file_content is None:
        return ActionResult(
            action="review_code",
            success=False,
            message="The selected file has not been read.",
        )

    content = state.current_file_content
    lines = content.splitlines()

    findings: list[str] = []

    if len(lines) > 300:
        findings.append(
            "The file is longer than 300 lines. "
            "Consider splitting it into smaller modules."
        )

    if "except Exception:" in content:
        findings.append(
            "Broad exception handling detected. "
            "Consider catching specific exception types."
        )

    if "print(" in content:
        findings.append(
            "print() statements detected. "
            "Consider using the logging module for application output."
        )

    if "TODO" in content or "FIXME" in content:
        findings.append(
            "TODO or FIXME comments detected. "
            "Review these before production use."
        )

    if not findings:
        findings.append(
            "No basic static-review issues were detected "
            "by the current rules."
        )

    # Store the result for the final project-level report.
    state.review_findings.append(
        {
            "file": state.current_file,
            "lines": len(lines),
            "findings": findings,
        }
    )

    review = "\n".join(
        f"{index}. {finding}"
        for index, finding in enumerate(findings, start=1)
    )

    return ActionResult(
        action="review_code",
        success=True,
        message="Code review completed.",
        data={
            "file": state.current_file,
            "lines": len(lines),
            "findings": findings,
            "review": review,
        },
    )


def improve_code_action(
    workspace: Workspace,
    state: CodingState,
) -> ActionResult:
    """Create a basic improvement proposal without modifying files."""

    if not state.current_file:
        return ActionResult(
            action="improve_code",
            success=False,
            message="No file is selected for improvement.",
        )

    if state.current_file_content is None:
        return ActionResult(
            action="improve_code",
            success=False,
            message="The selected file has not been read.",
        )

    content = state.current_file_content

    proposed_content = content
    improvements: list[str] = []

    if "print(" in content:
        improvements.append(
            "Consider replacing print() with structured logging."
        )

    if len(content.splitlines()) > 300:
        improvements.append(
            "Consider splitting this large file into smaller modules."
        )

    if not improvements:
        improvements.append(
            "No automatic improvement was applied. "
            "A deeper semantic review is recommended."
        )

    state.repair_description = (
        "Code improvement proposal created. "
        "No file was modified."
    )
    state.proposed_content = proposed_content
    state.repair_proposed = True
    state.user_input_needed = True

    return ActionResult(
        action="improve_code",
        success=True,
        message="Code improvement proposal created.",
        data={
            "file": state.current_file,
            "improvements": improvements,
            "original_content": content,
            "proposed_content": proposed_content,
            "requires_approval": True,
        },
    )
