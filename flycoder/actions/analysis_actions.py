"""Code analysis actions for FLY-CODER."""

from flycoder.actions.registry import ActionResult
from flycoder.state import CodingState
from flycoder.tools.code_paths import analyze_code_paths
from flycoder.tools.complexity import analyze_complexity
from flycoder.tools.control_flow import analyze_control_flow
from flycoder.tools.data_flow import analyze_data_flow
from flycoder.tools.filesystem import Workspace
from flycoder.tools.references import analyze_references
from flycoder.tools.symbols import (
    SymbolGraph,
    analyze_calls,
    analyze_decorators,
    analyze_imports,
    analyze_inheritance,
    analyze_relationships,
    analyze_symbols,
)


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


def _remove_file_analysis(
    state: CodingState,
    file_path: str,
) -> None:
    """Remove all existing intelligence for one file."""

    state.symbols = [
        item
        for item in state.symbols
        if item.file != file_path
    ]

    state.symbol_imports = [
        item
        for item in state.symbol_imports
        if item.file != file_path
    ]

    state.symbol_calls = [
        item
        for item in state.symbol_calls
        if item.file != file_path
    ]

    state.symbol_inheritance = [
        item
        for item in state.symbol_inheritance
        if item.file != file_path
    ]

    state.symbol_decorators = [
        item
        for item in state.symbol_decorators
        if item.file != file_path
    ]

    state.symbol_relationships = [
        item
        for item in state.symbol_relationships
        if item.file != file_path
    ]

    state.control_flow = [
        item
        for item in state.control_flow
        if item.file != file_path
    ]

    state.symbol_references = [
        item
        for item in state.symbol_references
        if item.file != file_path
    ]

    state.data_flow = [
        item
        for item in state.data_flow
        if item.file != file_path
    ]

    state.complexity_metrics = [
        item
        for item in state.complexity_metrics
        if item.file != file_path
    ]

    state.code_paths = [
        item
        for item in state.code_paths
        if item.file != file_path
    ]


def _analyze_current_file(
    state: CodingState,
) -> dict:
    """
    Run all Phase 3 and Phase 4 intelligence analyzers on the
    currently selected file.
    """

    if not state.current_file:
        return {}

    if state.current_file_content is None:
        return {}

    file_path = state.current_file
    content = state.current_file_content

    # ----------------------------------------------------------
    # Phase 3: Symbol intelligence
    # ----------------------------------------------------------

    symbols = analyze_symbols(
        content,
        file_path=file_path,
    )

    imports = analyze_imports(
        content,
        file_path=file_path,
    )

    calls = analyze_calls(
        content,
        file_path=file_path,
    )

    inheritance = analyze_inheritance(
        content,
        file_path=file_path,
    )

    decorators = analyze_decorators(
        content,
        file_path=file_path,
    )

    relationships = analyze_relationships(
        content,
        file_path=file_path,
    )

    # ----------------------------------------------------------
    # Phase 4: Code intelligence
    # ----------------------------------------------------------

    control_flow = analyze_control_flow(
        content,
        file_path=file_path,
    )

    references = analyze_references(
        content,
        file_path=file_path,
    )

    data_flow = analyze_data_flow(
        content,
        file_path=file_path,
    )

    complexity = analyze_complexity(
        content,
        file=file_path,
    )

    code_paths = analyze_code_paths(
        content,
        file=file_path,
    )

    # ----------------------------------------------------------
    # Replace this file's previous analysis.
    # ----------------------------------------------------------

    _remove_file_analysis(
        state,
        file_path,
    )

    # ----------------------------------------------------------
    # Store Phase 3 results.
    # ----------------------------------------------------------

    state.symbols.extend(symbols)
    state.symbol_imports.extend(imports)
    state.symbol_calls.extend(calls)
    state.symbol_inheritance.extend(inheritance)
    state.symbol_decorators.extend(decorators)
    state.symbol_relationships.extend(relationships)

    # ----------------------------------------------------------
    # Store Phase 4 results.
    # ----------------------------------------------------------

    state.control_flow.extend(control_flow)
    state.symbol_references.extend(references)
    state.data_flow.extend(data_flow)
    state.complexity_metrics.extend(complexity)
    state.code_paths.extend(code_paths)

    # Rebuild the relationship graph from the complete
    # project-level relationship collection.
    state.symbol_graph = SymbolGraph(
        state.symbol_relationships,
    )

    return {
        "symbols": symbols,
        "imports": imports,
        "calls": calls,
        "inheritance": inheritance,
        "decorators": decorators,
        "relationships": relationships,
        "control_flow": control_flow,
        "references": references,
        "data_flow": data_flow,
        "complexity": complexity,
        "code_paths": code_paths,
    }


# Preserve the Phase 3 helper name for existing callers/tests.
def _analyze_current_file_symbols(
    state: CodingState,
) -> dict:
    """Compatibility wrapper for the Phase 3 symbol-analysis helper."""

    data = _analyze_current_file(state)

    return {
        "symbols": data.get("symbols", []),
        "imports": data.get("imports", []),
        "calls": data.get("calls", []),
        "inheritance": data.get("inheritance", []),
        "decorators": data.get("decorators", []),
        "relationships": data.get("relationships", []),
    }


def review_code_action(
    workspace: Workspace,
    state: CodingState,
) -> ActionResult:
    """Perform static review and complete code intelligence analysis."""

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

    analysis_data = _analyze_current_file(
        state,
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

    symbol_counts = {
        "symbols": len(
            analysis_data.get("symbols", [])
        ),
        "imports": len(
            analysis_data.get("imports", [])
        ),
        "calls": len(
            analysis_data.get("calls", [])
        ),
        "inheritance": len(
            analysis_data.get("inheritance", [])
        ),
        "decorators": len(
            analysis_data.get("decorators", [])
        ),
        "relationships": len(
            analysis_data.get("relationships", [])
        ),
    }

    code_intelligence_counts = {
        "control_flow": len(
            analysis_data.get("control_flow", [])
        ),
        "references": len(
            analysis_data.get("references", [])
        ),
        "data_flow": len(
            analysis_data.get("data_flow", [])
        ),
        "complexity": len(
            analysis_data.get("complexity", [])
        ),
        "code_paths": len(
            analysis_data.get("code_paths", [])
        ),
    }

    return ActionResult(
        action="review_code",
        success=True,
        message=(
            "Code review and complete code intelligence "
            "analysis completed."
        ),
        data={
            "file": state.current_file,
            "lines": len(lines),
            "findings": findings,
            "review": review,
            "symbol_counts": symbol_counts,
            "code_intelligence_counts": (
                code_intelligence_counts
            ),
            "analysis_counts": {
                **symbol_counts,
                **code_intelligence_counts,
            },
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