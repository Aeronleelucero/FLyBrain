"""State definitions for the FLY-CODER agent."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from flycoder.tools.code_paths import CodePath
from flycoder.tools.complexity import ComplexityInfo
from flycoder.tools.control_flow import ControlFlowNode
from flycoder.tools.data_flow import DataFlowInfo
from flycoder.tools.references import ReferenceInfo
from flycoder.tools.symbols import (
    CallInfo,
    DecoratorInfo,
    ImportInfo,
    InheritanceInfo,
    Relationship,
    Symbol,
    SymbolGraph,
)

if TYPE_CHECKING:
    from flycoder.tools.completion_evidence import VerificationRecord


@dataclass
class CodingState:
    """Current state of a coding task."""

    task: str

    # ----------------------------------------------------------
    # Task
    # ----------------------------------------------------------

    task_intent: str | None = None

    # ----------------------------------------------------------
    # File discovery and selection
    # ----------------------------------------------------------

    files: list[str] = field(default_factory=list)

    relevant_files: list[str] = field(default_factory=list)

    files_analyzed: list[str] = field(default_factory=list)

    # ----------------------------------------------------------
    # Code review
    # ----------------------------------------------------------

    review_findings: list[dict] = field(default_factory=list)

    # ----------------------------------------------------------
    # Dependency analysis
    # ----------------------------------------------------------

    dependency_graph: dict[str, list[str]] = field(
        default_factory=dict
    )

    # ----------------------------------------------------------
    # Symbol intelligence
    # ----------------------------------------------------------

    symbols: list[Symbol] = field(default_factory=list)

    symbol_imports: list[ImportInfo] = field(
        default_factory=list
    )

    symbol_calls: list[CallInfo] = field(default_factory=list)

    symbol_inheritance: list[InheritanceInfo] = field(
        default_factory=list
    )

    symbol_decorators: list[DecoratorInfo] = field(
        default_factory=list
    )

    symbol_relationships: list[Relationship] = field(
        default_factory=list
    )

    symbol_graph: SymbolGraph | None = None

    # ----------------------------------------------------------
    # Code intelligence
    # ----------------------------------------------------------

    # Control-flow structures discovered during AST analysis.
    control_flow: list[ControlFlowNode] = field(
        default_factory=list
    )

    # Variable, attribute, and call references discovered during
    # AST analysis.
    symbol_references: list[ReferenceInfo] = field(
        default_factory=list
    )

    # Approximate source-to-target data-flow relationships.
    data_flow: list[DataFlowInfo] = field(
        default_factory=list
    )

    # Per-function complexity metrics.
    complexity_metrics: list[ComplexityInfo] = field(
        default_factory=list
    )

    # Approximate execution-path information.
    code_paths: list[CodePath] = field(
        default_factory=list
    )

    # ----------------------------------------------------------
    # Current file
    # ----------------------------------------------------------

    current_file: str | None = None

    current_file_content: str | None = None

    # ----------------------------------------------------------
    # Errors and testing
    # ----------------------------------------------------------

    last_error: str | None = None

    tests_passed: bool = False

    tests_run: bool = False

    error_inspected: bool = False

    # ----------------------------------------------------------
    # Verification freshness (Phase 10.5)
    #
    # code_version increases on every successful code modification
    # made through an agent action. tests_code_version records which
    # code version the last test run verified (None: recorded before
    # versioning, i.e. version 0). See flycoder.tools.freshness.
    # ----------------------------------------------------------

    code_version: int = 0

    tests_code_version: int | None = None

    # Latest Phase 7 verification pipeline result, bound to the code
    # signature it verified (a completion_evidence.VerificationRecord).
    verification_record: "VerificationRecord | None" = None

    # ----------------------------------------------------------
    # Adaptive execution
    # ----------------------------------------------------------

    # Most recently executed action.
    last_action: str | None = None

    # Whether the most recently executed action reported success.
    last_action_success: bool | None = None

    # Human-readable result of the most recently executed action.
    last_action_message: str | None = None

    # ----------------------------------------------------------
    # Repair workflow
    # ----------------------------------------------------------

    repair_proposed: bool = False

    repair_approved: bool = False

    repair_applied: bool = False

    repair_description: str | None = None

    proposed_content: str | None = None

    # Original content from the most recently applied repair.
    # Used to support safe rollback during the current agent run.
    repair_original_content: str | None = None

    # ----------------------------------------------------------
    # Agent control
    # ----------------------------------------------------------

    user_input_needed: bool = False

    # Why user_input_needed was set (Phase 10.6). One of the reasons in
    # flycoder.tools.human_input; None when no reason was recorded.
    human_input_reason: str | None = None

    finished: bool = False
