"""Choose one gold label per image: adjudicated -> gold_annotators -> unique `done` label."""
from __future__ import annotations

from dataclasses import dataclass, field

from smv.data.labels import LabelFile


@dataclass
class GoldChoice:
    gold: LabelFile | None
    reason: str
    others: list[LabelFile] = field(default_factory=list)   # other `done` sets, for agreement
    n_draft: int = 0
    n_needs_review: int = 0
    warnings: list[str] = field(default_factory=list)


def select_gold(labels: list[LabelFile], gold_annotators: list[str]) -> GoldChoice:
    adj = [lf for lf in labels if lf.adjudicated and lf.ann is not None]
    regular = [lf for lf in labels if not lf.adjudicated and lf.ann is not None]
    n_draft = sum(1 for lf in regular if lf.ann.status == "draft")
    n_review = sum(1 for lf in regular if lf.ann.status == "needs_review")
    done = [lf for lf in regular if lf.ann.status == "done"]
    warnings: list[str] = []

    def others_of(g: LabelFile | None) -> list[LabelFile]:
        return [lf for lf in done if lf is not g]

    if adj:
        if len(adj) > 1:
            warnings.append(f"{len(adj)} adjudicated labels; using the highest version")
        g = sorted(adj, key=lambda lf: (lf.ann.version or 0, str(lf.path)))[-1]
        return GoldChoice(g, "adjudicated", others_of(None), n_draft, n_review, warnings)
    for name in gold_annotators:
        cands = [lf for lf in done if lf.annotator == name]
        if cands:
            g = sorted(cands, key=lambda lf: (lf.ann.version or 0, str(lf.path)))[-1]
            return GoldChoice(g, f"gold_annotator:{name}", others_of(g), n_draft, n_review, warnings)
    if len(done) == 1:
        return GoldChoice(done[0], "unique_done", [], n_draft, n_review, warnings)
    if len(done) > 1:
        g = sorted(done, key=lambda lf: (-(lf.ann.version or 0), lf.annotator, str(lf.path)))[0]
        warnings.append(f"{len(done)} done labels and none from gold_annotators; picked {g.annotator} "
                        "(highest version, then alphabetical)")
        return GoldChoice(g, "multiple_done_pick_first", others_of(g), n_draft, n_review, warnings)
    reason = "only_draft" if n_draft else ("only_needs_review" if n_review else "no_label")
    return GoldChoice(None, reason, [], n_draft, n_review, warnings)
