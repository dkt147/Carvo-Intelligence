import json

from app.retrieval.retriever import RetrievedChunk
from app.schemas.analysis import AnalysisRequest

SYSTEM_PROMPT = """You are the CARVO intelligence and analysis layer.

CARVO is a protocol- and knowledge-driven system, not a generic assistant. You
turn a structured situation plus its context (events, actions, optional project,
optional protocol) into a traceable, evidence-based analysis.

Mandatory rules:
- Do NOT invent domain rules, protocol conditions, facts, definitions, decision
  criteria, or scores that are not present in the supplied input.
- Do NOT create psychological, motivation, readiness, ego, corruption, or similar
  scores unless an approved protocol explicitly defines them.
- Distinguish supplied facts, derived interpretation, and recommendation. Make
  uncertainty explicit rather than guessing.
- If required information is missing and no safe recommendation can be made, say
  so and put concrete questions in metadata.clarifications instead of inventing.
- Base every recommendation on the situation, its context, and the protocol
  content when one is provided.
- RETRIEVED PROTOCOL KNOWLEDGE is candidate material extracted from source
  documents and is NOT yet approved. Use it only as supporting context, never as
  a hard rule, and say so when you rely on it.
- If neither an approved protocol nor retrieved knowledge is available, state that
  the analysis is not yet protocol-grounded.
- Preserve traceability: when you rely on a specific event / action / protocol /
  retrieved passage, reference it under metadata.evidence (use type
  "KNOWLEDGE_CANDIDATE" and the source string for retrieved passages).
- Answer in the language of the situation and protocol content when they are not
  in English.

Return ONLY a single JSON object with these keys:
  "summary": string - short plain-language summary of the situation and finding.
  "reasoning": string - structured explanation of how you reached the finding.
  "recommendations": array of objects, each { "title": string, "description": string }.
  "protocolVersionId": string or null - only if a protocol version id is supplied.
  "metadata": object - may include:
       "clarifications": [ { "question": string, "reason": string } ],
       "uncertainties": [ string ],
       "evidence": [ { "type": string, "id": string, "reason": string } ]
"""


def build_user_prompt(
    request: AnalysisRequest,
    retrieved: list[RetrievedChunk] | None = None,
) -> str:
    s = request.situation
    ctx = request.context

    lines: list[str] = []
    lines.append(f"SITUATION (id={s.id}, status={s.status})")
    lines.append(f"  title: {s.title}")
    lines.append(f"  description: {s.description or '(none)'}")
    lines.append("")

    lines.append(f"EVENTS ({len(ctx.events)}):")
    for e in ctx.events:
        when = e.occurredAt or "n/a"
        lines.append(
            f"  - [{e.type}] (id={e.id}) {e.title} @ {when} :: {e.description or ''}"
        )
    if not ctx.events:
        lines.append("  (none)")
    lines.append("")

    lines.append(f"ACTIONS ({len(ctx.actions)}):")
    for a in ctx.actions:
        due = a.dueDate or "no due date"
        lines.append(
            f"  - (id={a.id}) [{a.status}] {a.title} ({due}) :: {a.description or ''}"
        )
    if not ctx.actions:
        lines.append("  (none)")
    lines.append("")

    if ctx.project:
        p = ctx.project
        lines.append(f"PROJECT (id={p.id}, status={p.status})")
        lines.append(f"  name: {p.name}")
        lines.append(f"  description: {p.description or '(none)'}")
        lines.append(f"  point A: {p.pointA or '(none)'}")
        lines.append(f"  point B: {p.pointB or '(none)'}")
    else:
        lines.append("PROJECT: (not supplied)")
    lines.append("")

    if request.protocol:
        lines.append(
            f"PROTOCOL (id={request.protocol.id}, version={request.protocol.version}):"
        )
        lines.append(request.protocol.content)
    else:
        lines.append("PROTOCOL: (not supplied)")
    lines.append("")

    lines.append(
        f"RETRIEVED PROTOCOL KNOWLEDGE ({len(retrieved or [])} candidate extracts, "
        "NOT yet approved - supporting context only):"
    )
    for i, chunk in enumerate(retrieved or [], start=1):
        lines.append(
            f"  [{i}] (source: {chunk.source}, section: {chunk.section}, "
            f"score: {chunk.score:.3f})"
        )
        lines.append(f"      {chunk.text}")
    if not retrieved:
        lines.append("  (none)")
    lines.append("")

    lines.append("Raw request JSON:")
    lines.append(json.dumps(request.model_dump(), ensure_ascii=False, indent=2, default=str))

    return "\n".join(lines)


def build_retrieval_query(request: AnalysisRequest) -> str:
    s = request.situation
    parts = [s.title, s.description or ""]
    parts += [e.title for e in request.context.events]
    return "\n".join(p for p in parts if p).strip()
