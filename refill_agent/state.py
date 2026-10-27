"""Explicit field updates and version-bound confirmation of a local read-back."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import re


FIELDS = ("medication", "dosage", "prescription_id", "pickup_time")
LABELS = {
    "medication": "medication",
    "dosage": "dosage",
    "prescription id": "prescription_id",
    "prescription number": "prescription_id",
    "pickup time": "pickup_time",
}
UPDATE = re.compile(
    r"^(?:actually[, ]+)?(?:my\s+)?"
    r"(medication|dosage|prescription id|prescription number|pickup time)"
    r"\s*(?::|=|\bis\b|\bto\b)\s*(.+)$", re.IGNORECASE
)


@dataclass
class RefillState:
    medication: str = ""
    dosage: str = ""
    prescription_id: str = ""
    pickup_time: str = ""
    revision: int = 0
    confirmed_revision: int | None = None

    def details(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in FIELDS}

    def missing(self) -> list[str]:
        return [name for name in FIELDS if not getattr(self, name)]

    def confirmation_code(self) -> str:
        payload = json.dumps(
            {"details": self.details(), "revision": self.revision},
            sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:8]


@dataclass
class Reply:
    text: str
    status: str
    state: dict
    changed_fields: list[str] = field(default_factory=list)


class RefillConversation:
    """Accept labeled utterances; never infer a medication, dose, or action."""

    def __init__(self) -> None:
        self.state = RefillState()
        self.offered_code: str | None = None

    def _reply(self, text: str, status: str, changed: list[str] | None = None) -> Reply:
        return Reply(text, status, asdict(self.state), changed or [])

    def readback(self) -> str:
        values = self.state.details()
        # Spell reference characters so the speech input preserves their sequence.
        reference = ", ".join("dash" if c == "-" else "space" if c == " " else c
                              for c in values["prescription_id"])
        return (f"Medication: {values['medication']}. Dosage: {values['dosage']}. "
                f"Prescription number: {reference}. Pickup time: {values['pickup_time']}.")

    def handle(self, utterance: str) -> Reply:
        if not isinstance(utterance, str) or not utterance.strip():
            return self._reply("Please provide a labeled detail or request a summary.", "needs_input")
        if len(utterance) > 2000:
            return self._reply("Please use a shorter turn, at most 2000 characters.", "invalid_input")
        text = utterance.strip()
        confirm = re.fullmatch(r"confirm\s+([a-f0-9]{8})[.!]?", text, re.IGNORECASE)
        plain_confirmation = text.lower().rstrip(".!?") in {"confirm", "yes"}
        if confirm or plain_confirmation:
            code = confirm.group(1).lower() if confirm else self.offered_code
            if (self.state.missing() or self.offered_code is None or
                    code != self.offered_code or code != self.state.confirmation_code()):
                return self._reply("That confirmation is stale or unavailable. Request a new summary.",
                                   "stale_confirmation")
            self.state.confirmed_revision = self.state.revision
            return self._reply("The current read-back is confirmed locally. No refill has been ordered.",
                               "confirmed")
        if text.lower().rstrip(".!?") in {"summary", "read back", "readback"}:
            if self.state.missing():
                return self._reply("Please provide: " + ", ".join(self.state.missing()) + ".",
                                   "missing_details")
            self.offered_code = self.state.confirmation_code()
            return self._reply(self.readback() + f" Say yes to confirm this version. Read-back code: {self.offered_code}.",
                               "awaiting_confirmation")

        updates: dict[str, str] = {}
        for statement in text.split(";"):
            match = UPDATE.fullmatch(statement.strip())
            if not match:
                return self._reply(
                    "Use labeled details separated by semicolons: medication is ...; dosage is ...; "
                    "prescription number is ...; pickup time is ... . Say summary for a read-back.",
                    "unsupported_utterance",
                )
            key = LABELS[match.group(1).lower()]
            value = match.group(2).strip().rstrip(".!?").strip()
            if not value or len(value) > 120 or not value.isprintable():
                return self._reply("Each detail must be 1 to 120 printable characters.", "invalid_detail")
            if key in updates:
                return self._reply("Give each detail once per turn; send a separate correction.",
                                   "ambiguous_update")
            if key == "prescription_id" and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 -]{1,39}", value):
                return self._reply("Use a prescription reference of 2 to 40 letters, digits, spaces or dashes.",
                                   "invalid_detail")
            updates[key] = value

        changed = [key for key, value in updates.items() if getattr(self.state, key) != value]
        if changed:
            for key, value in updates.items():
                setattr(self.state, key, value)
            self.state.revision += 1
            self.state.confirmed_revision = None
            self.offered_code = None
        missing = self.state.missing()
        if missing:
            return self._reply("Details recorded. Still needed: " + ", ".join(missing) + ".",
                               "collecting", changed)
        self.offered_code = self.state.confirmation_code()
        return self._reply(self.readback() + f" Say yes to confirm this version. Read-back code: {self.offered_code}.",
                           "awaiting_confirmation", changed)
