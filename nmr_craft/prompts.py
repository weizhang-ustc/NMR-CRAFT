"""Prompt formatting shared by inverse generation and forward scoring."""
from __future__ import annotations

from typing import Any


def _number(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}".rstrip("0").rstrip(".")


def spectrum_text(h_shift: list[float], c_shift: list[float]) -> str:
    carbon = " ".join(_number(value) for value in sorted(c_shift, reverse=True))
    proton = " ".join(_number(value) for value in sorted(h_shift, reverse=True))
    return f"<13C_NMR> {carbon}\n<1H_NMR> {proton}"


def forward_prompt(smiles: str) -> str:
    return (
        "Predict the NMR chemical-shift sets from the canonical SMILES.\n\n"
        f"<SMILES> {smiles}\n\n"
        "Return only the compact block with <13C_NMR> and <1H_NMR>."
    )


def _nmrtrans_h_text(peaks: list[dict[str, Any]] | None, shifts: list[float]) -> str:
    if not peaks:
        return " ".join(_number(value, 2) for value in sorted(shifts, reverse=True))
    values = []
    for peak in sorted(peaks, key=lambda item: float(item.get("shift", 0)), reverse=True):
        shift = _number(peak.get("shift", 0), 2)
        width = _number(peak.get("width", 0), 2)
        multiplicity = str(peak.get("multiplicity", "unk"))
        integral = str(peak.get("integral", "1H"))
        if not integral.lower().endswith("h"):
            integral += "H"
        couplings = peak.get("j", []) or []
        j_text = ",".join(_number(value, 1) for value in couplings) or "none"
        values.append(f"{shift}|{width}|{multiplicity}|{integral}|J={j_text}")
    return " ; ".join(values)


def inverse_prompt(
    h_shift: list[float],
    c_shift: list[float],
    formula: str,
    h_peaks: list[dict[str, Any]] | None = None,
    prompt_format: str = "nmrgym",
) -> str:
    if not formula:
        raise ValueError("inverse prompt requires a molecular formula")
    formula_line = f"<FORMULA> {formula}\n"
    if prompt_format == "nmrtrans":
        instruction = "Predict the canonical SMILES from the NMR data."
        carbon = " ".join(_number(value, 1) for value in sorted(c_shift, reverse=True))
        spectrum = f"<13C_NMR> {carbon}\n<1H_NMR> {_nmrtrans_h_text(h_peaks, h_shift)}"
    elif prompt_format == "nmrgym":
        instruction = "Predict the canonical SMILES from the NMR chemical-shift sets."
        spectrum = spectrum_text(h_shift, c_shift)
    else:
        raise ValueError(f"unknown prompt format: {prompt_format}")
    return (
        f"{instruction}\n\n"
        f"{formula_line}{spectrum}\n\n"
        "Return only the SMILES."
    )
