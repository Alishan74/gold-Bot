"""
Position sizing: turns a trade plan's price-distance risk into an actual
lot size, given YOUR account size and risk tolerance. Deliberately not
run automatically as part of a signal - account size and contract specs
are yours to provide. A signal by itself is just price levels; how much
to risk on it is a risk-management decision this system can't and
shouldn't make silently on your behalf.

Gold contract sizes vary by broker: standard lot = 100 oz is common, but
mini (10 oz) and micro (1 oz) contracts exist too, and CFD "lot" sizing
can differ from futures contract sizing entirely. Check your own
broker's contract specification before trusting the lot size this
produces - contract_size here is a required input, not a guessed default.
"""
from __future__ import annotations


def position_size(account_size: float, risk_pct: float, risk_per_unit: float,
                   contract_size: float = 100.0) -> dict:
    """
    account_size: your account equity, in account currency (e.g. USD).
    risk_pct: fraction of account to risk on this ONE trade (e.g. 0.01 = 1%).
              Conventional guidance is 0.5-2% per trade for a system with
              this system's ~1:4 structure; this function doesn't enforce
              any particular value, that's a risk-tolerance decision.
    risk_per_unit: the trade_plan's `risk` value (price distance from
              entry to stop, e.g. entry - stop_loss).
    contract_size: units of gold per 1.0 lot for YOUR broker (100oz
              standard is common but not universal - verify your own spec).
    """
    if account_size <= 0 or risk_pct <= 0 or risk_per_unit <= 0 or contract_size <= 0:
        raise ValueError("account_size, risk_pct, risk_per_unit, and contract_size must all be positive")

    dollar_risk = account_size * risk_pct
    units = dollar_risk / risk_per_unit
    lots = units / contract_size

    return {
        "account_size": account_size,
        "risk_pct": risk_pct,
        "dollar_risk": round(dollar_risk, 2),
        "risk_per_unit": round(risk_per_unit, 4),
        "units": round(units, 4),
        "lots": round(lots, 4),
        "contract_size_assumed": contract_size,
        "note": "contract_size varies by broker (100oz standard / 10oz mini / "
                "1oz micro are common patterns, not universal) - confirm your "
                "own broker's contract spec before trusting `lots`.",
    }
