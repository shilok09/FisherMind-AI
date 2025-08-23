
# -----------------------
# Analysis helpers for Fin-agent.py
# -----------------------
import statistics
import os
from typing import Any, List, Dict
def _v(item: Any, name: str):
    if item is None:
        return None
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name, None)


def analyze_fisher_growth_quality(financial_line_items: List[Any]) -> Dict[str, Any]:
    import logging
    logging.info(f"Received {len(financial_line_items)} items for analysis: {[list(item.keys()) if hasattr(item, 'keys') else type(item).__name__ for item in financial_line_items]}")
    if not financial_line_items or len(financial_line_items) < 2:
        return {"score": 0, "details": "Insufficient financial data for growth/quality analysis"}

    details: List[str] = []
    raw_score = 0

    revenues = [_v(fi, "revenue") for fi in financial_line_items if _v(fi, "revenue") is not None]
    if len(revenues) >= 2:
        latest_rev = revenues[0]
        oldest_rev = revenues[-1]
        if oldest_rev and oldest_rev > 0:
            rev_growth = (latest_rev - oldest_rev) / abs(oldest_rev)
            if rev_growth > 0.80:
                raw_score += 3
                details.append(f"Very strong multi-period revenue growth: {rev_growth:.1%}")
            elif rev_growth > 0.40:
                raw_score += 2
                details.append(f"Moderate multi-period revenue growth: {rev_growth:.1%}")
            elif rev_growth > 0.10:
                raw_score += 1
                details.append(f"Slight multi-period revenue growth: {rev_growth:.1%}")
            else:
                details.append(f"Minimal or negative multi-period revenue growth: {rev_growth:.1%}")
        else:
            details.append("Oldest revenue is zero/negative; cannot compute growth.")
    else:
        details.append("Not enough revenue data points for growth calculation.")

    eps_values = [_v(fi, "earnings_per_share") for fi in financial_line_items if _v(fi, "earnings_per_share") is not None]
    if len(eps_values) >= 2:
        latest_eps = eps_values[0]
        oldest_eps = eps_values[-1]
        if oldest_eps is not None and abs(oldest_eps) > 1e-9:
            eps_growth = (latest_eps - oldest_eps) / abs(oldest_eps)
            if eps_growth > 0.80:
                raw_score += 3
                details.append(f"Very strong multi-period EPS growth: {eps_growth:.1%}")
            elif eps_growth > 0.40:
                raw_score += 2
                details.append(f"Moderate multi-period EPS growth: {eps_growth:.1%}")
            elif eps_growth > 0.10:
                raw_score += 1
                details.append(f"Slight multi-period EPS growth: {eps_growth:.1%}")
            else:
                details.append(f"Minimal or negative multi-period EPS growth: {eps_growth:.1%}")
        else:
            details.append("Oldest EPS near zero; skipping EPS growth calculation.")
    else:
        details.append("Not enough EPS data points for growth calculation.")

    rnd_values = [_v(fi, "research_and_development") for fi in financial_line_items if _v(fi, "research_and_development") is not None]
    if rnd_values and revenues and len(rnd_values) == len(revenues):
        recent_rnd = rnd_values[0]
        recent_rev = revenues[0] if revenues[0] else 1e-9
        rnd_ratio = recent_rnd / recent_rev
        if 0.03 <= rnd_ratio <= 0.15:
            raw_score += 3
            details.append(f"R&D ratio {rnd_ratio:.1%} indicates significant investment in future growth")
        elif rnd_ratio > 0.15:
            raw_score += 2
            details.append(f"R&D ratio {rnd_ratio:.1%} is very high (could be good if well-managed)")
        elif rnd_ratio > 0.0:
            raw_score += 1
            details.append(f"R&D ratio {rnd_ratio:.1%} is somewhat low but still positive")
        else:
            details.append("No meaningful R&D expense ratio")
    else:
        details.append("Insufficient R&D data to evaluate")

    final_score = min(10, (raw_score / 9) * 10)
    return {"score": final_score, "details": "; ".join(details)}


def analyze_margins_stability(financial_line_items: List[Any]) -> Dict[str, Any]:
    import logging
    logging.info(f"Received {len(financial_line_items)} items for analysis: {[list(item.keys()) if hasattr(item, 'keys') else type(item).__name__ for item in financial_line_items]}")
    if not financial_line_items or len(financial_line_items) < 2:
        return {"score": 0, "details": "Insufficient data for margin stability analysis"}

    details: List[str] = []
    raw_score = 0

    op_margins = [_v(fi, "operating_margin") for fi in financial_line_items if _v(fi, "operating_margin") is not None]
    if len(op_margins) >= 2:
        oldest_op_margin = op_margins[-1]
        newest_op_margin = op_margins[0]
        if newest_op_margin is not None and oldest_op_margin is not None and newest_op_margin >= oldest_op_margin > 0:
            raw_score += 2
            details.append(f"Operating margin stable or improving ({oldest_op_margin:.1%} -> {newest_op_margin:.1%})")
        elif newest_op_margin and newest_op_margin > 0:
            raw_score += 1
            details.append("Operating margin positive but slightly declined")
        else:
            details.append("Operating margin may be negative or uncertain")
    else:
        details.append("Not enough operating margin data points")

    gm_values = [_v(fi, "gross_margin") for fi in financial_line_items if _v(fi, "gross_margin") is not None]
    if gm_values:
        recent_gm = gm_values[0]
        if recent_gm > 0.5:
            raw_score += 2
            details.append(f"Strong gross margin: {recent_gm:.1%}")
        elif recent_gm > 0.3:
            raw_score += 1
            details.append(f"Moderate gross margin: {recent_gm:.1%}")
        else:
            details.append(f"Low gross margin: {recent_gm:.1%}")
    else:
        details.append("No gross margin data available")

    if len(op_margins) >= 3:
        stdev = statistics.pstdev([m for m in op_margins if m is not None])
        if stdev < 0.02:
            raw_score += 2
            details.append("Operating margin extremely stable over multiple years")
        elif stdev < 0.05:
            raw_score += 1
            details.append("Operating margin reasonably stable")
        else:
            details.append("Operating margin volatility is high")
    else:
        details.append("Not enough margin data points for volatility check")

    final_score = min(10, (raw_score / 6) * 10)
    return {"score": final_score, "details": "; ".join(details)}


def analyze_management_efficiency_leverage(financial_line_items: List[Any]) -> Dict[str, Any]:
    import logging
    logging.info(f"Received {len(financial_line_items)} items for analysis: {[list(item.keys()) if hasattr(item, 'keys') else type(item).__name__ for item in financial_line_items]}")
    if not financial_line_items:
        return {"score": 0, "details": "No financial data for management efficiency analysis"}

    details: List[str] = []
    raw_score = 0

    ni_values = [_v(fi, "net_income") for fi in financial_line_items if _v(fi, "net_income") is not None]
    eq_values = [_v(fi, "shareholders_equity") for fi in financial_line_items if _v(fi, "shareholders_equity") is not None]
    if ni_values and eq_values and len(ni_values) == len(eq_values):
        recent_ni = ni_values[0]
        recent_eq = eq_values[0] if eq_values[0] else 1e-9
        if recent_ni and recent_ni > 0:
            roe = recent_ni / recent_eq
            if roe > 0.2:
                raw_score += 3
                details.append(f"High ROE: {roe:.1%}")
            elif roe > 0.1:
                raw_score += 2
                details.append(f"Moderate ROE: {roe:.1%}")
            elif roe > 0:
                raw_score += 1
                details.append(f"Positive but low ROE: {roe:.1%}")
            else:
                details.append(f"ROE is near zero or negative: {roe:.1%}")
        else:
            details.append("Recent net income is zero or negative, hurting ROE")
    else:
        details.append("Insufficient data for ROE calculation")

    debt_values = [_v(fi, "total_debt") for fi in financial_line_items if _v(fi, "total_debt") is not None]
    if debt_values and eq_values and len(debt_values) == len(eq_values):
        recent_debt = debt_values[0]
        recent_equity = eq_values[0] if eq_values[0] else 1e-9
        dte = recent_debt / recent_equity
        if dte < 0.3:
            raw_score += 2
            details.append(f"Low debt-to-equity: {dte:.2f}")
        elif dte < 1.0:
            raw_score += 1
            details.append(f"Manageable debt-to-equity: {dte:.2f}")
        else:
            details.append(f"High debt-to-equity: {dte:.2f}")
    else:
        details.append("Insufficient data for debt/equity analysis")

    fcf_values = [_v(fi, "free_cash_flow") for fi in financial_line_items if _v(fi, "free_cash_flow") is not None]
    if fcf_values and len(fcf_values) >= 2:
        positive_fcf_count = sum(1 for x in fcf_values if x and x > 0)
        ratio = positive_fcf_count / len(fcf_values)
        if ratio > 0.8:
            raw_score += 1
            details.append(f"Majority of periods have positive FCF ({positive_fcf_count}/{len(fcf_values)})")
        else:
            details.append("Free cash flow is inconsistent or often negative")
    else:
        details.append("Insufficient or no FCF data to check consistency")

    final_score = min(10, (raw_score / 6) * 10)
    return {"score": final_score, "details": "; ".join(details)}


def analyze_fisher_valuation(financial_line_items: List[Any], market_cap: float | None) -> Dict[str, Any]:
    import logging
    logging.info(f"Received {len(financial_line_items)} items for analysis: {[list(item.keys()) if hasattr(item, 'keys') else type(item).__name__ for item in financial_line_items]}")
    if not financial_line_items or market_cap is None:
        return {"score": 0, "details": "Insufficient data to perform valuation"}

    details: List[str] = []
    raw_score = 0

    net_incomes = [_v(fi, "net_income") for fi in financial_line_items if _v(fi, "net_income") is not None]
    fcf_values = [_v(fi, "free_cash_flow") for fi in financial_line_items if _v(fi, "free_cash_flow") is not None]

    recent_net_income = net_incomes[0] if net_incomes else None
    if recent_net_income and recent_net_income > 0:
        pe = market_cap / recent_net_income
        if pe < 20:
            raw_score += 2
            details.append(f"Reasonably attractive P/E: {pe:.2f}")
        elif pe < 30:
            raw_score += 1
            details.append(f"Somewhat high but possibly justifiable P/E: {pe:.2f}")
        else:
            details.append(f"Very high P/E: {pe:.2f}")
    else:
        details.append("No positive net income for P/E calculation")

    recent_fcf = fcf_values[0] if fcf_values else None
    if recent_fcf and recent_fcf > 0:
        pfcf = market_cap / recent_fcf
        if pfcf < 20:
            raw_score += 2
            details.append(f"Reasonable P/FCF: {pfcf:.2f}")
        elif pfcf < 30:
            raw_score += 1
            details.append(f"Somewhat high P/FCF: {pfcf:.2f}")
        else:
            details.append(f"Excessively high P/FCF: {pfcf:.2f}")
    else:
        details.append("No positive free cash flow for P/FCF calculation")

    final_score = min(10, (raw_score / 4) * 10)
    return {"score": final_score, "details": "; ".join(details)}


def analyze_insider_activity(insider_trades: List[Any]) -> Dict[str, Any]:
    score = 5
    details: List[str] = []
    if not insider_trades:
        details.append("No insider trades data; defaulting to neutral")
        return {"score": score, "details": "; ".join(details)}

    buys, sells = 0, 0
    for trade in insider_trades:
        shares = _v(trade, "transaction_shares")
        if shares is not None:
            if shares > 0:
                buys += 1
            elif shares < 0:
                sells += 1

    total = buys + sells
    if total == 0:
        details.append("No buy/sell transactions found; neutral")
        return {"score": score, "details": "; ".join(details)}

    buy_ratio = buys / total
    if buy_ratio > 0.7:
        score = 8
        details.append(f"Heavy insider buying: {buys} buys vs. {sells} sells")
    elif buy_ratio > 0.4:
        score = 6
        details.append(f"Moderate insider buying: {buys} buys vs. {sells} sells")
    else:
        score = 4
        details.append(f"Mostly insider selling: {buys} buys vs. {sells} sells")

    return {"score": score, "details": "; ".join(details)}


def analyze_sentiment(news_items: List[Any]) -> Dict[str, Any]:
    if not news_items:
        return {"score": 5, "details": "No news data; defaulting to neutral sentiment"}

    negative_keywords = ["lawsuit", "fraud", "negative", "downturn", "decline", "investigation", "recall"]
    negative_count = 0
    for news in news_items:
        title = _v(news, "title") or ""
        title_lower = str(title).lower()
        if any(word in title_lower for word in negative_keywords):
            negative_count += 1

    details: List[str] = []
    if negative_count > len(news_items) * 0.3:
        score = 3
        details.append(f"High proportion of negative headlines: {negative_count}/{len(news_items)}")
    elif negative_count > 0:
        score = 6
        details.append(f"Some negative headlines: {negative_count}/{len(news_items)}")
    else:
        score = 8
        details.append("Mostly positive/neutral headlines")

    return {"score": score, "details": "; ".join(details)}

