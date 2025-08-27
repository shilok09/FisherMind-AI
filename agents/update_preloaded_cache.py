def update_preloaded_cache(ticker: str, raw_results: dict, analysis_results: dict, preloaded: dict) -> dict:
    """
    Updates the preloaded cache for a given ticker with fetched tool outputs and computed Fisher-style analysis.

    Args:
        ticker (str): The ticker symbol.
        raw_results (dict): Tool outputs (fetched data).
        analysis_results (dict): Fisher-style computed analysis.
        preloaded (dict): The cache to update.

    Returns:
        dict: The updated preloaded cache.
    """
    from datetime import datetime

    # Helper to safely extract or set None
    def get_or_none(d, key):
        return d.get(key) if d.get(key) is not None else None

    # Build the structure
    cache_entry = {
        "financial_line_items": get_or_none(raw_results, "financial_line_items"),
        "market_cap": get_or_none(raw_results, "market_cap"),
        "insider_trades": get_or_none(raw_results, "insider_trades"),
        "company_news": get_or_none(raw_results, "company_news"),
        "analysis_data": {
            "signal": get_or_none(analysis_results, "signal"),
            "score": get_or_none(analysis_results, "score"),
            "max_score": 10,
            "growth_quality": get_or_none(analysis_results, "growth_quality"),
            "margins_stability": get_or_none(analysis_results, "margins_stability"),
            "management_efficiency": get_or_none(analysis_results, "management_efficiency"),
            "valuation_analysis": get_or_none(analysis_results, "valuation_analysis"),
            "insider_activity": get_or_none(analysis_results, "insider_activity"),
            "sentiment_analysis": get_or_none(analysis_results, "sentiment_analysis"),
        },
        "analysis_date": datetime.utcnow().strftime("%Y-%m-%d")
    }

    # If any top-level value is missing, set to None
    for key in ["financial_line_items", "market_cap", "insider_trades", "company_news"]:
        if cache_entry[key] is None:
            cache_entry[key] = None

    # Overwrite or insert
    preloaded[ticker] = cache_entry
    return preloaded
