from smolagents import tool
import requests
import warnings
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd

warnings.filterwarnings("ignore")

""" 

FUNCTIONS THAT START WITH AN "_" ARE REPRESENTED AS INTERNAL HELPERS
AGENTS SHOULD NOT CALL THESE FUNCTION

"""
def _session():
    """ API Request Handler -> This creates a reusable HTTP session.

        Returns:
            session
    """

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0",
        }
    )
    return session


def _ticker(symbol):
    return yf.Ticker(symbol.strip().upper(), session=_session())


def _money(value):
    if value is None:
        return "N/A"
    try:
        return f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return str(value)


def _date(value):
    if value is None:
        return "N/A"
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d %H:%M")
    return str(value)



def _safe_info(ticker):
    """ Retrieves general information about the stock

    Returns:
        {
        "longName": "Microsoft Corporation",
        "sector": "Technology",
        "marketCap": 123456789
        }
    """
    try:
        return ticker.info or {}
    except Exception:
        return {}


def _chart(symbol, rng="5d"):
    """ Shows the historical price chart of the symbol

        Returns:
            result: price chart for a specific period

    """

    url = "https://query1.finance.yahoo.com/v8/finance/chart/" + symbol.strip().upper()
    params = {"interval": "1d", "range": rng}
    try:
        response = _session().get(url, params=params, timeout=10)
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return None
    
    results = (payload.get("chart") or {}).get("result") or []
    return results[0] if results else None


def _price_block(symbol):
    """ This function turns chart data into a readable stock-price summary.


        DESCRIPTION:

        meta.get extracts the following potential metadata from the chart
        * Company name
        * Exchange
        * Currency
        * Current price
        * Day’s high and low
        * Previous close
        * Trading volume
        * 52-week high and low
    """

    symbol = symbol.strip().upper()
    data = _chart(symbol, "5d")
   
    if not data:
        return f"Price data for {symbol} is unavailable."
   
    meta = data.get("meta") or {}
    lines = []
    name = meta.get("longName") or meta.get("shortName") or symbol
   
    lines.append(f"Price and info for {name} ({symbol}):")
    lines.append(f"Exchange: {meta.get('exchangeName', 'N/A')}  |  Currency: {meta.get('currency', 'N/A')}")
    lines.append(f"Current price: {_money(meta.get('regularMarketPrice'))}")
    lines.append(f"Day range: {_money(meta.get('regularMarketDayLow'))} - {_money(meta.get('regularMarketDayHigh'))}")
    lines.append(f"Previous close: {_money(meta.get('chartPreviousClose'))}")
    lines.append(f"Regular market volume: {_money(meta.get('regularMarketVolume'))}")
    lines.append(f"52-week range: {_money(meta.get('fiftyTwoWeekLow'))} - {_money(meta.get('fiftyTwoWeekHigh'))}")
   
    change = meta.get("regularMarketChangePercent")
   
    if change is not None:
        lines.append(f"Daily change: {change:.2f}%")
    
    """
    Extracting historical prices then formats the most recent five sessions.
    """
    timestamps = data.get("timestamp") or []
    quotes = (data.get("indicators") or {}).get("quote") or [{}]
    quote = quotes[0] if quotes else {}
    closes = quote.get("close") or []
    volumes = quote.get("volume") or []
    sessions = list(zip(timestamps, closes, volumes))[-5:]
   
    if sessions:
        lines.append("Recent sessions:")
   
        for timestamp, close, volume in sessions:
            date = datetime.utcfromtimestamp(timestamp).strftime("%Y-%m-%d") if timestamp else "N/A"
            lines.append(f"  {date}: close {_money(close)}, volume {_money(volume)}")
   
    return "\n".join(lines)


def _news_items(query, count=5):
    """ Retrieves recent financial news from Yahoo Finance.

        Returns:
            joined news lines
    """

    url = "https://query2.finance.yahoo.com/v1/finance/search"
    params = {
        "q": query,
        "quotesCount": 3,
        "newsCount": count,
        "enableFuzzyQuery": "false",
    }
    try:
        response = _session().get(url, params=params, timeout=10)
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return "No recent news found."
    
    news = payload.get("news") or []

    if not news:
        return "No recent news found."
    
    lines = []
    
    for item in news[:count]:
        title = item.get("title", "Untitled")
        publisher = item.get("publisher", "Unknown")
        link = item.get("link", "")
        timestamp = item.get("providerPublishTime")
        date = (
            datetime.utcfromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M UTC")
            if timestamp
            else ""
        )
        lines.append(f"- {date} | {title} ({publisher})\n  {link}")

    return "\n".join(lines)


def _ticker_news(symbol):
    symbol = symbol.strip().upper()
    items = _news_items(symbol, 4)
    return f"Recent news for {symbol}:\n{items}"


def _market_news():
    return f"Recent stock market news:\n{_news_items('stock market today', 5)}"


def _calendar_block(ticker):

    symbol = ticker.strip().upper()
    stock = _ticker(symbol)
    blocks = []
    calendar = None
    
    try:
        calendar = stock.calendar
    except Exception:
        calendar = None
    
    if isinstance(calendar, dict) and calendar:
        blocks.append(f"Earnings/events calendar for {symbol}:")
    
        for key, value in calendar.items():
    
            if isinstance(value, list):
                value = ", ".join(str(_date(item)) for item in value)
            else:
                value = _date(value) if "Date" in str(key) else str(value)
            blocks.append(f"  {key}: {value}")
    else:
        blocks.append(f"No calendar data found for {symbol}.")
    
    try:
        earnings_dates = stock.earnings_dates
    
        if earnings_dates is not None and not earnings_dates.empty:
    
            now = datetime.now()
            tz = getattr(earnings_dates.index, "tz", None)
            now = pd.Timestamp.now(tz=tz) if tz is not None else pd.Timestamp.now()
            future = earnings_dates[earnings_dates.index >= now]
    
            if getattr(future, "empty", True):
                future = earnings_dates.head(3)
            lines = [f"Upcoming/recent earnings for {symbol}:"]
    
            for index, row in future.head(5).iterrows():
                estimate = row.get("EPS Estimate")
                estimate_text = "N/A" if estimate != estimate else estimate
                lines.append(
                    f"  {_date(index)}: EPS estimate {estimate_text}"
                )
    
            blocks.append("\n".join(lines))
    
    except Exception:
        pass
    
    return "\n".join(blocks)


def _updates_block(symbol):
    symbol = symbol.strip().upper()
    stock = _ticker(symbol)
    blocks = [f"Recent updates and filings for {symbol}:"]
    try:
        upgrades = stock.upgrades_downgrades

        if upgrades is not None and not upgrades.empty:
            lines = ["Analyst upgrades/downgrades (latest):"]
        
            for index, row in upgrades.head(3).iterrows():
                lines.append(
                    "  "
                    + f"{_date(index)} | {row.get('Firm', 'N/A')}: {row.get('Action', '')} "
                    + f"{row.get('FromGrade', '')} -> {row.get('ToGrade', '')}; "
                    + f"target {_money(row.get('currentPriceTarget'))}"
                )
            blocks.append("\n".join(lines))
    except Exception:
        pass

    try:
        summary = stock.recommendations_summary
    
        if summary is not None and not summary.empty:
            row = summary.iloc[0]
            blocks.append(
                "Analyst recommendation summary: "
                + f"strong buy {row.get('strongBuy', 'N/A')}, buy {row.get('buy', 'N/A')}, "
                + f"hold {row.get('hold', 'N/A')}, sell {row.get('sell', 'N/A')}, "
                + f"strong sell {row.get('strongSell', 'N/A')}"
            )
    
    except Exception:
        pass
    
    try:
        filings = stock.sec_filings
        if filings:
            lines = ["Recent SEC filings:"]
            for filing in filings[:3]:
                lines.append(
                    f"  {filing.get('date')} | {filing.get('type')} - {filing.get('title')}"
                )
            blocks.append("\n".join(lines))
    
    except Exception:
        pass
    
    try:
        insider = stock.insider_purchases
    
        if insider is not None and not insider.empty:
            lines = ["Insider activity (last 6 months):"]
    
            for _, row in insider.head(3).iterrows():
                lines.append(f"  {row.iloc[0]}: {row.get('Shares', 'N/A')} shares")
            blocks.append("\n".join(lines))
    
    except Exception:
        pass
    
    if len(blocks) == 1:
        blocks.append("No recent updates found.")
    
    return "\n".join(blocks)


def _market_earnings(days=7):
    calendar = yf.Calendars(session=_session())
    today = datetime.now().date()
    end = today + timedelta(days=days)

    try:
        frame = calendar.get_earnings_calendar(
            start=str(today), end=str(end), limit=25
        )

    except Exception:
        return "No upcoming earnings data found."
    
    if frame is None or getattr(frame, "empty", True):
        return "No upcoming earnings data found."
    
    lines = ["Upcoming earnings:"]
    
    for symbol, row in frame.iterrows():
        
        market_cap = row.get("Marketcap")
        market_cap_text = (
            f"${market_cap / 1e9:,.1f}B" if market_cap and market_cap == market_cap else "N/A"
        )
        
        date = row.get("Event Start Date", row.get("Earnings Date"))
        timing = row.get("Timing", "")
        estimate = row.get("EPS Estimate", "N/A")
        
        lines.append(
            f"  {symbol} | {row.get('Company', 'N/A')} | {date} {timing} | "
            f"EPS est {estimate} | market cap {market_cap_text}"
        )
    
    return "\n".join(lines)


def _new_ipos(days_past=7, days_future=30):

    calendar = yf.Calendars(session=_session())
    today = datetime.now().date()
    start = today - timedelta(days=days_past)
    end = today + timedelta(days=days_future)
    
    try:
        frame = calendar.get_ipo_info_calendar(
            start=str(start), end=str(end), limit=25
        )
    
    except Exception:
        return "No IPO calendar data found."
    
    if frame is None or getattr(frame, "empty", True):
        return "No IPO calendar data found."
    
    lines = ["Recent and upcoming IPOs:"]
    
    for symbol, row in frame.iterrows():
    
        price = row.get("Price")
        price_from = row.get("Price From")
        price_to = row.get("Price To")
    
        price_text = (
            _money(price)
            if price and price == price
            else (f"{_money(price_from)} - {_money(price_to)}" if price_from and price_from == price_from else "N/A")
        )
    
        shares = row.get("Shares")
        shares_text = (
            f"{int(shares):,}" if shares and shares == shares else "N/A"
        )
    
        lines.append(
            f"  {symbol} | {row.get('Company', 'N/A')} | date {_date(row.get('Date'))} | "
            f"exchange {row.get('Exchange', 'N/A')} | price {price_text} | shares {shares_text} | "
            f"{row.get('Action', 'N/A')}"
        )
    
    return "\n".join(lines)


@tool
def get_stock_info(ticker: str = "") -> str:
    """Retrieve recent stock information for a company or the overall market.

    Use this tool for broad stock questions about a specific company (e.g. stock prices, company news, updates,
    filings, insider activity, earnings, new IPOs, or market news.)
    Prefer this tool over calling multiple individual stock tools separately.

    Args:
        ticker: Stock ticker symbol such as AAPL, MSFT, TSLA. Leave empty for
            broad market IPOs, upcoming earnings, and market news.

    Returns: A text summary with current price data, recent news, earnings
    calendar, analyst updates, filings, and market-wide IPO and earnings data.
    """

    ticker = ticker.strip()
    if not ticker:
        blocks = [
            _new_ipos(),
            _market_earnings(),
            _market_news(),
        ]
        return "\n\n".join(blocks)

    blocks = [
        _price_block(ticker),
        _ticker_news(ticker),
        _calendar_block(ticker),
        _updates_block(ticker),
    ]
    return "\n\n".join(blocks)


@tool
def get_stock_price(ticker: str) -> str:
    """Get the current price and recent price history for a stock ticker.

    Args:
        ticker: Stock ticker symbol such as AAPL or TSLA.

    Returns: Current price, day range, previous close, market cap, and recent
    trading sessions.
    """

    return _price_block(ticker)


@tool
def get_stock_news(ticker: str = "") -> str:
    """Get recent news for a stock ticker or the broader stock market.

    Args:
        ticker: Stock ticker symbol. Leave empty for general stock market news.

    Returns: Recent headlines with publishers, dates, and links.
    """

    if ticker.strip():
        return _ticker_news(ticker)
    return _market_news()


@tool
def get_stock_updates(ticker: str) -> str:
    """Get recent analyst updates, SEC filings, and insider activity for a stock.

    Args:
        ticker: Stock ticker symbol such as AAPL or TSLA.

    Returns: Analyst upgrades/downgrades, recommendation summary, filings, and
    insider purchase activity.
    """

    return _updates_block(ticker)


@tool
def get_upcoming_earnings(ticker: str = "") -> str:
    """Get upcoming earnings for a stock or across the market.

    Args:
        ticker: Stock ticker symbol. Leave empty for market-wide upcoming
            earnings.

    Returns: Earnings dates, EPS estimates, timing, and market cap data.
    """

    if ticker.strip():
        return _calendar_block(ticker)
    return _market_earnings()


@tool
def get_new_ipos() -> str:
    """Get recently listed and upcoming IPOs from the market calendar.

    Returns: Ticker symbols, companies, dates, exchanges, prices, and share
    counts for recent and upcoming IPOs.
    """

    return _new_ipos()


get_stock_data = get_stock_info
get_stock = get_stock_info
get_stocks = get_stock_info
get_stock_prices = get_stock_price
get_price = get_stock_price
get_news = get_stock_news
get_stock_update = get_stock_updates
get_earnings = get_upcoming_earnings
get_ipos = get_new_ipos
get_recent_ipos = get_new_ipos
