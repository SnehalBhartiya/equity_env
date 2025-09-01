import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# --- Page Configuration ---
st.set_page_config(
    page_title="Indian Stock Recommender",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Data Store ---
STOCKS_BY_SECTOR = {
    "Information Technology": [
        'TCS.NS', 'INFY.NS', 'HCLTECH.NS', 'WIPRO.NS', 'TECHM.NS',
        'LTIM.NS', 'MPHASIS.NS', 'PERSISTENT.NS', 'COFORGE.NS', 'TATAELXSI.NS'
    ],
    "Financial Services": [
        'HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'KOTAKBANK.NS', 'AXISBANK.NS',
        'BAJFINANCE.NS', 'BAJAJFINSV.NS', 'HDFCLIFE.NS', 'SBILIFE.NS', 'INDUSINDBK.NS',
        'LICI.NS', 'CHOLAFIN.NS', 'BAJAJHLDNG.NS', 'SRTRANSFIN.NS', 'PNB.NS'
    ],
    "Fast Moving Consumer Goods (FMCG)": [
        'HINDUNILVR.NS', 'ITC.NS', 'NESTLEIND.NS', 'BRITANNIA.NS', 'TATACONSUM.NS',
        'DABUR.NS', 'MARICO.NS', 'COLPAL.NS', 'GODREJCP.NS', 'UBL.NS'
    ],
    "Oil, Gas & Consumable Fuels": [
        'RELIANCE.NS', 'ONGC.NS', 'IOC.NS', 'BPCL.NS', 'GAIL.NS',
        'HINDPETRO.NS', 'IGL.NS', 'PETRONET.NS', 'OIL.NS'
    ],
    "Automobile & Auto Components": [
        'MARUTI.NS', 'TATAMOTORS.NS', 'M&M.NS', 'BAJAJ-AUTO.NS', 'HEROMOTOCO.NS',
        'EICHERMOT.NS', 'TVSMOTOR.NS', 'BHARATFORG.NS', 'BOSCHLTD.NS', 'ASHOKLEY.NS'
    ],
    "Healthcare": [
        'SUNPHARMA.NS', 'CIPLA.NS', 'DRREDDY.NS', 'APOLLOHOSP.NS', 'DIVISLAB.NS',
        'LUPIN.NS', 'AUROPHARMA.NS', 'TORNTPHARM.NS', 'ALKEM.NS', 'MAXHEALTH.NS'
    ],
    "Metals & Mining": [
        'TATASTEEL.NS', 'JSWSTEEL.NS', 'HINDALCO.NS', 'VEDL.NS', 'COALINDIA.NS',
        'HINDZINC.NS', 'NMDC.NS', 'JINDALSTEL.NS'
    ],
    "Consumer Durables": [
        'TITAN.NS', 'ASIANPAINT.NS', 'ULTRACEMCO.NS', 'GRASIM.NS', 'PIDILITIND.NS',
        'HAVELLS.NS', 'VOLTAS.NS', 'DIXON.NS', 'WHIRLPOOL.NS'
    ]
}

@st.cache_data(ttl=3600)
def get_macro_indicators():
    """Fetches key macroeconomic indicators for India."""
    indicators = {
        'NIFTY 50': '^NSEI', 'India VIX': '^INDIAVIX', 'USD/INR': 'INR=X', 'India 10Y Bond Yield': '^NS11'
    }
    data = {}
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5)
    for name, ticker in indicators.items():
        try:
            hist = yf.Ticker(ticker).history(start=start_date, end=end_date)
            if not hist.empty:
                last_close = hist['Close'].iloc[-1]
                prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else last_close
                change = last_close - prev_close
                percent_change = (change / prev_close) * 100 if prev_close != 0 else 0
                data[name] = {'value': f"{last_close:,.2f}", 'delta': f"{change:,.2f} ({percent_change:.2f}%)"}
            else: data[name] = {'value': 'N/A', 'delta': ''}
        except Exception: data[name] = {'value': 'Error', 'delta': ''}
    return data

@st.cache_data(ttl=3600)
def get_stock_data(tickers, filter_by_age=False):
    """Fetches key financial data for a list of stock tickers, with an option to filter by age."""
    stock_data = []
    failed_tickers = []
    age_filtered_count = 0
    
    # Define the date 20 years ago from today
    cutoff_date = datetime.now() - timedelta(days=20 * 365.25)

    progress_bar = st.progress(0, text="Fetching stock data...")
    for i, ticker_symbol in enumerate(tickers):
        try:
            ticker = yf.Ticker(ticker_symbol)
            
            if filter_by_age:
                hist = ticker.history(period="max")
                if hist.empty or hist.index[0].to_pydatetime().replace(tzinfo=None) > cutoff_date:
                    age_filtered_count += 1
                    continue # Skip this stock as it's not old enough or has no history

            info = ticker.info
            if 'sector' in info and info.get('trailingPE') is not None:
                stock_data.append({
                    'Ticker': ticker_symbol, 'Company Name': info.get('shortName', 'N/A'),
                    'Sector': info.get('sector', 'N/A'), 'PE Ratio': info.get('trailingPE'),
                    'Dividend Yield (%)': (info.get('dividendYield') or 0) * 100,
                    'Current Price': info.get('regularMarketPrice', 0)
                })
        except Exception:
            failed_tickers.append(ticker_symbol)
        
        progress_bar.progress((i + 1) / len(tickers), text=f"Processing {ticker_symbol}...")
    
    progress_bar.empty()
    return pd.DataFrame(stock_data), failed_tickers, age_filtered_count

# --- UI Layout and Logic ---
st.title("📈 Indian Stock Market Recommender")
st.markdown(f"""
Welcome to the Stock Recommender System for the Indian market (NSE). 
This tool analyzes stocks by sector to identify potentially undervalued investment opportunities. 
*Last updated: {datetime.now().strftime('%d-%b-%Y %I:%M %p IST')}*
""")

# --- Macroeconomic Indicators Section ---
st.header("🇮🇳 Macroeconomic Indicators")
with st.spinner("Fetching latest market data..."):
    macro_data = get_macro_indicators()
    cols = st.columns(len(macro_data))
    for col, (name, data) in zip(cols, macro_data.items()):
        col.metric(label=name, value=data['value'], delta=data['delta'])
st.divider()

# --- Main Analysis Section ---
st.header("🔍 Stock Analysis & Recommendations")

# Sidebar for user inputs
st.sidebar.header("⚙️ Analysis Parameters")
st.sidebar.info("Select a sector and adjust the parameters to refine your stock search.")

selected_sector = st.sidebar.selectbox(
    "1. Select a Sector to Analyze", options=list(STOCKS_BY_SECTOR.keys()), index=0
)
dividend_threshold = st.sidebar.slider(
    "2. Minimum Dividend Yield (%)", 0.0, 10.0, 2.0, 0.5,
    help="Stocks with a dividend yield above this value will be considered 'High Dividend'."
)
age_filter_enabled = st.sidebar.checkbox(
    "3. Filter for companies 20+ years in the market", value=True,
    help="Only include companies with a public trading history of over 20 years."
)

if st.button(f"🚀 Analyze {selected_sector} Sector", type="primary", use_container_width=True):
    tickers_to_fetch = STOCKS_BY_SECTOR[selected_sector]
    df, failed_tickers, age_filtered_count = get_stock_data(tickers_to_fetch, age_filter_enabled)

    # Display informative messages about the filtering process
    if failed_tickers: st.warning(f"Could not fetch data for: {', '.join(failed_tickers)}")
    if age_filter_enabled:
        st.info(f"Age filter applied: Found {len(tickers_to_fetch)} stocks in sector. Removed {age_filtered_count} younger companies. Analyzing {len(df)} established companies.")

    strong_buy_stocks, undervalued_stocks, high_dividend_stocks = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    if df.empty:
        st.error(f"No valid stock data to analyze for the {selected_sector} sector with the current filters.")
    else:
        with st.spinner(f"Analyzing {len(df)} stocks..."):
            df.dropna(subset=['PE Ratio', 'Current Price', 'Sector'], inplace=True)
            df = df[df['PE Ratio'] > 0] 

            if df.empty:
                st.warning("No stocks with valid positive P/E Ratios remain after cleaning.")
            else:
                sector_average_pe = df['PE Ratio'].mean()
                df['Sector Average PE'] = sector_average_pe
                df['Valuation'] = df.apply(lambda r: 'Undervalued' if r['PE Ratio'] < r['Sector Average PE'] else 'Overvalued', axis=1)
                
                undervalued_stocks = df[df['Valuation'] == 'Undervalued'].copy()
                high_dividend_stocks = df[df['Dividend Yield (%)'] > dividend_threshold].copy()
                strong_buy_stocks = undervalued_stocks[undervalued_stocks['Ticker'].isin(high_dividend_stocks['Ticker'])].copy()
        
        st.success("Analysis Complete!")
        st.subheader(f"📊 Analysis Summary for {selected_sector}")
        col1, col2, col3 = st.columns(3)
        col1.metric("Strong Buy Candidates", len(strong_buy_stocks))
        col2.metric("Undervalued Stocks", len(undervalued_stocks))
        col3.metric(f"High Dividend Stocks (> {dividend_threshold}%)", len(high_dividend_stocks))

        tab1, tab2, tab3, tab4 = st.tabs(["🏆 Strong Buy", "📉 Undervalued", "💰 High Dividend", "📄 Full Dataset"])
        
        with tab1:
            st.info(f"These **{len(strong_buy_stocks)}** stocks are potentially undervalued and offer a high dividend yield (above {dividend_threshold}%).")
            st.dataframe(strong_buy_stocks[['Ticker', 'Company Name', 'Current Price', 'PE Ratio', 'Sector Average PE', 'Dividend Yield (%)']].round(2), use_container_width=True) if not strong_buy_stocks.empty else st.warning("No stocks met both criteria.")
        with tab2:
            st.info(f"These **{len(undervalued_stocks)}** stocks have a P/E ratio lower than the sector average of **{df['Sector Average PE'].mean():.2f}**.")
            st.dataframe(undervalued_stocks[['Ticker', 'Company Name', 'Current Price', 'PE Ratio', 'Sector Average PE']].round(2), use_container_width=True) if not undervalued_stocks.empty else st.warning("No undervalued stocks found.")
        with tab3:
            st.info(f"These **{len(high_dividend_stocks)}** stocks have a dividend yield greater than **{dividend_threshold}%**.")
            st.dataframe(high_dividend_stocks[['Ticker', 'Company Name', 'Current Price', 'Dividend Yield (%)', 'PE Ratio']].round(2), use_container_width=True) if not high_dividend_stocks.empty else st.warning(f"No stocks found with a dividend yield above {dividend_threshold}%.")
        with tab4:
             st.info(f"Full, cleaned dataset for the **{selected_sector}** sector used for this analysis.")
             st.dataframe(df.round(2), use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Disclaimer:** This is a financial analysis tool and not investment advice. All data is provided by Yahoo Finance. Always conduct your own research before making any investment decisions.
""")

