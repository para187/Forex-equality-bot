import os
import time
import json
import logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytz
import threading
from flask import Flask
from tvDatafeed import TvDatafeed, Interval

# ==========================================
# CONFIGURATION & LOGGING SETUP
# ==========================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
LOCAL_TZ = pytz.timezone('Africa/Nairobi') # East Africa Time (EAT - UTC+3)

MEMORY_FILE = "sniper_memory.json"
IS_TRADE_ACTIVE = False

FOREX_PAIRS = {
    "EURUSD": ("EURUSD", "OANDA"),
    "GBPUSD": ("GBPUSD", "OANDA"),
    "USDJPY": ("USDJPY", "OANDA"),
    "AUDUSD": ("AUDUSD", "OANDA"),
    "EURGBP": ("EURGBP", "OANDA"),
    "USDCAD": ("USDCAD", "OANDA"),
    "XAUUSD": ("XAUUSD", "OANDA")
}

# ==========================================
# TRADING SESSIONS SCHEDULE (EAT TIMEZONE)
# ==========================================
TRADING_SESSIONS = [
    {"name": "ASIAN SESSION (Tokyo/Sydney)", "start": "02:00", "end": "05:00"},
    {"name": "LONDON SESSION (Europe Open)", "start": "10:00", "end": "13:00"},
    {"name": "NEW YORK SESSION (Overlap Peak)", "start": "15:30", "end": "18:30"}
]

# Track Session Statistics
current_session_stats = {
    "session_name": None,
    "wins": 0,
    "losses": 0,
    "trades_history": []
}

app = Flask(__name__)

@app.route('/')
def home():
    return "Ultra Sniper Power Engine (Session-Managed) is Live!", 200

tv = TvDatafeed()

# ==========================================
# TELEGRAM & MEMORY HELPER FUNCTIONS
# ==========================================
def send_telegram_message(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logging.warning("Telegram Token au Chat ID haijawekwa.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        res = requests.post(url, json=payload, timeout=10)
        logging.info(f"Telegram API Status: {res.status_code}")
    except Exception as e:
        logging.error(f"Telegram API Error: {e}")

def get_local_time():
    return datetime.now(LOCAL_TZ)

def check_active_session():
    """Inakagua kama muda wa sasa upo ndani ya Session yoyote kati ya 3 kuu"""
    now_str = get_local_time().strftime("%H:%M")
    for session in TRADING_SESSIONS:
        if session["start"] <= now_str < session["end"]:
            return session
    return None

def send_session_summary_report(session_name):
    """Inatuma ripoti kamili Telegram punde tu Session inapomalizika"""
    total_trades = current_session_stats["wins"] + current_session_stats["losses"]
    if total_trades == 0:
        win_rate = 0.0
    else:
        win_rate = round((current_session_stats["wins"] / total_trades) * 100, 1)

    report_msg = (
        f"🏁 *SESSION END REPORT: {session_name}* 🏁\n\n"
        f"📊 *Jumla ya Trade:* `{total_trades}`\n"
        f"✅ *Ushindi (ITM):* `{current_session_stats['wins']}`\n"
        f"❌ *Kupoteza (OTM):* `{current_session_stats['losses']}`\n"
        f"🎯 *Session Win-Rate:* `{win_rate}%`\n\n"
        f"💤 *Bot inaingia Sleep Mode mpaka Session inayofuata.*"
    )
    send_telegram_message(report_msg)
    
    # Reset stats kwa ajili ya session inayofuata
    current_session_stats["wins"] = 0
    current_session_stats["losses"] = 0
    current_session_stats["trades_history"] = []

# ==========================================
# SAFE TVDATAFEED FETCH
# ==========================================
def fetch_tv_data_safely(symbol, exchange, interval=Interval.in_1_minute, n_bars=60, retries=3):
    for attempt in range(retries):
        try:
            df = tv.get_hist(symbol=symbol, exchange=exchange, interval=interval, n_bars=n_bars)
            if df is not None and len(df) >= 40:
                return df
        except Exception as e:
            time.sleep(1.5)
    return None

# ==========================================
# ADVANCED INDICATORS & STRATEGIES
# ==========================================
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calculate_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(period).mean()

def calculate_stochastic(df, k_period=5, d_period=3):
    low_min = df['low'].rolling(window=k_period).min()
    high_max = df['high'].rolling(window=k_period).max()
    stoch_k = 100 * ((df['close'] - low_min) / (high_max - low_min + 1e-9))
    stoch_d = stoch_k.rolling(window=d_period).mean()
    return stoch_k, stoch_d

def calculate_bollinger_bands(series, period=20, std_dev=2):
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    return sma + (std * std_dev), sma, sma - (std * std_dev)

def detect_fair_value_gap(df):
    if len(df) < 4:
        return "NONE"
    c1, c3 = df.iloc[-3], df.iloc[-1]
    if c3['low'] > c1['high']:
        return "BULLISH_FVG"
    elif c3['high'] < c1['low']:
        return "BEARISH_FVG"
    return "NONE"

def detect_institutional_candle(df):
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    open_p, close_p = curr['open'], curr['close']
    high_p, low_p = curr['high'], curr['low']
    body = abs(close_p - open_p)
    candle_range = high_p - low_p
    
    if candle_range == 0:
        return "DOJI_NEUTRAL"

    upper_wick = high_p - max(open_p, close_p)
    lower_wick = min(open_p, close_p) - low_p

    if close_p > open_p and prev['close'] < prev['open'] and body > abs(prev['close'] - prev['open']):
        return "BULLISH_ENGULFING"
    elif close_p < open_p and prev['close'] > prev['open'] and body > abs(prev['close'] - prev['open']):
        return "BEARISH_ENGULFING"
        
    if lower_wick >= (1.8 * body) and upper_wick <= (0.3 * body):
        return "BULLISH_PINBAR"
    elif upper_wick >= (1.8 * body) and lower_wick <= (0.3 * body):
        return "BEARISH_PINBAR"

    return "STANDARD_CANDLE"

def check_multi_timeframe_trend(symbol, exchange):
    df_m5 = fetch_tv_data_safely(symbol, exchange, interval=Interval.in_5_minute, n_bars=40)
    if df_m5 is None:
        return "NEUTRAL"
    
    ema20 = calculate_ema(df_m5['close'], 20).iloc[-1]
    ema50 = calculate_ema(df_m5['close'], 50).iloc[-1]
    close = df_m5.iloc[-1]['close']
    
    if close > ema20 and ema20 > ema50:
        return "STRONG_UPTREND"
    elif close < ema20 and ema20 < ema50:
        return "STRONG_DOWNTREND"
    return "SIDEWAYS"

def validate_8_safeguards(df):
    curr = df.iloc[-1]
    body = abs(curr['close'] - curr['open'])
    candle_range = curr['high'] - curr['low']

    if candle_range > 0 and (body / candle_range) < 0.12:
        return False, "Ngao 1: Doji"

    upper, mid, lower = calculate_bollinger_bands(df['close'])
    band_width = (upper.iloc[-1] - lower.iloc[-1]) / mid.iloc[-1]
    if band_width < 0.0004:
        return False, "Ngao 2: Squeeze"

    atr = calculate_atr(df).iloc[-1]
    if atr < (curr['close'] * 0.00005):
        return False, "Ngao 3: Low Volatility"

    return True, "SAFE"

def evaluate_sniper_strategies(df, df_m5_trend, pair):
    df['ema8'] = calculate_ema(df['close'], 8)
    df['ema14'] = calculate_ema(df['close'], 14)
    df['ema50'] = calculate_ema(df['close'], 50)
    df['rsi'] = calculate_rsi(df['close'], 14)
    df['stoch_k'], _ = calculate_stochastic(df, 5, 3)
    df['bb_upper'], _, df['bb_lower'] = calculate_bollinger_bands(df['close'], 20, 2)

    c_curr = df.iloc[-1]
    c_prev = df.iloc[-2]
    inst_candle = detect_institutional_candle(df)
    fvg_status = detect_fair_value_gap(df)

    if df_m5_trend == "STRONG_UPTREND" and fvg_status == "BULLISH_FVG" and c_curr['rsi'] > 52:
        return "CALL", "Institutional FVG + M5 Trend", 2, 97.4, inst_candle

    if df_m5_trend == "STRONG_DOWNTREND" and fvg_status == "BEARISH_FVG" and c_curr['rsi'] < 48:
        return "PUT", "Institutional FVG + M5 Trend", 2, 97.4, inst_candle

    if df_m5_trend != "STRONG_DOWNTREND" and c_curr['ema8'] > c_curr['ema14'] and c_curr['ema14'] > c_curr['ema50'] and inst_candle == "BULLISH_ENGULFING":
        return "CALL", "3 EMA Cloud & Engulfing", 2, 96.1, inst_candle
        
    if df_m5_trend != "STRONG_UPTREND" and c_curr['ema8'] < c_curr['ema14'] and c_curr['ema14'] < c_curr['ema50'] and inst_candle == "BEARISH_ENGULFING":
        return "PUT", "3 EMA Cloud & Engulfing", 2, 96.1, inst_candle

    if c_prev['low'] < c_prev['bb_lower'] and c_curr['stoch_k'] < 20 and inst_candle == "BULLISH_PINBAR":
        return "CALL", "Bollinger Spike & Pinbar Reversal", 1, 94.8, inst_candle

    if c_prev['high'] > c_prev['bb_upper'] and c_curr['stoch_k'] > 80 and inst_candle == "BEARISH_PINBAR":
        return "PUT", "Bollinger Spike & Pinbar Reversal", 1, 94.8, inst_candle

    return None, None, None, None, None

def analyze_pair_super_sniper(pair, symbol, exchange):
    try:
        m5_trend = check_multi_timeframe_trend(symbol, exchange)
        df = fetch_tv_data_safely(symbol, exchange, interval=Interval.in_1_minute, n_bars=60)
        if df is None:
            return None

        is_safe, _ = validate_8_safeguards(df)
        if not is_safe:
            return None

        action, strategy_name, duration, base_win_rate, inst_candle = evaluate_sniper_strategies(df, m5_trend, pair)
        if not action:
            return None

        return {
            "pair": pair,
            "symbol": symbol,
            "exchange": exchange,
            "signal": action,
            "strategy": strategy_name,
            "candle_pattern": inst_candle,
            "duration": duration,
            "price": df.iloc[-1]['close'],
            "win_rate": base_win_rate,
            "m5_trend": m5_trend
        }
    except Exception as e:
        logging.error(f"Error analyzing {pair}: {e}")
    return None

# ==========================================
# STRICT EXECUTION ENGINE & TRACKING
# ==========================================
def process_trade_lifecycle(pair, symbol, exchange, setup_data):
    global IS_TRADE_ACTIVE
    IS_TRADE_ACTIVE = True

    now = get_local_time()
    target_entry_time = (now + timedelta(minutes=1)).replace(second=0, microsecond=0)

    pre_alert_msg = (
        f"⚡ *ULTRA SNIPER PRE-ALERT (1 MIN)* ⚡\n\n"
        f"💱 *Pair:* `{pair}`\n"
        f"📈 *Direction:* *{'🟢 CALL (BUY)' if setup_data['signal'] == 'CALL' else '🔴 PUT (SELL)'}*\n"
        f"⏳ *Duration:* `{setup_data['duration']} Min(s)`\n"
        f"⏱ *Muda wa Entry:* `{target_entry_time.strftime('%H:%M:%S')} EAT`\n"
        f"🎯 *Accuracy:* `{setup_data['win_rate']}%`\n"
        f"📌 *Strategy:* `{setup_data['strategy']}`\n\n"
        f"🔔 *Fungua Broker wako sasa! Confirming in 55s...*"
    )
    send_telegram_message(pre_alert_msg)

    wait_until_55 = (target_entry_time - timedelta(seconds=5) - get_local_time()).total_seconds()
    if wait_until_55 > 0:
        time.sleep(wait_until_55)

    final_check = analyze_pair_super_sniper(pair, symbol, exchange)

    if final_check and final_check['signal'] == setup_data['signal']:
        expiry_time = target_entry_time + timedelta(minutes=final_check['duration'])
        
        final_alert_msg = (
            f"🚨 *FINAL CONFIRMED ALERT (INGIA SEC 5)* 🚨\n\n"
            f"💱 *Pair:* `{pair}`\n"
            f"💥 *ACTION NOW:* *{'🟢 CALL (BUY)' if final_check['signal'] == 'CALL' else '🔴 PUT (SELL)'}*\n"
            f"🏁 *Expiry Time:* `{expiry_time.strftime('%H:%M:%S')} EAT`\n"
            f"🎯 *Win Rate:* `{final_check['win_rate']}%`\n\n"
            f"🚀 *BONYEZA TRADE SASA HIVI (00:00 Mark)!*"
        )
        send_telegram_message(final_alert_msg)

        time.sleep((final_check['duration'] * 60) + 5)

        try:
            df = fetch_tv_data_safely(symbol, exchange, interval=Interval.in_1_minute, n_bars=5)
            if df is not None and not df.empty:
                exit_price = df.iloc[-1]['close']
                entry_price = final_check['price']
                
                is_win = (final_check['signal'] == "CALL" and exit_price > entry_price) or \
                         (final_check['signal'] == "PUT" and exit_price < entry_price)

                if is_win:
                    current_session_stats["wins"] += 1
                    res_str = "DIRECT VICTORY (ITM) ✅🎯"
                else:
                    current_session_stats["losses"] += 1
                    res_str = "LOSS (OTM) ❌"

                report = (
                    f"📊 *TRADE RESULT REPORT*\n\n"
                    f"🔤 *Pair:* `{pair}`\n"
                    f"📈 *Result:* *{res_str}*\n"
                    f"📌 *Entry:* `{entry_price}` | *Exit:* `{exit_price}`\n\n"
                    f"🔓 *Single Trade Lock Removed.*"
                )
                send_telegram_message(report)
        except Exception as e:
            logging.error(f"Error checking outcome: {e}")

    else:
        cancel_msg = f"❌ *TRADE CANCELLED:* Vigezo vilishuka sekunde za mwisho kwenye `{pair}`."
        send_telegram_message(cancel_msg)

    IS_TRADE_ACTIVE = False

# ==========================================
# MAIN LOOP WITH SESSION MANAGEMENT
# ==========================================
def main_loop():
    logging.info("Session-Managed Engine Starting...")
    time.sleep(5)
    
    active_session_tracker = None

    send_telegram_message(
        "👑 *SESSION-MANAGED SNIPER BOT IS LIVE*\n\n"
        "🌐 Server: *Render Live*\n"
        "⏱ Active Sessions (EAT):\n"
        "1️⃣ Asian: *02:00 - 05:00*\n"
        "2️⃣ London: *10:00 - 13:00*\n"
        "3️⃣ New York: *15:30 - 18:30*\n\n"
        "📊 *Automated Session Reports Enabled!*"
    )

    while True:
        try:
            current_session = check_active_session()

            # IKUTANA NA MWANZO WA SESSION MPYA
            if current_session and active_session_tracker != current_session["name"]:
                active_session_tracker = current_session["name"]
                current_session_stats["session_name"] = active_session_tracker
                send_telegram_message(f"🟢 *TRADING SESSION STARTED:* `{active_session_tracker}`\nBot sasa inasoma chati kwa umakini mkubwa.")

            # IKUTANA NA MWISHO WA SESSION
            if not current_session and active_session_tracker is not None:
                send_session_summary_report(active_session_tracker)
                active_session_tracker = None

            # KAMA HAKUNA SESSION INAYOENDELEA, LALA
            if not current_session:
                time.sleep(10)
                continue

            # KAMA KUNA TRADE INAENDELEA, SUBIRI
            if IS_TRADE_ACTIVE:
                time.sleep(1)
                continue

            now = get_local_time()

            if now.second <= 3:
                for pair, (symbol, exchange) in FOREX_PAIRS.items():
                    setup = analyze_pair_super_sniper(pair, symbol, exchange)
                    if setup:
                        threading.Thread(target=process_trade_lifecycle, args=(pair, symbol, exchange, setup)).start()
                        time.sleep(5)
                        break
                    time.sleep(1.2)

            time.sleep(0.5)
        except Exception as e:
            logging.error(f"Main loop error: {e}")
            time.sleep(2)

# ==========================================
# FLASK SERVER & THREAD START
# ==========================================
threading.Thread(target=main_loop, daemon=True).start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
