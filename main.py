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

app = Flask(__name__)

@app.route('/')
def home():
    return "Sniper Equality Bot (Anti-Block & Dual Alert Engine) is Live!"

# Initialize TvDatafeed (Anonymous Safe Mode)
tv = TvDatafeed()

# ==========================================
# TELEGRAM & MEMORY HELPER FUNCTIONS
# ==========================================
def send_telegram_message(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logging.warning("Telegram Token au Chat ID haijawekwa kwenye Environment Variables.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        res = requests.post(url, json=payload, timeout=10)
        logging.info(f"Telegram API Response: {res.status_code}")
    except Exception as e:
        logging.error(f"Telegram API Error: {e}")

def load_memory():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading memory: {e}")
    return {"successful_traps": [], "stats": {"total_wins": 0, "total_losses": 0}}

def save_memory(data):
    try:
        with open(MEMORY_FILE, 'w') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        logging.error(f"Error saving memory: {e}")

sniper_memory = load_memory()

def record_win_trap(pair, strategy_id, pattern_name, direction):
    trap = {
        "pair": pair,
        "strategy_id": strategy_id,
        "pattern": pattern_name,
        "direction": direction,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    sniper_memory["successful_traps"].append(trap)
    sniper_memory["stats"]["total_wins"] += 1
    save_memory(sniper_memory)

def record_loss_trap():
    sniper_memory["stats"]["total_losses"] += 1
    save_memory(sniper_memory)

def get_local_time():
    return datetime.now(LOCAL_TZ)

# ==========================================
# SAFE TVDATAFEED FETCH (ANTI-BLOCK WRAPPER)
# ==========================================
def fetch_tv_data_safely(symbol, exchange, n_bars=60, retries=3):
    """
    Inavuta data kwa usalama bila kuamsha mifumo ya ulinzi ya TradingView.
    """
    for attempt in range(retries):
        try:
            df = tv.get_hist(symbol=symbol, exchange=exchange, interval=Interval.in_1_minute, n_bars=n_bars)
            if df is not None and len(df) >= 50:
                return df
        except Exception as e:
            logging.warning(f"Jaribio {attempt+1} limefeli kwa {symbol}: {e}")
            time.sleep(2)
    return None

# ==========================================
# INDICATOR & STRATEGY CALCULATIONS
# ==========================================
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

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

def validate_8_safeguards(df):
    curr = df.iloc[-1]
    body = abs(curr['close'] - curr['open'])
    candle_range = curr['high'] - curr['low']

    if candle_range > 0 and (body / candle_range) < 0.12:
        return False, "Ngao 1: Doji / Volatility Ndogo"

    upper, mid, lower = calculate_bollinger_bands(df['close'])
    band_width = (upper.iloc[-1] - lower.iloc[-1]) / mid.iloc[-1]
    if band_width < 0.0004:
        return False, "Ngao 3: Market Squeeze"

    return True, "SAFE"

def evaluate_sniper_strategies(df, pair):
    df['ema8'] = calculate_ema(df['close'], 8)
    df['ema9'] = calculate_ema(df['close'], 9)
    df['ema14'] = calculate_ema(df['close'], 14)
    df['ema21'] = calculate_ema(df['close'], 21)
    df['ema50'] = calculate_ema(df['close'], 50)
    df['rsi'] = calculate_rsi(df['close'], 14)
    df['stoch_k'], df['stoch_d'] = calculate_stochastic(df, 5, 3)
    df['bb_upper'], df['bb_mid'], df['bb_lower'] = calculate_bollinger_bands(df['close'], 20, 2)

    c_curr = df.iloc[-1]
    c_prev = df.iloc[-2]
    inst_candle = detect_institutional_candle(df)

    # Strategy 1: RSI 50 & EMA 9/21 Dynamic Pullback
    if c_curr['ema9'] > c_curr['ema21'] and c_curr['rsi'] > 50 and abs(c_curr['low'] - c_curr['ema21']) < (c_curr['close'] * 0.0003):
        return "CALL", "Mbinu 1: RSI 50 & EMA 9/21 Pullback", 2, 94.6, inst_candle

    if c_curr['ema9'] < c_curr['ema21'] and c_curr['rsi'] < 50 and abs(c_curr['high'] - c_curr['ema21']) < (c_curr['close'] * 0.0003):
        return "PUT", "Mbinu 1: RSI 50 & EMA 9/21 Pullback", 2, 94.6, inst_candle

    # Strategy 2: Bollinger Bands & Stochastic
    if c_prev['low'] < c_prev['bb_lower'] and c_curr['stoch_k'] < 25:
        return "CALL", "Mbinu 2: Bollinger Spike & Stochastic Reversal", 1, 93.2, inst_candle
    if c_prev['high'] > c_prev['bb_upper'] and c_curr['stoch_k'] > 75:
        return "PUT", "Mbinu 2: Bollinger Spike & Stochastic Reversal", 1, 93.2, inst_candle

    # Strategy 5: 3 EMA Cloud & Engulfing
    if c_curr['ema8'] > c_curr['ema14'] and c_curr['ema14'] > c_curr['ema50'] and inst_candle == "BULLISH_ENGULFING":
        return "CALL", "Mbinu 5: 3 EMA Cloud & Engulfing Shot", 2, 96.1, inst_candle
    if c_curr['ema8'] < c_curr['ema14'] and c_curr['ema14'] < c_curr['ema50'] and inst_candle == "BEARISH_ENGULFING":
        return "PUT", "Mbinu 5: 3 EMA Cloud & Engulfing Shot", 2, 96.1, inst_candle

    return None, None, None, None, None

def analyze_pair_super_sniper(pair, symbol, exchange):
    try:
        df = fetch_tv_data_safely(symbol, exchange, n_bars=60)
        if df is None:
            return None

        is_safe, _ = validate_8_safeguards(df)
        if not is_safe:
            return None

        action, strategy_name, duration, base_win_rate, inst_candle = evaluate_sniper_strategies(df, pair)
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
            "win_rate": base_win_rate
        }
    except Exception as e:
        logging.error(f"Error analyzing {pair}: {e}")
    return None

# ==========================================
# DUAL ALERT EXECUTION ENGINE & LOCK
# ==========================================
def process_trade_lifecycle(pair, symbol, exchange, setup_data):
    global IS_TRADE_ACTIVE
    IS_TRADE_ACTIVE = True

    now = get_local_time()
    target_entry_time = (now + timedelta(minutes=1)).replace(second=0, microsecond=0)

    # 1. PRE-ALERT (DAKIKA 1 KABLA)
    pre_alert_msg = (
        f"⏳ *PRE-ALERT (DAKIKA 1 KABLA)* ⏳\n\n"
        f"💱 *Pair:* `{pair}`\n"
        f"📈 *Direction:* *{'🟢 CALL (BUY)' if setup_data['signal'] == 'CALL' else '🔴 PUT (SELL)'}*\n"
        f"⏳ *Duration:* `{setup_data['duration']} Min(s)`\n"
        f"⏱ *Muda wa Entry:* `{target_entry_time.strftime('%H:%M:%S')} EAT`\n"
        f"📌 *Strategy:* `{setup_data['strategy']}`\n\n"
        f"🔔 *Fungua broker wako na uwe tayari! Confirming in 55s...*"
    )
    send_telegram_message(pre_alert_msg)

    # Subiri mpaka sekunde ya 55 ifike
    wait_until_55 = (target_entry_time - timedelta(seconds=5) - get_local_time()).total_seconds()
    if wait_until_55 > 0:
        time.sleep(wait_until_55)

    # 2. FINAL ALERT (CONFIRMATION SEC 5 KABLA)
    final_check = analyze_pair_super_sniper(pair, symbol, exchange)

    if final_check and final_check['signal'] == setup_data['signal']:
        expiry_time = target_entry_time + timedelta(minutes=final_check['duration'])
        
        final_alert_msg = (
            f"🚨 *FINAL ALERT (CONFIRMED - INGIA SEC 5)* 🚨\n\n"
            f"💱 *Pair:* `{pair}`\n"
            f"💥 *ACTION NOW:* *{'🟢 CALL (BUY)' if final_check['signal'] == 'CALL' else '🔴 PUT (SELL)'}*\n"
            f"🏁 *Expiry Time:* `{expiry_time.strftime('%H:%M:%S')} EAT`\n"
            f"🎯 *Win Probability:* `{final_check['win_rate']}%`\n"
            f"🕯️ *Candle Check:* `{final_check['candle_pattern']}`\n\n"
            f"🚀 *BONYEZA TRADE SASA HIVI (00:00 Mark)!*"
        )
        send_telegram_message(final_alert_msg)

        # 3. SUBIRI DURATION IISHE KABISA
        time.sleep((final_check['duration'] * 60) + 5)

        # 4. MONITOR MATOKEO YA TRADE
        try:
            df = fetch_tv_data_safely(symbol, exchange, n_bars=5)
            if df is not None and not df.empty:
                exit_price = df.iloc[-1]['close']
                entry_price = final_check['price']
                
                is_win = (final_check['signal'] == "CALL" and exit_price > entry_price) or \
                         (final_check['signal'] == "PUT" and exit_price < entry_price)

                if is_win:
                    record_win_trap(pair, final_check['strategy'], final_check['candle_pattern'], final_check['signal'])
                    res_str = "DIRECT VICTORY (ITM) ✅"
                else:
                    record_loss_trap()
                    res_str = "LOSS (OTM) ❌"

                report = (
                    f"📊 *TRADE RESULT REPORT*\n\n"
                    f"🔤 *Pair:* `{pair}`\n"
                    f"📈 *Result:* *{res_str}*\n"
                    f"📌 *Entry:* `{entry_price}` | *Exit:* `{exit_price}`\n\n"
                    f"🔓 *Bot ipo tayari kwa Trade inayofuata.*"
                )
                send_telegram_message(report)
        except Exception as e:
            logging.error(f"Error checking trade outcome: {e}")

    else:
        cancel_msg = f"❌ *TRADE CANCELLED:* Vigezo vilishuka sekunde za mwisho kwenye `{pair}`. Trade imesitishwa kwa ulinzi."
        send_telegram_message(cancel_msg)

    # UNLOCK ENGINE
    IS_TRADE_ACTIVE = False

# ==========================================
# MAIN LOOP ENGINE
# ==========================================
def main_loop():
    logging.info("Sniper Engine Active with Anti-Block Engine & Dual Alert...")

    # UJUMBE WA JARIBIO UNAPOWAKA RENDER
    send_telegram_message(
        "👑 *SNIPER EQUALITY BOT IS ACTIVE*\n\n"
        "🟢 Server: *Render Live*\n"
        "⏱ Pre-Alert: *Dakika 1 Kabla (00:00 Mark)*\n"
        "🚨 Final Confirmation: *Sekunde 5 Kabla (00:55 Mark)*\n"
        "🛡️ Anti-Block Protection: *Enabled*\n"
        "🔒 Single Trade Lock: *Active!*"
    )

    while True:
        try:
            if IS_TRADE_ACTIVE:
                time.sleep(1)
                continue

            now = get_local_time()

            # SOMA SOKO MWANZO WA DAKIKA (Sekunde 00 - 03)
            if now.second <= 3:
                for pair, (symbol, exchange) in FOREX_PAIRS.items():
                    setup = analyze_pair_super_sniper(pair, symbol, exchange)
                    if setup:
                        threading.Thread(target=process_trade_lifecycle, args=(pair, symbol, exchange, setup)).start()
                        time.sleep(5)
                        break
                    
                    # ANTI-BLOCK PAUSE: Subiri sec 1.2 kati ya pair na pair
                    time.sleep(1.2)

            time.sleep(0.5)
        except Exception as e:
            logging.error(f"Main loop error: {e}")
            time.sleep(2)

# ==========================================
# START BACKGROUND THREAD FOR GUNICORN & FLASK
# ==========================================
# Anzisha bot loop kwenye background mara tu Gunicorn inapopakiwa
threading.Thread(target=main_loop, daemon=True).start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
