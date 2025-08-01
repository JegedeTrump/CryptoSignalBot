import os
import requests
import numpy as np
import pandas as pd
import asyncio
from datetime import datetime
from telegram import Bot
from dotenv import load_dotenv

load_dotenv()

# Configuration
PAIRS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]
TIMEFRAME = "4h"  # 1h, 4h, 1d
ANALYSIS_WINDOW = 100  # Candles to analyze

class SignalGenerator:
    @staticmethod
    def calculate_ema(series, period):
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def calculate_rsi(series, period=14):
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_atr(high, low, close, period=14):
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def generate_signal(self, ohlc_data):
        close = ohlc_data['close']
        high = ohlc_data['high']
        low = ohlc_data['low']

        # Indicators
        ema20 = self.calculate_ema(close, 20)
        ema50 = self.calculate_ema(close, 50)
        rsi = self.calculate_rsi(close)
        atr = self.calculate_atr(high, low, close)

        current_close = close.iloc[-1]
        current_rsi = rsi.iloc[-1]
        current_atr = atr.iloc[-1]

        # Signal Logic
        signals = []
        confidence = 0

        # Bullish EMA Crossover
        if ema20.iloc[-1] > ema50.iloc[-1] and ema20.iloc[-2] <= ema50.iloc[-2]:
            entry = current_close
            tp1 = entry + (1.5 * current_atr)
            tp2 = entry + (2.5 * current_atr)
            sl = entry - (1 * current_atr)
            confidence = min(80 + (current_rsi - 30), 95)
            signals.append(("BUY", entry, [tp1, tp2], sl, confidence))

        # Bearish EMA Crossover
        elif ema20.iloc[-1] < ema50.iloc[-1] and ema20.iloc[-2] >= ema50.iloc[-2]:
            entry = current_close
            tp1 = entry - (1.5 * current_atr)
            tp2 = entry - (2.5 * current_atr)
            sl = entry + (1 * current_atr)
            confidence = min(80 + (70 - current_rsi), 95)
            signals.append(("SELL", entry, [tp1, tp2], sl, confidence))

        return signals

async def fetch_ohlcv(symbol):
    """Fetch OHLCV data from Binance API"""
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol.replace("/", ""),
        "interval": TIMEFRAME,
        "limit": ANALYSIS_WINDOW
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_volume',
            'taker_buy_quote_volume', 'ignore'
        ])
        df = df.apply(pd.to_numeric)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None

async def analyze_markets():
    generator = SignalGenerator()
    all_signals = []
    
    for pair in PAIRS:
        ohlc = await fetch_ohlcv(pair)
        if ohlc is not None:
            signals = generator.generate_signal(ohlc)
            for signal in signals:
                all_signals.append({
                    "pair": pair,
                    "direction": signal[0],
                    "entry": signal[1],
                    "take_profit": signal[2],
                    "stop_loss": signal[3],
                    "confidence": signal[4]
                })
    
    return sorted(all_signals, key=lambda x: x["confidence"], reverse=True)

async def send_signals():
    signals = await analyze_markets()
    if not signals:
        return
    
    bot = Bot(token=os.getenv("TELEGRAM_TOKEN"))
    
    message = "🚀 *PROFESSIONAL TRADING SIGNALS* 🚀\n\n"
    message += f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC\n"
    message += f"📊 Timeframe: {TIMEFRAME}\n\n"
    
    for signal in signals[:3]:  # Top 3 signals
        emoji = "🟢" if signal["direction"] == "BUY" else "🔴"
        message += (
            f"{emoji} *{signal['pair']} {signal['direction']}*\n"
            f"🎯 Entry: ${signal['entry']:.2f}\n"
            f"💰 Take Profit: ${signal['take_profit'][0]:.2f} → ${signal['take_profit'][1]:.2f}\n"
            f"🛑 Stop Loss: ${signal['stop_loss']:.2f}\n"
            f"📈 Confidence: {signal['confidence']:.0f}%\n\n"
        )
    
    message += (
        "⚡ *Risk Management*\n"
        "- Position Size: 1-2% of capital\n"
        "- Split TP: 50% at TP1, 50% at TP2\n"
        "- Move SL to breakeven at TP1\n\n"
        "#TradingSignals #Crypto"
    )
    
    await bot.send_message(
        chat_id=os.getenv("TELEGRAM_CHAT_ID"),
        text=message,
        parse_mode="Markdown"
    )

async def run_bot():
    """Main async function to run the bot continuously"""
    print("🚀 Crypto Signal Bot Started")
    while True:
        try:
            await send_signals()
            print(f"⏳ Next analysis in 4 hours at {(datetime.now() + pd.Timedelta(hours=4)).strftime('%Y-%m-%d %H:%M')}")
            await asyncio.sleep(4 * 3600)  # 4 hour interval
        except Exception as e:
            print(f"⚠️ Error in main loop: {e}")
            await asyncio.sleep(60)  # Wait 1 minute before retry

def main():
    """Synchronous entry point"""
    try:
        asyncio.run(run_bot())
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user")

if __name__ == "__main__":
    main()
