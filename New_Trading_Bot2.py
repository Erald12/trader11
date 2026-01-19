# ============================
# IMPORTS
# ============================
import time
import ccxt
import torch
import pandas as pd
import numpy as np
from ta.trend import SMAIndicator
from datetime import datetime, timedelta, timezone
import okx.Trade as Trade
import okx.Account as Account

# ============================
# CONFIG
# ============================
SYMBOL_OKX = "XRP-USDT-SWAP"
TIMEFRAME = "30m"
INTERVAL_SECONDS = 1800  # 30 min
LEVERAGE = 15
STOP_LOSS_PCT = 0.03
MARGIN_MODE = "isolated"
LENGTH_THRESHOLD = 22  # 11 hours in 30-min bars
DEVICE = "cpu"


# ============================
# LOAD NEURAL HMM MODEL
# ============================
class NeuralEmissionHMM(torch.nn.Module):
    def __init__(self, input_dim, num_states):
        super().__init__()
        self.num_states = num_states
        self.input_dim = input_dim

        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU()
        )
        self.mu = torch.nn.Linear(64, num_states * input_dim)
        self.logvar = torch.nn.Linear(64, num_states * input_dim)

    def forward(self, x):
        h = self.net(x)
        mu = self.mu(h).view(-1, self.num_states, self.input_dim)
        logvar = self.logvar(h).view(-1, self.num_states, self.input_dim)
        return mu, logvar

# Load model and transition probabilities
NUM_STATES = 4
INPUT_DIM = 7  # age + 6 SMA features
model = NeuralEmissionHMM(INPUT_DIM, NUM_STATES).to(DEVICE)
model.load_state_dict(torch.load("best_neural_hmm.pth", map_location=DEVICE))
model.eval()

transition_probs = torch.load("transition_probs.pth", map_location=DEVICE)
log_transition = torch.log(transition_probs + 1e-12)

int2regime = {0: "Bull", 1: "Bear", 2: "Bull_Long", 3: "Bear_Long"}

# ============================
# OKX CONNECTION
# ============================

api = ''
sec = ''
passwd = ''

okx = ccxt.okx({
    "apiKey": api,
    "secret": sec,
    "password": passwd,
    "enableRateLimit": True
})

tradeAPI = Trade.TradeAPI(api, sec, passwd, False, "0")
accountAPI = Account.AccountAPI(api, sec, passwd, False, "0")

# ============================
# TRADING FUNCTIONS
# ============================
def set_leverage(symbol, leverage, marginmode):
    accountAPI.set_leverage(instId=symbol, lever=str(leverage), mgnMode=marginmode)

def get_latest_price(symbol):
    return okx.fetch_ticker(symbol)["last"]

def place_market_order(symbol, side, usdt):
    price = get_latest_price(symbol)
    size = round((usdt * LEVERAGE) / price, 2)
    return tradeAPI.place_algo_order(
            instId=symbol,  # Instrument ID
            tdMode="isolated",  # Isolated margin mode
            side=side,  # "buy" for long, "sell" for short
            ordType="trigger",  # Trigger order type
            sz=str(size),  # Number of contracts
            triggerPx=str(okx.fetch_ticker(symbol)["last"]),  # Trigger price for the order to be activated
            orderPx=-1,
            triggerPxType="mark",
    )

def place_stop_loss(symbol, side, price, size):
    return tradeAPI.place_algo_order(
        instId=symbol,
        tdMode=MARGIN_MODE,
        side=side,
        ordType="conditional",
        sz=str(size),
        slTriggerPx=str(price),
        slOrdPx=-1,
        slTriggerPxType="mark"
    )

# ============================
# FEATURE ENGINE
# ============================
def compute_features(df):
    f1 = (df["SMA4"] - df["SMA6"]) / df["SMA6"]
    f2 = (df["SMA4"] - df["SMA15"]) / df["SMA15"]
    f3 = (df["SMA6"] - df["SMA15"]) / df["SMA15"]

    df1 = f1.diff()
    df2 = f2.diff()
    df3 = f3.diff()

    # Stack as array (T x 6)
    return np.vstack([f1, f2, f3, df1, df2, df3]).T.astype(np.float32)[1:]

# ============================
# REGIME LABELING
# ============================
def compute_regime_labels(df):
    regimes = np.where(df["SMA4"] > df["SMA6"], "Bull", "Bear")
    labeled = regimes.copy()

    start = 0
    while start < len(regimes):
        cur = regimes[start]
        end = start
        while end + 1 < len(regimes) and regimes[end + 1] == cur:
            end += 1

        if (end - start + 1) >= LENGTH_THRESHOLD:
            labeled[start:end + 1] = f"{cur}_Long"
        start = end + 1

    df["regime_labeled"] = labeled
    return df

# ============================
# GAUSSIAN LOG-PROBABILITY
# ============================
def gaussian_log_prob(x, mu, logvar):
    return -0.5 * (
        logvar + (x.unsqueeze(1) - mu) ** 2 / torch.exp(logvar) + np.log(2 * np.pi)
    ).sum(dim=2)

# ============================
# SYNC TO BAR CLOSE
# ============================
def wait_until_next_30min_bar():
    now = datetime.now(timezone.utc)

    # Determine next 30-min mark
    if now.minute < 30:
        next_bar = now.replace(minute=30, second=0, microsecond=0)
    else:
        next_bar = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)

    sleep_sec = (next_bar - now).total_seconds()
    print(f"[SYNC] Sleeping {sleep_sec:.1f}s until next 30-min candle...")
    time.sleep(sleep_sec)

# ============================
# MAIN LOOP
# ============================
position_open = False
position_side = None
entry_time = None
stop_loss_id = None
prev_log_alpha = torch.log(torch.ones(NUM_STATES) / NUM_STATES)  # uniform initial

wait_until_next_30min_bar()

time.sleep(10)

while True:
    try:
        # Fetch OHLCV
        candles = okx.fetch_ohlcv(SYMBOL_OKX, timeframe=TIMEFRAME, limit=120)
        df = pd.DataFrame(candles, columns=["t","o","h","l","c","v"])
        df["close"] = df["c"]

        # SMA
        df["SMA4"] = SMAIndicator(df["close"], 4).sma_indicator()
        df["SMA6"] = SMAIndicator(df["close"], 6).sma_indicator()
        df["SMA15"] = SMAIndicator(df["close"], 15).sma_indicator()
        df.dropna(inplace=True)

        # Regime labeling
        df = compute_regime_labels(df)

        # Features and age
        feats = compute_features(df)
        ages = np.zeros(len(df), dtype=np.float32)
        age = 1
        prev_state = None
        for i, r in enumerate(df["regime_labeled"]):
            if prev_state == r:
                age += 1
            else:
                age = 1
                prev_state = r
            ages[i] = np.log1p(age)

        # Prepare current timestep features
        x_t = torch.tensor(
            np.hstack([ages[-2], feats[-2]]),
            dtype=torch.float32
        ).unsqueeze(0)

        # Neural HMM forward filter
        with torch.no_grad():
            mu, logvar = model(x_t)
            log_emission = gaussian_log_prob(x_t, mu, logvar).squeeze(0)
            log_alpha = torch.logsumexp(prev_log_alpha.unsqueeze(1) + log_transition, dim=0) + log_emission
            log_alpha -= torch.logsumexp(log_alpha, dim=0)  # normalize
            alpha = torch.exp(log_alpha)
            state = torch.argmax(alpha).item()
            predicted_regime = int2regime[state]
            prev_log_alpha = log_alpha

        # Entry signals (SMA crossover)
        long_signal = (
            predicted_regime == "Bull_Long"
            and df["SMA4"].iloc[-2] <= df["SMA6"].iloc[-2]
            and df["SMA4"].iloc[-1] > df["SMA6"].iloc[-1]
        )
        short_signal = (
            predicted_regime == "Bear_Long"
            and df["SMA4"].iloc[-2] >= df["SMA6"].iloc[-2]
            and df["SMA4"].iloc[-1] < df["SMA6"].iloc[-1]
        )

        # Entry
        if (long_signal or short_signal) and not position_open:
            balance = okx.fetch_balance()
            usdt = balance["total"]["USDT"]

            set_leverage(SYMBOL_OKX, LEVERAGE, MARGIN_MODE)
            side = "buy" if long_signal else "sell"
            place_market_order(SYMBOL_OKX, side, usdt)
            print(f'Order placed at side: {side}')

            while True:
                pos = okx.fetch_positions([SYMBOL_OKX])
                if len(pos)>0:
                    for position in pos:
                        entry_price = position["avgPx"]
                        size = position["contracts"]

                        # Stop-loss
                        sl_price = entry_price * (1 - STOP_LOSS_PCT / LEVERAGE) if long_signal else entry_price * (1 + STOP_LOSS_PCT / LEVERAGE)
                        sl_side = "sell" if long_signal else "buy"
                        sl = place_stop_loss(SYMBOL_OKX, sl_side, sl_price, size)
                        stop_loss_id = sl["data"][0]["algoId"]

                        position_open = True
                        position_side = "long" if long_signal else "short"
                        entry_time = datetime.now(timezone.utc)

                        print(f'{entry_time}')
                        print(f"Order Filled: Side={position_side.upper()} | Regime={predicted_regime} | Price={entry_price}")
                    break
                else:
                    print('Order not filled')

        # Exit after LENGTH_THRESHOLD bars
        if position_open:
            bars_held = (datetime.now(timezone.utc) - entry_time).total_seconds() / INTERVAL_SECONDS
            if bars_held >= LENGTH_THRESHOLD:
                tradeAPI.close_positions(instId=SYMBOL_OKX, mgnMode=MARGIN_MODE)
                if stop_loss_id:
                    tradeAPI.cancel_algo_order([{"instId": SYMBOL_OKX, "algoId": stop_loss_id}])
                print(f"[EXIT] TIME EXIT | Held={bars_held:.1f} bars")
                position_open = False
                position_side = None
                entry_time = None
                stop_loss_id = None

        if position_open == False:
            print(f'Waiting for Signal, Predicted Regime: {predicted_regime}, XRPUSDT.P: {okx.fetch_ticker(SYMBOL_OKX)["last"]}')

    except Exception as e:
        print(f"[ERROR] {datetime.now(timezone.utc)} - {e}")

    wait_until_next_30min_bar()
    time.sleep(10)
