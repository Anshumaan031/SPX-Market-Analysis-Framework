import numpy as np
from typing import List, Tuple

def calculate_ema(data: List[float], period: int) -> List[float]:
    ema = [data[0]]
    multiplier = 2 / (period + 1)
    for i in range(1, len(data)):
        ema.append((data[i] - ema[-1]) * multiplier + ema[-1])
    return ema

def calculate_rsi(data: List[float], period: int = 14) -> List[float]:
    deltas = np.diff(data)
    seed = deltas[:period+1]
    up = seed[seed >= 0].sum()/period
    down = -seed[seed < 0].sum()/period
    rs = up/down
    rsi = np.zeros_like(data)
    rsi[:period] = 100. - 100./(1. + rs)

    for i in range(period, len(data)):
        delta = deltas[i - 1]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up*(period - 1) + upval)/period
        down = (down*(period - 1) + downval)/period
        rs = up/down
        rsi[i] = 100. - 100./(1. + rs)

    return rsi.tolist()

def calculate_macd(data: List[float], fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple[List[float], List[float]]:
    ema_fast = calculate_ema(data, fast_period)
    ema_slow = calculate_ema(data, slow_period)
    macd_line = [fast - slow for fast, slow in zip(ema_fast, ema_slow)]
    signal_line = calculate_ema(macd_line, signal_period)
    return macd_line, signal_line