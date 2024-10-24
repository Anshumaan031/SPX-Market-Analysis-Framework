# SPX Market Analysis Framework 🚀

A sophisticated Python-based tool for analyzing S&P 500 (SPX) Index data using various technical indicators and backtesting trading strategies. This project implements multiple analysis approaches including EMA (Exponential Moving Average), RSI (Relative Strength Index), and MACD (Moving Average Convergence Divergence) to identify market trends and optimize trading decisions.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Dependencies](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)

## 📋 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technical Implementation](#technical-implementation)
- [Configuration](#configuration)
- [Testing](#testing)
- [Performance Analysis](#performance-analysis)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

- **Data Processing**
  - Efficient CSV data loading and validation
  - Robust error handling for malformed data
  - Support for custom date formats
  - Automatic data cleaning and normalization

- **Technical Analysis**
  - EMA (Exponential Moving Average) calculation
  - RSI (Relative Strength Index) analysis
  - MACD (Moving Average Convergence Divergence) indicators
  - Custom trend identification algorithms

- **Trading Strategies**
  - Two distinct backtesting scenarios:
    1. Unlimited transactions with fees
    2. Limited transactions without fees
  - Dynamic position sizing
  - Customizable trading parameters
  - Transaction cost analysis

- **Performance Analysis**
  - Comprehensive performance metrics
  - Risk-adjusted returns calculation
  - Visual performance comparison
  - Detailed trading logs

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/spx-analysis-tool.git
cd spx-analysis-tool
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Basic Usage

```python
from spx_analyzer import SPXAnalyzer

# Initialize analyzer with your data file
analyzer = SPXAnalyzer("path/to/your/spx_data.csv")

# Generate comprehensive analysis report
report = analyzer.generate_report()
print(report)
```

### Advanced Usage

```python
# Customize analysis parameters
analyzer = SPXAnalyzer("spx_data.csv")

# Scenario 1: With transaction fees
returns_s1 = analyzer.calculate_max_return_scenario1(
    transaction_fee=0.02,
    trend_change_threshold=20,
    max_hold_days=252
)

# Scenario 2: With transaction limits
returns_s2 = analyzer.calculate_max_return_scenario2(
    max_transactions=10,
    trend_change_threshold=50,
    lookback_period=252
)
```

## 📁 Project Structure

```
spx-analysis-tool/
├── src/
│   ├── stats.py                 # Statistical calculations
│   ├── spx_analyzer.py          # Main analysis logic
│   └── future_portfolio.py      # Portfolio optimization
├── tests/
│   ├── test_stats.py
│   └── test_spx_analyzer.py
├── data/
│   └── sample_spx_data.csv
├── notebooks/
│   └── analysis_examples.ipynb
├── requirements.txt
├── setup.py
└── README.md
```

## 🔧 Technical Implementation

### Core Components

1. **DataCollector**
   - Handles data ingestion and preprocessing
   - Implements robust error handling
   - Supports multiple data sources

2. **FeatureEngineer**
   - Calculates technical indicators
   - Implements feature normalization
   - Manages feature selection

3. **ModelTrainer**
   - Implements multiple model architectures
   - Handles model training and validation
   - Provides performance metrics

4. **PortfolioManager**
   - Manages position sizing
   - Implements transaction logic
   - Tracks portfolio performance

### Key Algorithms

```python
def calculate_ema(data: List[float], period: int) -> List[float]:
    """
    Calculate Exponential Moving Average
    """
    multiplier = 2 / (period + 1)
    ema = [data[0]]
    for price in data[1:]:
        ema.append((price - ema[-1]) * multiplier + ema[-1])
    return ema
```

## ⚙️ Configuration

The tool supports various configuration options through a YAML file:

```yaml
analysis:
  ema_short: 50
  ema_long: 200
  rsi_period: 14
  macd_fast: 12
  macd_slow: 26
  macd_signal: 9

backtest:
  transaction_fee: 0.02
  max_transactions: 10
  trend_threshold: 20
```

## 🧪 Testing

Run the test suite:

```bash
python -m pytest tests/
```

For coverage report:

```bash
python -m pytest --cov=src tests/
```

## 📊 Performance Analysis

The tool provides comprehensive performance metrics:

- Total Return
- Risk-Adjusted Return
- Sharpe Ratio
- Maximum Drawdown
- Transaction Costs
- Portfolio Turnover

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

