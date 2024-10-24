import csv
from datetime import datetime
from typing import List, Tuple
import math
from stats import calculate_ema, calculate_rsi, calculate_macd
import numpy as np

class SPXAnalyzer:
    def __init__(self, csv_file: str):
        self.data = self._load_data(csv_file)

    def _load_data(self, csv_file: str) -> List[Tuple[datetime, float]]:
        """Loads S&P 500 index prices from a CSV file into memory.

        Each row in the CSV file is expected to contain a date and a price, in
        the format '%d-%m-%Y %H:%M' and a float respectively.

        The function returns a list of tuples, where each tuple contains a
        datetime object and a float representing the price of the S&P 500 index
        at that time.
        """
        
        data = []
        with open(csv_file, 'r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)  # Skip header
            for row in csv_reader:
                date = datetime.strptime(row[0], '%d-%m-%Y %H:%M')
                price = float(row[1])
                data.append((date, price))
        return data

    def _calculate_returns(self) -> List[float]:
        returns = []
        for i in range(1, len(self.data)):
            prev_price = self.data[i-1][1]
            curr_price = self.data[i][1]
            returns.append((curr_price - prev_price) / prev_price)
        return returns

    def _identify_trends(self, ema_short: int = 50, ema_long: int = 200, rsi_period: int = 14, rsi_threshold: int = 50, 
                         macd_fast: int = 12, macd_slow: int = 26, macd_signal: int = 9) -> List[bool]:
        """Identify trends in the S&P 500 index using multiple indicators.

        Uses a combination of EMA, RSI and MACD indicators to identify trends in
        the S&P 500 index. The function returns a list of boolean values,
        where each boolean value represents whether a trend was identified at
        that time point or not.

        Args:
            ema_short: The short period for the exponential moving average.
            ema_long: The long period for the exponential moving average.
            rsi_period: The period for the relative strength index.
            rsi_threshold: The threshold value for the relative strength index.
            macd_fast: The fast period for the moving average convergence divergence.
            macd_slow: The slow period for the moving average convergence divergence.
            macd_signal: The signal period for the moving average convergence divergence.

        Returns:
            A list of boolean values, where each boolean value represents
            whether a trend was identified at that time point or not.
        """
        prices = [price for _, price in self.data]
        
        # Calculate indicators
        ema_short_values = calculate_ema(prices, ema_short)
        ema_long_values = calculate_ema(prices, ema_long)
        rsi = calculate_rsi(prices, rsi_period)
        macd_line, signal_line = calculate_macd(prices, macd_fast, macd_slow, macd_signal)
        
        trends = []
        for i in range(len(prices)):
            if i < max(ema_long, rsi_period, macd_slow, macd_signal):
                trends.append(False)  # Assume no trend for initial points where we don't have all indicator data
            else:
                # EMA trend (golden cross / death cross)
                ema_trend = ema_short_values[i] > ema_long_values[i]
                
                # RSI trend
                rsi_trend = rsi[i] > rsi_threshold
                
                # MACD trend
                macd_trend = macd_line[i] > signal_line[i]
                
                # Combine indicators (you can adjust the weighting or logic here)
                combined_trend = (ema_trend and rsi_trend) or (ema_trend and macd_trend) or (rsi_trend and macd_trend)
                
                trends.append(combined_trend)
        
        return trends

    def calculate_max_return_scenario1(self, transaction_fee: float = 0.02, trend_change_threshold: int = 20, 
                                       max_hold_days: int = 252) -> float:
            """
            Calculate the maximum return that could have been achieved by following a simple
            buy-and-hold strategy based on the identified trends.

            Parameters:
                transaction_fee: The fee to pay for each transaction (default=0.02).
                trend_change_threshold: The number of consecutive points to use for identifying
                    a strong buy or sell signal (default=20).
                max_hold_days: The maximum number of days to hold the position (default=252).

            Returns:
                The total return percentage that could have been achieved by following this strategy.

            Notes:
                This is a simple example and does not take into account various aspects of
                quantitative trading, such as position sizing, risk management, or more complex
                trading strategies.
            """
            trends = self._identify_trends()
            initial_value = self.data[0][1]  # Start with the initial price of the index
            current_value = initial_value
            in_market = False
            last_trade_index = -1
            hold_days = 0

            print(f"Debug - Starting value: {initial_value}")

            for i in range(len(trends)):
                if not in_market:
                    # Check for a strong buy signal
                    if all(trends[max(0, i-trend_change_threshold+1):i+1]):
                        current_value *= (1 - transaction_fee)  # Apply transaction fee
                        in_market = True
                        last_trade_index = i
                        hold_days = 0
                        print(f"Debug - Day {i}: Bought at {self.data[i][1]}. Current value: {current_value}")
                else:
                    hold_days += 1
                    # Check for a strong sell signal or if max hold period is reached
                    if all(not trend for trend in trends[max(0, i-trend_change_threshold+1):i+1]) or hold_days >= max_hold_days:
                        current_value *= (self.data[i][1] / self.data[last_trade_index][1])  # Apply price change
                        current_value *= (1 - transaction_fee)  # Apply transaction fee
                        in_market = False
                        hold_days = 0
                        print(f"Debug - Day {i}: Sold at {self.data[i][1]}. Current value: {current_value}")

                if i % 1000 == 0:  # Periodic logging
                    print(f"Debug - Day {i}: Current value: {current_value}")

            # Ensure we sell at the end if still in the market
            if in_market:
                current_value *= (self.data[-1][1] / self.data[last_trade_index][1])  # Apply final price change
                current_value *= (1 - transaction_fee)  # Apply final transaction fee

            total_return = (current_value / initial_value - 1) * 100  # Calculate total return percentage
            print(f"Debug - Final value: {current_value}")
            print(f"Debug - Total return percentage: {total_return:.2f}%")

            return total_return
    
    def _validate_data(self):
        """Validate the input data for any anomalies."""
        for i in range(1, len(self.data)):
            daily_return = self.data[i][1] / self.data[i-1][1]
            if daily_return > 1.5 or daily_return < 1/1.5:
                print(f"Warning: Unusual price movement detected on day {i}")
                print(f"  Date: {self.data[i][0]}")
                print(f"  Previous price: {self.data[i-1][1]}")
                print(f"  Current price: {self.data[i][1]}")
                print(f"  Daily return: {daily_return}")

    def calculate_max_return_scenario2(self, max_transactions: int = 10, trend_change_threshold: int = 50, 
                                   lookback_period: int = 252) -> float:
        """
        Calculate the maximum possible return for a given set of parameters using the
        "Trend Following" strategy.

        Parameters:
        max_transactions (int): The maximum number of transactions allowed.
        trend_change_threshold (int): The number of days over which a trend change is
            considered significant.
        lookback_period (int): The number of days to look back for trend calculations.

        Returns:
        float: The maximum possible return percentage.
        """
        trends = self._identify_trends(ema_short=50, ema_long=200, rsi_period=14, macd_fast=12, macd_slow=26, macd_signal=9)
        initial_value = self.data[0][1]  # Start with the initial price of the index
        current_value = initial_value
        in_market = False
        last_trade_index = -1
        transactions = 0

        def calculate_strength(start_index, end_index):
            if start_index < 0:
                start_index = 0
            price_change = self.data[end_index][1] / self.data[start_index][1] - 1
            trend_strength = sum(trends[start_index:end_index+1]) / (end_index - start_index + 1)
            return price_change * trend_strength

        print(f"Debug - Starting value: {initial_value}")

        for i in range(lookback_period, len(trends)):
            if transactions >= max_transactions:
                break

            strength = calculate_strength(i-lookback_period, i)

            if not in_market:
                if strength > 0.05 and all(trends[i-trend_change_threshold+1:i+1]):
                    in_market = True
                    last_trade_index = i
                    transactions += 1
                    print(f"Debug - Day {i}: Bought at {self.data[i][1]}")
            else:
                # Exit if trend reverses or after holding for 2 years with minimal gain
                if (strength < -0.05 and all(not trend for trend in trends[i-trend_change_threshold+1:i+1])) or \
                (i - last_trade_index > 504 and (self.data[i][1] / self.data[last_trade_index][1] - 1) < 0.1):
                    current_value *= (self.data[i][1] / self.data[last_trade_index][1])
                    in_market = False
                    transactions += 1
                    print(f"Debug - Day {i}: Sold at {self.data[i][1]}. Current value: {current_value}")

            if i % 1000 == 0:  # Periodic logging
                print(f"Debug - Day {i}: Current value: {current_value}")

        # Ensure we sell at the end if still in the market
        if in_market:
            current_value *= (self.data[-1][1] / self.data[last_trade_index][1])
            transactions += 1

        total_return = (current_value / initial_value - 1) * 100  # Calculate total return percentage
        print(f"Debug - Final value: {current_value}")
        print(f"Debug - Total return percentage: {total_return:.2f}%")
        print(f"Debug - Total transactions: {transactions}")

        return total_return


    def generate_report(self):
        """
        Generate a report comparing the performance of two different trading strategies
        to the overall market performance.

        The report includes the total return, annualized return, and outperformance
        of each strategy, as well as a visual comparison of the returns using an ASCII chart.

        The strategies are:

        1. Scenario 1 (2% transaction fee, no turnover limit): Buy/Sell based on
        EMA, RSI, MACD indicators.
        2. Scenario 2 (0% transaction fee, max 20 transactions): Buy/Sell based on
        EMA, RSI, MACD indicators, with a limit on the number of transactions.

        The report also includes an interpretation of the results, highlighting which
        strategy performed better and by how much.

        :return: A string containing the report
        """
        self._validate_data()  
        scenario1_return = self.calculate_max_return_scenario1()
        scenario2_return = self.calculate_max_return_scenario2()

        total_days = len(self.data)
        total_return = (self.data[-1][1] / self.data[0][1] - 1) * 100
        annualized_return = ((1 + total_return / 100) ** (365 / total_days) - 1) * 100

        report = f"""
                {'='*60}
                                SPX ANALYSIS REPORT
                {'='*60}

                Period: {self.data[0][0].strftime('%Y-%m-%d')} to {self.data[-1][0].strftime('%Y-%m-%d')}
                Total Days: {total_days}

                MARKET PERFORMANCE
                ------------------
                Total Return: {total_return:.2f}%
                Annualized Return: {annualized_return:.2f}%

                STRATEGY PERFORMANCE
                --------------------
                Scenario 1 (2% transaction fee, no turnover limit):
                    Max Return: {scenario1_return:.2f}%
                    Outperformance: {scenario1_return - total_return:.2f}%

                Scenario 2 (0% transaction fee, max 20 transactions):
                    Max Return: {scenario2_return:.2f}%
                    Outperformance: {scenario2_return - total_return:.2f}%

                VISUAL COMPARISON
                -----------------
                {self._generate_ascii_chart(total_return, scenario1_return, scenario2_return)}

                INTERPRETATION
                --------------
                - The strategy with {('no transaction limit' if scenario1_return > scenario2_return else 'limited transactions')} performed better.

                Note: The strategies use a combination of technical indicators and do not account for fundamental analysis or market conditions.
                """
        return report

    def _generate_ascii_chart(self, market_return, scenario1_return, scenario2_return):
        try:
            returns = [market_return, scenario1_return, scenario2_return]
            labels = ['Market', 'Scenario 1', 'Scenario 2']
            
            print("Debug - Returns before processing:")
            for label, value in zip(labels, returns):
                print(f"{label}: {value}")

            valid_returns = [r for r in returns if not math.isnan(r) and not math.isinf(r)]
            
            if not valid_returns:
                return "Unable to generate chart due to invalid return values."

            max_return = max(valid_returns)
            min_return = min(valid_returns)
            
            if max_return == min_return:
                scale = 1
            else:
                scale = 20 / (max_return - min_return)

            def get_bar(value):
                if math.isnan(value) or math.isinf(value):
                    return "N/A"
                bar_length = max(0, int((value - min_return) * scale))
                return '█' * bar_length

            chart = ""
            for label, value in zip(labels, returns):
                bar = get_bar(value)
                chart += f"{label:11}: {bar} {value:.2f}%\n"

            return chart
        except Exception as e:
            return f"An error occurred while generating the chart: {str(e)}"

def main(csv_file: str):
    analyzer = SPXAnalyzer(csv_file)
    report = analyzer.generate_report()
    print(report)

if __name__ == "__main__":
    main("spx_data.csv")  # Replace with your actual CSV file name