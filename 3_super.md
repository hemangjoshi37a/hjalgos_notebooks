# Algorithmic Trading Script using Supertrend Indicator

This script is designed to place orders for financial instruments using the Kite Connect API and the supertrend technical indicator as input. It retrieves historical data for a particular instrument and calculates the supertrend values for three different periods. If all three supertrends cross the values of the current price of the instrument, the script will place an order to buy or sell the instrument, depending on the direction of the crossover.

## Requirements

- Python 3.x
- pandas
- pandas_ta
- kiteconnect

## Usage

1. Make sure you have installed the required dependencies.
2. Fill in the required values in the "NEW VERSION" section of the script. This includes the `fullquant`, `symbol_ip`, `inst_token`, `order_type`, `exchange_type`, `offset_quantity`, and `away_from_circuit` variables.
3. Set the `supertrend_period` and `supertrend_multiplier` variables for each of the three periods.
4. Run the script using `python script.py`.

## Notes

- The `auto_login()` function in the `login` module is currently commented out. To use the script, you will need to implement this function or provide an alternative way to authenticate with the Kite Connect API.
- The script has been tested with the `GBPINR20DECFUT` symbol, but it should work with other symbols as well. You will need to set the `symbol_ip` and `inst_token` variables accordingly.
- The `offset_quantity` and `away_from_circuit` variables are used to offset the price at which the orders are placed. You may need to adjust these values depending on your trading strategy.

