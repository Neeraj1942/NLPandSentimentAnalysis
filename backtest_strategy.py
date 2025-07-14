# backtest_strategy.py
import backtrader as bt
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import sys

# Define the Sentiment indicator
class Sentiment(bt.Indicator):
    lines = ('sentiment',)
    plotinfo = dict(
        plotymargin=0.5,
        plothlines=[0],
        plotyticks=[1.0, 0, -1.0])

    def next(self):
        self.sentiment = 0.0
        self.date = self.data.datetime
        date = bt.num2date(self.date[0]).date()
        prev_sentiment = self.sentiment        
        if date in date_sentiment:
            self.sentiment = date_sentiment[date]
        self.lines.sentiment[0] = self.sentiment


# Define the backtest strategy class
class SentimentStrat(bt.Strategy):
    params = (
        ('period', 15),
        ('printlog', True),
    )

    def log(self, txt, dt=None, doprint=False):
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.sma = bt.indicators.SimpleMovingAverage(self.datas[0], period=self.params.period)
        self.date = self.data.datetime
        self.sentiment = None
        Sentiment(self.data)
        self.plotinfo.plot = False

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log('BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                         (order.executed.price, order.executed.value, order.executed.comm))
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                         (order.executed.price, order.executed.value, order.executed.comm))

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
                 (trade.pnl, trade.pnlcomm))

    def next(self):
        date = bt.num2date(self.date[0]).date()
        prev_sentiment = self.sentiment
        if date in date_sentiment:
            self.sentiment = date_sentiment[date]

        if self.order:
            return
        if not self.position and prev_sentiment:
            if self.dataclose[0] > self.sma[0] and self.sentiment - prev_sentiment >= 0.5:
                self.log('Previous Sentiment %.2f, New Sentiment %.2f BUY CREATE, %.2f' % (prev_sentiment, self.sentiment, self.dataclose[0]))
                self.order = self.buy()

        elif prev_sentiment:
            if self.dataclose[0] < self.sma[0] and self.sentiment - prev_sentiment <= -0.5:
                self.log('Previous Sentiment %.2f, New Sentiment %.2f SELL CREATE, %.2f' % (prev_sentiment, self.sentiment, self.dataclose[0]))
                self.order = self.sell()

    def stop(self):
        self.log('(MA Period %2d) Ending Value %.2f' %
                 (self.params.period, self.broker.getvalue()), doprint=True)


# Run the backtest strategy
def run_strategy(ticker, start, end, date_sentiment):
    print(f"Running strategy for {ticker}")
    ticker_data = yf.Ticker(ticker)
    df_ticker = ticker_data.history(start=start, end=end)

    cerebro = bt.Cerebro()
    cerebro.addstrategy(SentimentStrat)

    # Add the data
    data = bt.feeds.PandasData(dataname=df_ticker)
    cerebro.adddata(data)

    start_value = 100000.0
    cerebro.broker.setcash(start_value)
    cerebro.addsizer(bt.sizers.FixedSize, stake=100)

    print('Starting Portfolio Value: %.2f' % start_value)
    cerebro.run()

    # Plot the results with matplotlib (disable interactive mode)
    #fig = cerebro.plot(volume=False, iplot=False, plotname=ticker)[0]
    #fig.set_size_inches(10, 6)  # Adjust figure size manually
    #plt.show()  # Ensure the plot is displayed with Matplotlib

    end_value = cerebro.broker.getvalue()
    print(f'Start Portfolio value: %.2f\nFinal Portfolio Value: %.2f\nProfit: %.2f\n' %
          (start_value, end_value, end_value - start_value))

    return df_ticker['Close'][0], (end_value - start_value)





if __name__ == "__main__":
    ticker = sys.argv[1]  # pass ticker as argument
    start_date = sys.argv[2]  # pass start date
    end_date = sys.argv[3]  # pass end date
    date_sentiment_file = sys.argv[4]  # pass sentiment data file

    # Load sentiment data from a CSV file
    date_sentiment_df = pd.read_csv(date_sentiment_file)
    date_sentiment_df['date'] = pd.to_datetime(date_sentiment_df['date']).dt.date
    date_sentiment = date_sentiment_df.set_index('date')['sentiment_lex'].to_dict()

    run_strategy(ticker, start_date, end_date, date_sentiment)
