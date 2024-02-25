import pydet as pd

filt = pd.Filters("test/test.png")

filt.saturate(5, True)