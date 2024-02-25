import pydet as pd

filt = pd.Filters("test_images/test.png")

filt.saturate(5, True)