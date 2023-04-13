import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

mtcars = pd.read_csv('mtcars.csv')

plt.figure()
mtcars2 = mtcars.groupby('cyl').mpg.mean()
mtcars2.plot.bar()
plt.show()
