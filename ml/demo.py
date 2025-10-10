import pandas as pd;
import matplotlib.pyplot as mp
df=pd.read_csv('StressLevelDataset.csv')
print(df.head)
print(df.info)
print(df.tail)
#p.plot(df['sleep_quality'],df['anxiety_level'])
#mp.bar(df['mental_health_history'],df['depression'])
mp.hist(df['mental_health_history'])
mp.show()
