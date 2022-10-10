##### Carnegie Mellon University - Tepper School of Business
##### 46-881 - Programming in R and Python - 2022 - Mini 1 - Farahat
##### Class 12 version
---

### DELIVERABLE 4 - SOLUTIONS


### Problem 1 (50 points)

The SIR model is the original and simplest (in fact, *simplistic*) model of disease spread in epidemiology. It is a *compartmental model* where each individual in a population is classified as being in exactly one of 3 states at any gien time: **S**usceptible to infection, **I**nfectious (i.e. infected and capable of infecting susceptible individuals), or **R**ecovered (or "removed"). The model was developed by Kermack and McKendrick in 1927. Many improved predictive models have since been developed for disease spread. In this problem you will use Python to implement and explore this model.

Let S(t), I(t), and R(t) denote, respectively, the number of individuals in a given population who are susceptible, infectious, and recovered on day t. The model utilizes two parameters: Beta and Gamma. For a population of size N, the time-dynamics of the model are as follows:
- S(t+1) = S(t) - Beta * I(t) * S(t) / N
- I(t+1) = I(t) + Beta * I(t) * S(t) / N - Gamma * I(t)
- R(t+1) = R(t) + Gamma * I(t) 

The ratio Beta / Gamma is sometimes referred to as the R0 (R naught) of the disease.

Develop a Python model of the above SIR model and use it to answer the following questions. Assume a scenario where N = 1000, S(0) = 990, I(0) = 10, Beta = 0.10, and Gamma = 0.05.


```python
# Model parameters
N = 1000 
beta = 0.10
gamma = 0.05
S = [990]
I = [10]
R = [0]
max_t = 365
```


```python
# Note the indices correctly correspond to time
for t in range(max_t-1):
    S_to_I = beta * I[t] * S[t] / N
    I_to_R = gamma * I[t]
    S.append(S[t] - S_to_I)
    I.append(I[t] + S_to_I - I_to_R)
    R.append(R[t] + I_to_R) 
```

#### Part(a)
Suppose our healthcare system has a capacity to accommodate a maximum of 120 infectious persons. Is this capacity sufficient according to the model?


```python
max_I = max(I)
max_I
```




    160.37770863476757



The answer is NO. The maximum number of infectious persons peaks at 160, exceeding system capacity. 

#### Part (b)
Now suppose that our healthcare system can currently accommodate only 120 infectious persons. However, its capacity can be increased to accommodate 180 infectious persons in 2 months. Is this current capacity and expansion plan sufficient according to the model?


```python
max_I_2months = max(I[0:61]) 
max_I_2months
```




    111.74956698000823



The answer is YES, it is sufficient. The current capacity is sufficient over the first two months. The proposed expanded capacity is sufficient for the peak afterwards.

### Problem 2 (100 points)

*The purpose of this problem is to help you learn/review Python, while retaining your knowledge of R, through a direct comparison of both languages. In particular, you are asked to answer the very same exploratory data analysis questions you have answered in Problems 1 and 2 of Deliverable 3, but using Python.*

Answer the following question based on the 2021 Healthy Ride rentals dataset(s) discussed in class September 12th - September 14th. The datasets, <rentals_all.csv> and <final_dataset.csv> are provided again accompanying this deliverable on the assignment's Canvas page.


```python
import numpy as np
import pandas as pd
```


```python
df_hourly = pd.read_csv("final_dataset.csv")
df_rentals = pd.read_csv("rentals_all.csv")
```

#### Part (a)
Which month has the most total precipitation? 


```python
df_hourly["precipitation"].groupby(df_hourly["month"]).sum()
```




    month
    1     1.82
    2     2.95
    3     3.21
    4     2.34
    5     3.31
    6     7.45
    7     4.52
    8     6.20
    9     4.42
    10    4.50
    11    0.80
    12    4.23
    Name: precipitation, dtype: float64



The month of June (month 6) has the most precipitation.

#### Part (b)
How many days of the year have precipitation at some point during that day (do not count trace “T” amounts)?


```python
daily_pre = df_hourly["precipitation"].groupby([df_hourly["month"],df_hourly["day"]]).sum()
sum(daily_pre > 0)
```




    143




```python
sum(daily_pre > 0)
```




    143



#### Part (c)
What is the average daily high temperature (maximum hourly temperature during the day) in September? 


```python
df_hourly_sept = df_hourly[df_hourly["month"] == 9]
sept_grouped_daily = df_hourly_sept["temperature"].groupby(df_hourly_sept["day"])
sept_daily_high = sept_grouped_daily.agg(lambda x: x.max(skipna=False))
sept_daily_high.mean()
```




    74.96428571428571



#### Part (d)
What is the coefficient of correlation between hourly rentals and hourly windspeed?


```python
df_hourly[["rentals","windspeed"]].corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rentals</th>
      <th>windspeed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>rentals</th>
      <td>1.000000</td>
      <td>0.005181</td>
    </tr>
    <tr>
      <th>windspeed</th>
      <td>0.005181</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



The correlation coefficient is 0.0052 (pretty small)

#### Part (e)
Which day of the week (Monday - Sunday) has, on average, the largest number of rentals? What is that average number of rentals?


```python
df_hourly["rentals"].groupby(df_hourly["day_of_week"]).mean()
```




    day_of_week
    Fri    13.347484
    Mon    12.432692
    Sat    19.044872
    Sun    15.893344
    Thu    12.092147
    Tue    13.384615
    Wed    12.599359
    Name: rentals, dtype: float64



Saturday has on average the largest number of rentals with a rate of 19 rentals per hour (456 per day)

#### Part (f)
What percentage of rentals were carried out by subscribers?  


```python
df_rentals["Usertype"].value_counts()
```




    Customer      68498
    Subscriber    55095
    Name: Usertype, dtype: int64



Subscriber rentals account for 55095 out of 123593 rentals, or 44.6%

#### Part (g)
Which hour of the day has, on average, the largest number of rides? Which hour has the second largest number of rentals? 


```python
df_hourly["rentals"].groupby(df_hourly["hour"]).mean()
```




    hour
    0      7.734247
    1      5.054795
    2      3.961538
    3      2.701370
    4      1.202740
    5      1.465753
    6      2.161644
    7      3.643836
    8      6.967123
    9     10.443836
    10    13.210959
    11    15.728767
    12    19.652055
    13    22.550685
    14    23.487671
    15    26.208219
    16    27.808219
    17    28.339726
    18    28.827397
    19    25.920548
    20    21.347945
    21    17.084932
    22    12.956164
    23    10.180822
    Name: rentals, dtype: float64



The 6 p.m. - 7 p.m. hour followed by the 5 p.m. - 6 p.m. hour.

#### Part (h)
What percentage of rentals has a dockless origin? What percentage of rentals has a dockless destination? What percentage of rentals has either a dockless origin or a dockless destination?


```python
n = df_rentals.shape[0] # getting the number of rentals (equals number of rows on df_rentals dataframe)
```


```python
dockless_origin = df_rentals["From station name"].str.startswith("BIKE")
dockless_origin.sum() / n * 100
```




    5.615695792880259




```python
dockless_destination = df_rentals["To station name"].str.startswith("BIKE")
dockless_destination.sum() / n * 100
```




    7.610032362459547




```python
dockless_either = dockless_origin | dockless_destination
dockless_either.sum() / n * 100
```




    11.6626213592233



#### Part (i)
What bike station (provide street address) has the most volume of outbound rentals. What bike station (provide street address) has the most volume of inbound rentals.


```python
df_rentals["From station name"].value_counts()
```




    Liberty Ave & Stanwix St                    4495
    North Shore Trail & Fort Duquesne Bridge    3834
    S Bouquet Ave & Sennott St                  3255
    S 27th St & Sidney St. (Southside Works)    3038
    Centre Ave & N Craig St                     2924
                                                ... 
    BIKE 70482                                     1
    70234                                          1
    BIKE 70934                                     1
    BIKE 70151 - doubletree                        1
    BIKE 70317                                     1
    Name: From station name, Length: 767, dtype: int64



Liberty Ave & Stanwix St has the most volume of outbound rentals.


```python
df_rentals["To station name"].value_counts()
```




    Liberty Ave & Stanwix St                     4796
    North Shore Trail & Fort Duquesne Bridge     3963
    S Bouquet Ave & Sennott St                   3307
    S 27th St & Sidney St. (Southside Works)     3267
    Schenley Dr & Forbes Ave (Schenley Plaza)    2997
                                                 ... 
    Fort Pitt Museum                                1
    recording_120320969                             1
    recording_113911594                             1
    BIKE 70692                                      1
    Frankstown Ave & Eastview St                    1
    Name: To station name, Length: 832, dtype: int64



Liberty Ave & Stanwix St also has the most volume of inbound rentals.

#### Part (j)
Which pair of stations has the most number of rentals between them (add the rentals in each direction).

We can follow the same approach described for Problem 1(f) of Deliverable 3 (you can review the solution beforehand). The idea is to arrange each pair of stations alphabetically then group by distinct pairs.


```python
# To guarantee that alphabetical ordering works properly, 
# we'll first ensure that From and To stations names are string arrays.
df = df_rentals[["From station name", "To station name"]]

df["alpha_first"] = df.apply(lambda x: min(str(x["From station name"]), str(x["To station name"])),
                             axis=1)

df["alpha_second"] = df.apply(lambda x: max(str(x["From station name"]), str(x["To station name"])),
                             axis=1)

df = df.drop(["From station name", "To station name"], axis = 1)

#df_distinct = df[df["alpha_first"] != df["alpha_second"]]

df.groupby([df["alpha_first"],df["alpha_second"]]).value_counts().sort_values(ascending=False)
```

    C:\Users\afarahat\AppData\Local\Temp\ipykernel_10592\3233856748.py:5: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df["alpha_first"] = df.apply(lambda x: min(str(x["From station name"]), str(x["To station name"])),
    C:\Users\afarahat\AppData\Local\Temp\ipykernel_10592\3233856748.py:8: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df["alpha_second"] = df.apply(lambda x: max(str(x["From station name"]), str(x["To station name"])),
    




    alpha_first                               alpha_second                                         
    Liberty Ave & Stanwix St                  Liberty Ave & Stanwix St                                 2291
    North Shore Trail & Fort Duquesne Bridge  North Shore Trail & Fort Duquesne Bridge                 2189
    33rd St & Penn Ave                        33rd St & Penn Ave                                       1711
    S 27th St & Sidney St. (Southside Works)  S 27th St & Sidney St. (Southside Works)                 1337
    Microsoft                                 Microsoft                                                1001
                                                                                                       ... 
    BIKE 70642                                Frew St & Schenley Dr                                       1
                                              S Bouquet Ave & Sennott St                                  1
                                              Schenley Dr at Schenley Plaza (Carnegie Library Main)       1
                                              Wightman St & Forbes Ave                                    1
    recording_119278362                       recording_119278362                                         1
    Length: 13640, dtype: int64



On average then, the number of rides originating and ending at the Liberty Ave & Stanwix St station (2291 rides) exceed the number of rides between any other pair of stations.

### Problem 3 (50 points)

Suppose you drive a car whose fuel efficiency is 18 miles per gallon and whose gas tank size is 16 gallons. Gas prices fluctuate according to a **uniform distribution** ranging from 3 dollars per gallon to 5 dollars per gallon. That is, prices remain average equal to 4 dollars per gallon but fluctuate each day according to this uniform distribution. Every day you drive your car for exactly 10 miles.

You are contemplating two gas filling strategies:
- *“Fill-it up” strategy*: You wait till the tank is (almost) empty then fill the entire tank;
- *“30-dollar” strategy*: You wait till the tank is (almost) empty then fill in only 30 dollars worth of gas.

Assume that there are many gas stations around, so you can time your visits almost exactly when the tank is empty. Assume also that any inconvenience from visiting the gas station is negligible. Your only objective is to minimize your gas expenditure.

***Construct a simulation model in Python to determine if one strategy is more economical than the other (and, if so, by how much).*** 



```python
import random
```


```python
# Common parameters:
efficiency = 18 # miles per gallon
daily_distance = 10 # miles
price_low = 3 # dollars per gallon
price_high = 5 # dollars per gallon

# Fill-it up strategy parameters:
fill_gallons = 16

# Fill-it up strategy parameters:
fill_dollars = 30
```

The two simulation approaches below are equivalent and yield the same conclusion: **the 30 dollar strategy is more economical, but not by much (about 8 cents)**. 

### Simulation Approach 1: Fixed number of fills

#### Fill-it up strategy


```python
num_refills = 100_000 # number of trials in simulation
dollars_paid_list = []
random.seed(881)

for i in range(num_refills):
    price = random.uniform(price_low, price_high)
    dollars_paid_list.append(fill_gallons*price)
    
av_price_per_gallon = sum(dollars_paid_list) / (num_refills * fill_gallons)  

av_price_per_gallon
```




    4.001126688157896



#### 30-dollar strategy


```python
num_refills = 100_000 # number of trials in simulation
gallons_filled_list = []
random.seed(881)

for i in range(num_refills):
    price = random.uniform(price_low, price_high)
    gallons_filled_list.append(fill_dollars / price)
    
av_price_per_gallon = (num_refills * fill_dollars) / sum(gallons_filled_list) 

av_price_per_gallon
```




    3.916118376400644



### Simulation Approach 2: Fixed elapsed time

#### Fill-it up strategy


```python
elapsed_time = 10_000_000 # number of *days* elapsed
t = 0
total_gallons_purchased = 0
total_dollars_spent = 0

random.seed(881)

while t < elapsed_time:
    price = random.uniform(price_low, price_high)
    gallons_purchased = fill_gallons
    dollars_spent = fill_gallons * price
    total_gallons_purchased += gallons_purchased
    total_dollars_spent += dollars_spent
    t += gallons_purchased * efficiency / daily_distance
    
av_price_per_gallon = total_dollars_spent / total_gallons_purchased  
av_price_per_gallon
```




    3.9999830051639154



#### 30-dollar strategy


```python
elapsed_time = 10_000_000 # number of *days* elapsed
t = 0
total_gallons_purchased = 0
total_dollars_spent = 0

random.seed(881)

while t < elapsed_time:
    price = random.uniform(price_low, price_high)
    gallons_purchased = fill_dollars / price
    dollars_spent = fill_dollars
    total_gallons_purchased += gallons_purchased
    total_dollars_spent += dollars_spent
    t += gallons_purchased * efficiency / daily_distance
    
av_price_per_gallon = total_dollars_spent / total_gallons_purchased  
av_price_per_gallon
```




    3.9152087996901366


