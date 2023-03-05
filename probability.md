# `Probability`

## `Events and Sample spaces`
$$
P(event) = {events\over \Omega}
$$
- event = outcomes of event
- $\Omega$ = outcomes of $\Omega$

## `Multiple independent observations`
- It's flipping a coin 2-3 times instead of 1  
- Combining probability: The probability of throwing 5 heads in a row = probability of throwing 2 heads muliply by 3 heads in a row
  $$P(HHHHH)=P(HH)*P(HHH)={1\over 4}*{1\over 8}={1\over 32}$$

## `Combinatorics`
- It's a field in mathematics devoted to counting
- Can use factorials to calculate the probability 
$$
\binom nk = \frac{n!}{k!(n-k)!}
$$
- If we're interested in choosing 2 heads or tails in 3 flips then n=3 and k=2
$$
\binom 32 = \frac{3!}{2!(3-2)!}=3
$$
- This means that there are 3 outcomes where there are exactly 2 tails or 2 heads
- So the probability is
$$P = \frac{3}{2^n}=0.375$$
- Since there are only 2 outcomes per coin flip
$$\binom 22 = \frac{2!}{2!(2-2)!} = 1$$
$$P = \frac{1}{52^2}=0.00036982248$$
$$P(any) = \frac{52}{52}$$
$$P{same card} = \frac{1}{52}$$
$$P=P(any)P(samecard)=\frac{1}{52}$$
$$\binom 52 = \frac{5!}{2!(5-2)!}=10$$
$$P=\frac{10}{2^5} = 0.3125$$

## `The law of large number`
- The larger the number we run the tests, the closer we get to the expected results

## `Probability ditributions`
  ```python
  import numpy as np
  import scipy.stats as st
  import matplotlib.pyplot as plt

  n_experiments = 1000
  heads_count = np.random.binomial(5, 0.5, n_experiments)

  heads, event_count = np.unique(heads_count, return_counts=True)

  print(heads)
  #[0 1 2 3 4 5]
  print(event_count)
  #[ 39 156 284 315 164  42]
  # There are 39 times there are no heads out of 5 tosses
  # There are 156 times there are 1 head only
  # The closer to the edges, the less likely it happens
  ```
## `Bayesian Statistics vs Frequentist Statistics`
- Bayesian statistics:
  - Draw conclusions from beliefs and experimental results
  - Computationally expensive
- Frequentist statistics:
  - Focus on "objective" probabilities
  - Abitrary threshold 5%
  - Not designed for large feature sets, 5% threshold is too high for large sample size
- Machine learning usually speaks probability. For example: a photo could be 98% a hot dog

## `Discrete and continuous variables`
- `Random variable`: a variable whose value is determined by a process that has uncertainty. For example: a coin toss
- Notation: regular character if it's abstract, *italics* if it's assigned a value
- `Discrete type` is random variables that have countable number of states like a coin toss has only 2 states. It could be finite or infinite
- `Continuous type` is real values represented by float numbers like height, speed, temperature
- A `Probability Mass Function (PMF)` is a function describing the probability distribution of discrete random variables, denoted *P*
- *P(x)* must cover every possible value of x is within the domain
- Each *P(x)* can only range from 0 to 1
- Sum of all *P(x)* can only 1 (normalization)
- A `Probability Density Function (PDF)` is analouge for continuous random variables
## `Expected value`
- The `expected value` is the long-term average for some random variable x
- If x is decrete:
  $$\Epsilon = \sum_xxP(x)$$
- If x is continuous:
  $$\Epsilon = \int xp(x)dx$$
- Example: coin toss
  - P(0)=P(5)=0.031
  - P(1)=P(4)=0.16
  - P(2)=P(3)=0.31
  - $\Epsilon=0{1\over 32}+1{5\over 32}+2{10\over 32}+3{10\over 32}+4{5\over 32}+5{1\over 32}=2.5$
  ```python
  from math import factorial 

  def coinflip_prob(n, k):
      n_choose_k = factorial(n)/(factorial(k)*factorial(n-k))
      return n_choose_k/2**n
      
  P = [coinflip_prob(5, x) for x in range(6)]    

  E = sum(P[x]*x for x in range(6))
  print(E)
  # 2.5
  ```
  ```python
  P = [coinflip_prob(2, x) for x in range(3)]
  # [0.25, 0.5, 0.25]
  # if flip twice, the prob of x=0 is 0.25 (no head), x=1 is 0.5(exact 1 head) and x=2 is 0.25(exact 2 heads)

  E = sum(P[x]*x for x in range(3))
  print(E)
  #1
  ```
## `Measures of central tendency`
- Measure of central tendency provides a summary statistic on the center of a given distribution or `average mean`, denoted with $\mu$ (population) or $\overline{x}$ (sample)
  $$\mu=\Epsilon$$
  ```python
  import numpy as np

  heads_count = np.random.binomial(5, 0.5, 1000)

  mean = sum(heads_count)/len(heads_count)
  # 2.51
## `Median`
 - The mid point value in the distribution
  ```python
  import numpy as np

  heads_count = np.random.binomial(5, 0.5, 1000)
  heads_count.sort()
  np.median(heads_count)
  #3.0
  ```
## `Mode`
- The value that occurs most frequently
  ```python
  import scipy.stats as st

  st.mode(heads_count)
  # ModeResult(mode=array([2]), count=array([334])) 
  ```
## `Quantiles: Percentiles, quartiles and deciles`
- `Median` is the most well-known quantile
  ```python
  np.median(heads_count)
  # 3.0
  np.quantile(heads_count, 0.5)
  # 3.0
  ```
- `Percentiles` divide the distribution at any point out of 100.
  - If we want to see the top 5%, we divide at 95th percentile
  - If we want to see the threshold for the top 1%, we devide at 99th percentile
- `Quartiles` divide the distribution into quarters at the 25th, 50th and 75th percentile
  ```python
  np.percentile(heads_count, [25, 50, 75])
  # [2. 3. 3.]
  ```
- `Deciles` (means tenth) divide the distribution into 10 segments
  ```python
  np.percentile(heads_count, [i for i in range(10, 100, 10)])
  # [1. 1. 2. 2. 2. 3. 3. 3. 4.]
  ```
- Skewed distributions drag the mean away from the center and toward the edge
  ```python
  x = st.skewnorm.rvs(10, size=1000)
  ```
# `The box and whisker plots`
- It's another tool to plot the distributions
- The box edges define the `inter-quatile range (IQR)`
- The whisker ranges are determined by the furthest data points within 1.5 x IQR
  ```python
  import seaborn as sns
  import matplotlib.pyplot as plt
  import numpy as np
  import scipy.stats as st

  sns.set(style='whitegrid')
  x = st.skewnorm.rvs(10, size=1000)
  sns.boxplot(x = x)
  plt.show()

  q = np.percentile(x, [25, 50, 75])
  # [0.33229548 0.66228792 1.17018088]

  IQR= q[2] - q[0]
  # 0.8378854014194564

  lowest_whisker = q[0] - 1.5*IQR
  # -0.9245326211391839
  np.min(x)
  # -0.19540695486545973
  # the min should be within the lowest whisker

  upper_whisker = q[2] + 1.5*IQR
  # 2.427008984538642
  ```
- There are several values beyond the upper whisker, these are the outliers and plotted as individual points

## `Variance`
- It's a measure of dispersion
  $$
  \sigma^2 = \frac{\sum_{i=1}^n(x_i-\overline{x})^2}{n}
  $$
  - $x_i$ is each instance
  - $\overline x$ is mean 
  - $n$ is the number of instances
  ```python
  x = st.skewnorm.rvs(10, size=1000)
  xbar = np.mean(x)
  # 0.804847888858779

  squared_diff = [(x_i - xbar)**2 for x_i in x]

  variance = np.sum(squared_diff) / len(x)
  # 0.3736374776235986
  np.var(x)
  # 0.3736374776235986
  ```

## `Standard deviation`
- Measure of the amount of variation or dispersion.
- The lower the standard deviation, the closer the values tend to be to the mean
  $$\sigma = \sqrt{\sigma^2}$$
  ```python
  sigma = variance**(1/2)
  # 0.5867545501284294
  np.std(x)
  # 0.5867545501284294
  ```
## `Standard error`
- Measure of how different the population mean is likely to be from a sample mean
- This shows us how the sample mean could vary if we repeat the study multiple times within the same population
  $$\sigma_{\overline{x}} = \frac{\sigma}{\sqrt{n}}$$
  ```python
  st.sem(x)
  ```
## `Coveriance` 
- If we have x and y are 2 collections of pair data
  $$
  cov(x, y)=\frac{\sum_i^n(x_i-\overline{x})(y_i-\overline{y})}{n}
  $$
  ```python
  x = iris.sepai_length
  y = iris.petal_length
  np.cov(x, y, ddof=0)
  # [[0.68112222, 1.26582][1.26582, 3.09550267]] 
  # 0.68112222 is variance of x
  # 3.09550267 is variance of y
  # 1.26582 is the coveriance between the 2
  ```
- If the coveriance is positive, meaning there is a positive relationship between the 2 and via versa
- For example if x gets larger, y also gets larger
- The less related, the closer coveriance to 0
  ```python
  np.cov(x=sepai_length, y=petal_width)
  # [[0.68112222 0.51282889][0.51282889 0.57713289]]
  ```
## `Correlation`
- Correlation is preferred when it comes to relatedness between 2 dataset because it doesn't get affected by the change of scale
  $$
  \rho_{x,y} = \frac{cov(x,y)}{\sigma_x\sigma_y}
  $$
  ```python
  st.pearsonr(iris.sepal_length, iris.petal_length)
  # (0.8717537758865833, 1.0386674194496954e-47)
  # 0.87 is the correlation
  ``
- The strongest correlations are 1 and -1, 0 means no correlation

## `Joint and marginal probability distribution`
- Joint probability is when we want to see the probability of 2 variables x and y happen at the same time P(x=x,y=y)
- Marginal probability is when we want to see the total probability of 1 variable
- For example, given a table of probability for discrete variables
  - x=1 if having the disease, x=0 if not
  - y=1 if having the symtoms, y=0 if not
    | y\x | 0   | 1   | 
    | --- | --- | --- |
    | 0   | 0.5 | 0.1 |
    | 1   | 0.1 | 0.3 |
  - The total probability of all scenarios is $\sum P(x=x,y=y)=1$
  - The joint probability of people who have the symtoms and disease is $P(x=1,y=1)=0.3$
  - The marginal probability of people who have symtoms is $P(y=1)=0.1+0.3=0.4$
  - The marginal probability of people who don't have disease is $P(x=0)=0.5+0.1=0.6$
- For continuous variables, we can integrate:
  $$p(x) = \int p(x,y)dy$$
## `Conditional probability`
- Probability of an outcome given that another outcome already occurred
  $$P(y=y|x=x)=\frac{P(y=y,x=x)}{P(x=x)}$$
- Example: If we flip a coin twice and the first flip is already a head, then the probability of second flip to be head is:
  $$P(f2=heads,f1=heads)=0.25$$
  $$P(f2=heads|f1=heads)=0.25/0.5=0.5$$
- So the probability of having heads on the second flip is still 50% because it's independent from the first flip

## `Chain rule of probability`
  $$P(y|x)=\frac{P(y,x)}{P(x)}$$
  $$P(y,x)=P(y|x)P(x)$$
  $$P(z,y,x)=P(z|y,x)P(y|x)P(x)$$

## `Conditional independence`
- x and y are independent given z ($x\perp y|z$)
  $$p(x=x, y=y|z=z)=P(x=x|z=z)P(y=y|z=z)$$
- Probability of wrestler winning gold medal (x) and weightlifter winning gold medal (y) if both of them comes from the country with doping scandal (z)

## `Uniform distribution`
- Uniform distribution has the constant probability across all of its outcome
- For example: every day I take a bus to work and this bus could be late from 2-10 mins, the probability of late for *n* min is equally distributed. If the bus is 7 mins late then I'm late for work, so what is the probability of being late?
  $$P(late)=\frac{10-7}{10-2}=\frac{3}{8}$$
  - The deviation from the average waiting time is
  $$\sigma = \sqrt{\frac{(10+2)^2}{12}}=2.31mins$$

## `Gaussian: Normal and standard normal`
- Gaussian aka normal distribution is the bell-shape distribution
- The normal distribution has mean = 0
- If the standard deviation = 1 then it's a standard normal distribution N(0, 1)
  
# `The central limit theorem`
```python
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def sample_mean(dist, sample_size, n_samples):
    sample_means = []
    for i in range(n_samples):
        sample = np.random.choice(dist, sample_size, replace=False)
        sample_means.append(sample.mean())
    return sample_means

x = np.random.normal(size=10000)
sns.displot(sample_mean(x, 10, 1000), color='green')
plt.xlim(-1.5, 1.5)

plt.show()
```
- The distribution of means is always close to normal distribution
```python
x = st.skewnorm.rvs(10, size=10000)
sns.displot(sample_mean(x, 1000, 1000), color='green')
plt.show()
```
- Even with the skewed distribution, the mean of random choices are still normally distributed

## `Log-normal`
- The distribution of a variable is Log-normal when its logarithm is normally distributed

## `Exponential distribution`
- Usually describes the distribution of time between events where events happens continously and independently at a constant average rate
- Laplace distribution is similar to exponential distribution but instead of just positive values, it also has a mirror portion of negative values

## `Binomial and Multinomial`
- Binomial distribution summerizes the number of trials or observations when each trial has the same probability
- Multinomial disitrbution is the generailization of binomial 

## `Poisson`
- To count data like the number of cars driving by in a min
  
## `Preprocessing data for model input`
- Use Box-Cox transformation to normalize non-normal variables so we can run broader tests
- Standard normal distribution is ideal for machine learning:
  - Subtract mean so that $\mu=0$
  - Divide by standard deviation so that $\sigma=1$
  - In neural network, we can pass inputs through a normalization layer
- Encode binary variables as 0 and 1

## `Information theory`
- Likelier events have less info than rare ones, for example: the sun rises in the morning every day
  $$I(x)=-logP(x)$$
  ```python
  import numpy as np

  def self_info(p):
      return -1*np.log(p)

  print(self_info(1))
  # -0.0 no information
  print(self_info(0.1))
  # 2.3025850929940455
  print(self_info(0.001))
  # 6.907755278982137
  ```
- The units of self information vary from bit to shannon

## `Shannon and differential entropy`
- To quantify the self-information of a probability distribution, we use Shannon entropy, denoted $H(x)$ or $H(P)$
- Low entropy means the outcome is highly certain
- High entropy means the outcome is highly uncertain
- Shannon entropy for a binary random variable (coin flip) is:
  $$(p-1)log(1-p)-plogp$$
  ```python
  def binary_entropy(p):
      return (p-1)*np.log(1-p)-p*np.log(p)

  print(binary_entropy(0.99999))
  # close to always tail (edge of the distribution)
  # almost certain tail => low entropy
  # 0.00012512920464901166
  print(binary_entropy(0.0001))
  # close to always heads
  # 0.0010210290370309323
  print(binary_entropy(0.9))
  # head 90% of the times
  # 0.3250829733914482
  print(binary_entropy(0.5))
  # head 50% => highly uncertain
  # 0.6931471805599453
  ```
## `Kullback-Leibler divergence and Cross-entropy`
- KL divergence enables us to quantify the shannon entropy of 2 probability distributions over the same random variable $x$
- Cross-entropy is a concept derived from KL divergence, provides us with a cross-entropy cost function
- This cost function is ubiquitous 

## `Statistics`
- Examine data distribution
- Examine relationships between data
- Compare model performance
- Ensure model isn't biased against particular demographic groups
- Bayesian stats can be used when:
  - The sample sizes are not very large
  - Typically have evidence for priors
  ```python
  import numpy as np
  import scipy.stats as st
  import matplotlib.pyplot as plt
  import seaborn as sns

  np.random.seed(42)
  x = st.skewnorm.rvs(10, size=1000)

  xbar = x.mean()
  median = np.median(x)
  std = x.std()
  # standard deviation
  stde = st.sem(x, ddof=0)
  # standard error

  fig, ax = plt.subplots()
  plt.axvline(x = xbar, color='orange')
  plt.axvline(x = xbar+std, color='green')
  plt.axvline(x = xbar-std, color='green')
  plt.hist(x, color='lightgray')
  # the 2 green lines are 2 standard deviation lines
  # the line in the middle is mean line
  plt.show()
  ```
## `z-scores and outliners`
- A z-score represents how many `standard deviations` aways from the mean a data point is
  $$z=\frac{x_i-\mu}{\sigma}$$
- If a data point lies more than 3 standard deviations away from the mean an outlier

## `p-value`
- A p-value is to quantify the probability that an observation would occur by chance a lone
- For example: Given 10000 exam results, there are 67 results attained z-score above 2.5 and 69 results attained z-score below -2.5. Thus if we pick a random sample, the probability of this sample to be outside of 2.5 deviation is:
  $$\frac{67+69}{10000}=0.0136$$
- If we increase the size from 10k to infinity, the probability of sample outside of 2.5 std can be determined with the `cumulative distribution function (CDF)`
  ```python
  p_below = st.norm.cdf(-2.5)
  # 0.006209665325776132
  # If we have an infinite students then 62 will fall below z-score of -2.5
  p_above = 1-st.norm.cdf(2.5)
  # 0.006209665325776159
  # Thera are 62 students will fall above z-score of 2.5
  # st.norm.cdf(2.5) is number of students fall within z-score 2.5, which is the majority
  ```
- In frequentist statistics, if a p-value is less than 0.05, we say it's statistically significant
- If we take sample and its p-value is less than 5%, we say it's statistically meaningful because it's unlikely to happen
- A `null hypothesis` is the baseline assumption, for example, if we are handed a coin, we assume the coin is fair.
- If we flip this coin and have 6 heads/tails in a row, we should reject the null hypothesis because the chance of this happening is , p=3%<5%, so this is not a fair coin.
- `Percent point function (PPF)` is the inverse of `CDF`. It shows the values around the mean that are within the given z-score
  ```python
  st.norm.ppf(.025)
  # -1.9599639845400545
  # 95% of the values, except the 2.5% at the bottom, have z-score greater than -1.95
  st.norm.ppf(.975)
  # 1.9599639845400545
  # 95% of the values except the 2.5% at the top have z-score less than 1.95
  # so if a value has z-score < -1.95 or > 1.95 it's consider statistically signinficant
  ```
## `Single-sample t-test and degrees of freedom`
- Single-sample t-test is a variant of the z-score
  $$t=\frac{\overline{x}-u_0}{s_{\overline{x}}}$$
  - $\overline{x}$ is the sample mean
  - $u_0$ is a reference mean
  - $s_{\overline{x}}$ is the sample standard error
- Let's say you're a brewer, the baseline brewing processing yields 50L of stout. Using modified yeast, you obtain the following yields in 4 experiments
  ```python
  x = [48, 50, 54, 60]
  ```
- The t value is:
  ```python
  xbar = np.mean(x)
  sx = st.sem(x)
  t = (xbar - 50)/sx
  # 1.1338934190276817
  ```
- We can convert t value to p value
  ```python
  def p_from_t(t, n):
      return 2*st.t.cdf(-abs(t), n-1) # 2nd arg of t.cdf() is degrees of freedom

  p_from_t(t, 4)
  # 0.3392540508564543
  # this is greater than 0.05 so this result is not statistically significant and doesn't vary much from the baseline
  ```
- Degress of freedom is if we have to pick a sample of $n$ elements and we already know the mean, the $n-1$ elements can be picked freely but in order to maintain the mean, the last element has to be picked specifically
  ```python
  st.ttest_1samp(x, 50)
  # x is the vector and 50 is the mean
  ```
## `Independent t-tests`
- To compare t-value from multiple samples
  $$t=\frac{\overline
  {x}-\overline{y}}{\sqrt{\frac{s^2_x}{n_x}+\frac{s^2_y}{n_y}}}$$
  ```python
  st.ttest_ind(f, m, equal_var=False)
  # calculate t-test for means of 2 independent vectors f and m
  # equal_var if true assume equal population variances, false use Welch's test
  ```
## `Paired t-test`
- When we collect data from the same sample on 2 different occasions, we can use paired t-test to evaluate if the mean difference is 0 or not.
- This is useful in cases such as a group of people taking diet, we can weight them before and after, pair 2 results of 1 person and see if the diet is significantly useful.
  $$t=\frac{\overline{d}-\mu_0}{s_{\overline{d}}}$$
  - $d$ is a vector of differences between $x$ and $y$
  - $\overline{d}$ is the mean of the differences
  - $\mu_0$ is typically 0, meaning the null hypothesis is that there is no difference between $x$ and $y$
  - $s_{\overline{d}}$ is the standard error of the differences
  ```python
  st.ttest_rel(min15, min1)
  # compare heart rate after 15 mins walking vs 1 min
  ```
## `Application of t-test in machine learning`
- Compare models by running the models and collect samples. If the differences between different inputs are significant, the models may have biases or problems

## `Confidence interval`
- A range where the true mean is likely to be found with a certain probability
- The 95% and 99% are often used as high probabilities.

## `Correlation`
- If we have 2 vectors $x$ and $y$, each element of $x$ pairs with each element of $y$, covariance provides measure of how related the variables are to each other.
- The drawback of covariance is that it confounds the relative scale of 2 variables.
- Correlation overcomes this by rescaling, measuring relatedness.

## `R-Squared coefficient of determination`
- $R^2$ is a number ranges from 0-1 that measures how well a statistical model predicts an outcome.
- It's a proportion of variation in the independent variable that is predicted by the model.
- If $R^2=0.75$ meaning 75% of variance of $x$ can be explained by $y$
- $R^2$ is equal to Pearson's $r$ squared

## `Causation`
- Correlation doesn't imply causation on its own.
- To be causation, there are 3 criterions:
  - `Covariance`: 2 variables must vary together
  - `Temporal precedence`: the causer must vary before the affected variable varies
  - `Elimination of extraneous variable`: the variation cannot be caused by a third variable

## `Correcting for Multiple comparisons`
- Bonferroni correction: assuming in all tests, we assume overall threshold $\alpha=0.05$, if we conduct 10 tests, then each test $\alpha=0.005$

## `Independent vs dependent variables`
- Outcome:
  - Dependent variables
  - Denoted with $y$
- Feature
  - Independent variables
  - May predict the outcome
  - Denoted with $x$

## `Linear regression model`
  $$y=a+bx_1+cx_2+...+mx_n$$
- $a$ is y-intercept because we rarely wants out models to predict $y=0$

## `Fitting a line to points on a Cartesian plane`
- If $y=\beta_0+\beta_1x+\epsilon$, we want $\epsilon$ to be close to 0 as possible so that $\hat{y}$ is close to $y$
- In case of a model with a single predictor $x$ we can use linear least squares formula to calculate $\beta_1$
  $$\hat{\beta_1}=\frac{cov(x,y)}{\sigma^2_x}$$
  ```python
  import numpy as np
  import matplotlib.pyplot as plt
  import seaborn as sns

  iris = sns.load_dataset('iris')

  x = iris.sepal_length
  y = iris.petal_length
  n = iris.sepal_width.size

  sns.scatterplot(x=x, y=y)
  xbar, ybar = x.mean(), y.mean()
  product = []
  for i in range(n):
      product.append((x[i]-xbar)*(y[i]-ybar))

  cov = sum(product)/n
  beta1 = cov/np.var(x)
  # 18584329782548417
  # Once we have beta1, we can calculate beta0
  beta0 = ybar - beta1*xbar
  # -7.101443369602459 
  xline = np.linspace(4, 8, 1000)
  yline = beta0+beta1*xline

  plt.plot(xline, yline, color='orange')
  plt.show()
  ```

## `Ordinary least squares`
- It's amethod to estimate the parameters of regression models with more than 1 predictor
  $$y=\beta_0+\beta_1x_1+\beta_2x_2+\epsilon$$

- Typicall a regression model has more equations than the number of unknowns thus we can't solve them by rearraging.
- However we can use Moore-Penrose Psudoinverse or partial derivative calculus to estimate. The goal is to reduce the sum of squared errors (SSE)
  $$\sum(\hat{y_i}-y_i)^2$$
- For example given the equations
  $$6=\beta_0+\beta_1$$
  $$5=\beta_0+2\beta_1$$
  $$7=\beta_0+3\beta_1$$
  $$10=\beta_0+4\beta_1$$
- The SSE is:
  $$S(\beta_0,\beta_1)=4\beta_0^2 + 30\beta_1^2 + 20\beta_0\beta_1 - 56\beta_0 - 154\beta_1 + 210$$
- This is a 2nd order equation with curves. To minimize SSE, we can use partial derivative to find where there is no slope of SSE with respect to $\beta_0$ and $\beta_1$
  $$\frac{\partial S}{\partial \beta_0}=8\beta_0+20\beta_1-56=0$$
  $$\frac{\partial S}{\partial \beta_1}=20\beta_0+60\beta_1-154=0$$

## `Categorical Dummy features`
- Categorical features can be represented using dummy features
- Given 3 categories: setosa, versicolor, virginica. For equation of type setosa, the setosa's dummy feature is 1 while the other 2 are 0.

## `Logistic regression to predict categories`
- We can use logit function to predict binary outcome (hot dog or not hot dog)
  $$log(\frac{p}{1-p})$$
  ```python
  import scipy.special as sc
  sc.logit(0.5)
  # 0.0
  sc.logit(0.00000000000000000000000000001)
  # -66.77496769682732
  # the closer to 0 the closer logit to negative infinity
  sc.expit(0.0)
  # 0.5
  sc.expit(-66.77496769682732)
  # 1.000000000000007e-29
  ```
- Use logistic regression to predict if a passenger can survive the Titanic
  ```python
  import pandas as pd
  import statsmodels.api as sm

  titanic = sns.load_dataset('titanic')

  gender = pd.get_dummies(titanic['sex'])
  clas = pd.get_dummies(titanic['class'])
  y = titanic.survived
  # 0 0
  # 1 1
  # 2 1
  # 3 1
  # 4 0 
  X = pd.concat([clas.First, clas.Second, gender.female, titanic.age], axis=1)
  X = sm.add_constant(X)
  # Add constant because we always want the y-intercept to be non-zero
  #    const  First  Second  female    age
  #0     1.0     0      0      0       22.0
  #1     1.0     1      0      1       38.0
  #2     1.0     0      0      1       26.0
  #3     1.0     1      0      1       35.0
  #4     1.0     0      0      0       35.0
  model = sm.Logit(y, X, missing = 'drop')
  result = model.fit()

  beta = result.params
  # const    -1.326394
  # First     2.580625
  # Second    1.270826
  # female    2.522781
  # age      -0.036985
  linear_out = beta[0] + beta[1]*1 + beta[3]*1 + beta[4]*17
  # If I'm in first class, female and age of 17, my surviving probability is
  sc.expit(linear_out)
  # 0.9588402300157423
  ```

## `Bayesian statistics`
- Older than Frequentist statistics how ever today machine learning deals with big datasets
- No arbitrary threshold 0.05
- Used with smaller feature sets, prior information

## `Prior probabilities`
- Can be obtained from:
  - Observations
  - Existing literature / knowledge
  - Tangentially-related model results
  - Reasoning
  - Hunches

## `Bayes' Theorem`
- 