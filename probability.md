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
  print(np.median(heads_count))
  #3.0
  ```