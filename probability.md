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