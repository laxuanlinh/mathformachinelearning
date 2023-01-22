import torch
import numpy as np
import tensorflow as tf

x = tf.Variable(5.0)
with tf.GradientTape() as t:
	#track forward pass
	t.watch(x)
	y = x**2
#auto diff
print(t.gradient(y, x))
