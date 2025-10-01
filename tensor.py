# import tensorflow as tf

# a = tf.constant([[1, 2], [3, 4]], dtype=tf.int32) 
# b = tf.constant([[5, 6], [7, 8]], dtype=tf.int32)

# c = a + b  
# print("c:", c.numpy()) 

# w = tf.Variable([[0.1, 0.2], [0.3, 0.4]], dtype=tf.float32)
# w.assign_add(tf.ones_like(w) * 0.1) 
# print("w:", w.numpy())
# import tensorflow as tf

# X = tf.Variable(10)
# @tf.function
# def my_func(x):
#     return tf.reduce_sum(x)
# print(my_func(X))

# def fun(x):
#   return x**2+2*x-6

# with tf.GradientTape() as tape:
#   y = fun(x)

# grad = tape.gradient(y, x)
# print(grad)

# var = tf.Variable([0.0, 0.0, 0.0])
# var.assign([1, 2, 3])
# var.assign_add([1, 1, 1])
# print(var)

# v1 = tf.Variable([0.0 , 1.0 , 2.0])
# v1.assign([1,2,3])
# v1.assign_sub([4,5,6])
# print(v1)
# x = tf.constant([[1., 2., 3.],
#                  [4., 5., 6.]])

# print(x)
# print("-------")
# print(x.shape)
# print(x.dtype)
# print(x + x)
# print("--------------")
# print(5 * x)
# print("============")
# print(x @ tf.transpose(x))
# print("-------------")
# print(tf.concat([x, x], axis=1))


# import tensorflow as tf
# import time

# def slow_func(x):
#   return tf.square(x)

# @tf.function
# def fast_func(x):
#   return tf.square(x)

# x = tf.random.normal([10000,10000])

# start = time.time()
# for _ in range(100):
#   slow_func(x)
# print("Time taken in slow function" , time.time()-start)

# start = time.time()
# for _ in range(100):
#   fast_func(x)
# print("Time Taken in fast function" , time.time()-start)