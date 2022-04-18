# def foo(re,*args,**kwargs):
# 	print(re)
# 	if args:
# 		print(f'args is {args}')
# 		print(type(args))
# 	if kwargs:
# 		print(f'args is {kwargs}')
# 		print(type(kwargs))
# def outer(func):
#
# 	def inner(x1,y1):
# 		nonlocal x1,y1
# 		y+=y1
# 		x+=x1
# 		func()
# 		print(f'now, x={x}, y={y}')
# 	return inner
# move = outer
# move(1,2)
# move(-2,2)

import time
# def time_master(func,x):
# 	def call_func():
# 		print('start',x)
# 		start = time.time()
# 		func()
# 		end = time.time()
# 		print('end')
# 		print(f'total time: {(end-start):.2f}s')
# 	return call_func
# @time_master
# def myfunc(x=0):
# 	time.sleep(0.5)
# 	print('this is my func')
# myfunc()
# def time_master( func ):
# 	def call_func( c,m ):
# 		start = time.time( )
# 		func( c )
# 		end = time.time( )
# 		print( 'end' )
# 		print( f'total time: {(end - start):.2f}s' )
# 		# print( f'msg is {msg}' )
#
# 	return call_func
# def logger(msg):
# 	def time_master(func):
# 		def call_func(c):
# 			start = time.time( )
# 			func( c )
# 			end = time.time( )
# 			print( 'end' )
# 			print( f'total time: {(end - start):.2f}s' )
# 			print(f'msg is {msg}')
# 		return call_func
# 	return time_master
# @time_master
# def funA(c):
# 	time.sleep(0.5)
# 	print('this is funA', c)
# funA(c=123,m=345)

# def start_end_d(func):
# 	def wrapper(*args, **kwargs):
# 		print('star')
# 		re = func(*args, **kwargs)
# 		print('end')
# 		return re
# 	return wrapper
# @start_end_d
# def add(x):
# 	return x +5
# result = add(10)
# import numpy as np
# def get(*args, **kwargs):
# 	for i in args:
# 		print(i)
# 	for j in kwargs:
# 		print(j)
# a = [1,2,3]
# b = np.random.randint(15)
# get(a,b,[123,312,43],f =2)
import copy

def plus(a):

	a.append(2)
if __name__ == '__main__':
	sin = [1,2,3,4,5]
	print(sin)
	plus(a = sin)
	print(sin)
