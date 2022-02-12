# def foo(re,*args,**kwargs):
# 	print(re)
# 	if args:
# 		print(f'args is {args}')
# 		print(type(args))
# 	if kwargs:
# 		print(f'args is {kwargs}')
# 		print(type(kwargs))
def outer(func):

	def inner(x1,y1):
		nonlocal x,y
		y+=y1
		x+=x1
		func()
		print(f'now, x={x}, y={y}')
	return inner
move = outer
move(1,2)
move(-2,2)

import time
def time_master(func):
	def call_func():
		print('start')
		start = time.time()
		func()
		end = time.time()
		print('end')
		print(f'total time: {(end-start):.2f}s')
	return call_func
@time_master
def myfunc():
	time.sleep(2)
	print('this is my func')
myfunc()

func = time_master(myfunc)
func()
