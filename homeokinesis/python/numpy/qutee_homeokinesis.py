import numpy as np
import py_dynamixel.io as io
import time
from robots.qutee_parameters import *
from robots.qutee_utils import *
from controllers.homeokinesis import Homeokinesis
from robots.qutee import Qutee

control_time_step = 0.1  
control_frequency = 1

def main():
	ports = io.get_available_ports()
	print('available ports:', ports)
	if not ports:
		raise IOError('No port available.')

	port = ports[0]
	print('Using the first on the list', port)
	ctrl_freq = 100
	qutee = Qutee(port, ctrl_freq)
	hk = Homeokinesis(N_MOTORS, N_MOTORS)

	## Go to goal pos slowly, in robot angle frame
	init_pos = clip_motor_pos(scale_control(np.zeros(12), safe=True), safe=True)
	actual_pos = qutee.get_present_position()
	init_time = 100
	s = 0.0
	while(s <= init_time):
		rate = s / init_time
		x = actual_pos * (1.0 - rate) + init_pos * rate
		# print(s, rate,  "\nin", init_pos, "\nxx", x, "\nac", \
			# actual_pos, "\npr", scale_state(qutee.get_present_position(), safe=False)) #, "\ngp: ", qutee.get_goal_position())
		qutee.set_goal_position(x)
		s += 1.0

	qutee.get_present_position()
	then = time.perf_counter()
	steps = 0
	while(True):
		now = time.perf_counter()
		if (now - then > control_time_step):
			steps += 1
			# print(steps, hk.t)
			then = now
			if (steps % control_frequency == 0):
				x_raw = qutee.get_present_position()
				x = scale_state(x_raw, safe=True)
				y = hk.step(x)
				print(x, y)
				if (y is not None): 
					y_raw = clip_motor_pos(scale_control(y, safe=True), safe=True)
					# print("\ny robot", y)
					qutee.set_goal_position(y_raw)
					# print("C: ", hk.get_C(), "\nA: ", hk.get_A())
					#print("x", x, "\ny", y, "\nx_raw", x_raw, "\ny_raw", y_raw)
	qutee.shutdown()


if __name__ == "__main__":
	main()
