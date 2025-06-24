# Following Sox controller from LpzRobots implementation
import numpy as np

class Homeokinesis:
	def __init__(self, number_sensor, number_motor):
		self.t = 0
		self.buffer_size = 10
		self.number_sensor = number_sensor
		self.number_motor = number_motor
		self.loga = False
		self.eps_C = 0.001
		self.eps_A = 0.005
		self.sense = 1.0
		self.creativity = .1 #1.0
		self.damping = 0.001
		self.init_feedback_strength = 1.0 #1.0
		self.use_extended_model = True
		#self.use_teaching = True
		self.causeaware = 0.01 if self.use_extended_model else 0.0
		self.harmony = 0.0
		self.gamma = 0.0
		self.inter_is_teaching = False
		self.A = np.eye(self.number_sensor, self.number_motor)
		self.S = np.eye(self.number_sensor, self.number_motor) * 0.05
		self.C = np.eye(self.number_motor, self.number_sensor) * self.init_feedback_strength # + np.random.rand(self.number_motor, self.number_sensor) * 0.1
		self.b = np.zeros((self.number_sensor))
		self.h = np.zeros((self.number_motor))
		self.L = np.zeros((self.number_sensor, self.number_sensor))
		self.v_avg = np.zeros((self.number_sensor))
		self.A_native = np.eye(self.number_sensor, self.number_motor)
		self.C_native = np.eye(self.number_motor, self.number_sensor) * 1.2
		self.R = np.zeros((self.number_sensor, self.number_sensor))
		self.y_teaching = np.zeros((self.number_motor))
		self.x = np.zeros((self.number_sensor))
		self.x_smooth = np.zeros((self.number_sensor))
		self.x_buffer = np.zeros((self.number_sensor, self.buffer_size))
		self.y_buffer = np.zeros((self.number_motor, self.buffer_size))

		self.steps_for_averaging = 10
		self.steps_for_delay = 10
		self.factor_S = 1.0
		self.factor_b = 1.0
		self.factor_h = 1.0
		self.clip_limit = 0.005


	def get_A(self):
		return self.A

	def set_A(self, A):
		self.A = A

	def get_C(self):
		return self.C

	def set_C(self, C):
		self.C = C

	def get_h(self):
		return self.h
	
	def set_h(self, h):
		self.h = h

	def g(self, z):
		return np.tanh(z)

	def g_s(self, z):
		return 1.0 - np.tanh(z) ** 2

	def step(self, x_):
		yy_ = self.step_no_learning(x_)
		if (self.t <= self.buffer_size): return
		self.t += -1  # step_no_learning increments t
		if (self.eps_C != 0.0 or self.eps_A != 0.0):
			self.learn()
		# print (self.C, self.h)
		# print (self.h)
		self.t += 1
		return yy_
	
	def step_no_learning(self, x_):
		assert x_.size <= self.number_sensor
		self.x = np.copy(x_)
		self.steps_for_averaging = np.clip(self.steps_for_averaging, 1, self.buffer_size - 1)
		if (self.steps_for_averaging > 1):
			self.x_smooth += (self.x - self.x_smooth) / self.steps_for_averaging
		else:
			self.x_smooth = np.copy(self.x)
		self.x_buffer[:, int(self.t % self.buffer_size)] = np.copy(self.x_smooth)
		y_ = self.g(self.C.dot(self.x_smooth + (self.v_avg * self.creativity)) + self.h)
		self.y_buffer[:, self.t % self.buffer_size] = np.copy(y_)
		self.t += 1
		return y_


	def learn(self):
		s_for_delay = np.clip(self.steps_for_delay, 1, self.buffer_size - 1)
		x = np.copy(self.x_buffer[:, int((self.t - max(s_for_delay, 1) + self.buffer_size) % self.buffer_size)])
		y_creat = np.copy(self.y_buffer[:, int((self.t - max(s_for_delay, 1) + self.buffer_size) % self.buffer_size)])
		x_fut = np.copy(self.x_buffer[:, int(self.t % self.buffer_size)])
		# xi = np.abs(x_fut - (self.A.dot(y_creat) + self.b + self.S.dot(x))) # shouldn't this bee absolute value?
		xi = x_fut - (self.A.dot(y_creat) + self.b + self.S.dot(x)) # shouldn't this bee absolute value?
		z = self.C.dot(x) + self.h
		y = np.copy(self.g(z))
		g_prime = np.copy(self.g_s(z))
		self.L = self.A.dot(self.C.dot(g_prime)) + self.S
		R = self.A.dot(self.C) + self.S
		eta = np.linalg.pinv(self.A).dot(xi)
		y_hat = y + eta * self.causeaware
		Lplus = np.linalg.pinv(self.L)
		v = Lplus.dot(xi)
		chi = np.transpose(Lplus).dot(v)
		mu = (np.transpose(self.A) * (g_prime)).dot(chi)
		mu2 = self.C.dot(v)
		eps_rel = mu * mu2 * (self.sense * 2.0)
		v_hat = v + x * self.harmony
		self.v_avg += (v - self.v_avg) * 0.1
		EE = 1.0
		if (self.loga):
			EE = 0.1 / (np.linalg.norm(v) + 0.001)
		if (self.eps_A > 0.0):
			eps_S = self.eps_A * self.factor_S
			eps_b = self.eps_A * self.factor_b
			self.A += np.clip(xi * np.transpose(y_hat) * self.eps_A, -self.clip_limit, self.clip_limit)
			if (self.damping > 0.0):
				self.A += np.clip(((self.A_native - self.A) ** 3) * self.damping, -self.clip_limit, self.clip_limit)
			if (self.use_extended_model):
				self.S += np.clip(xi.dot(np.transpose(x)) * eps_S + (self.S.dot((-self.damping * 10))), -self.clip_limit, self.clip_limit)
			self.b += np.clip(xi * eps_b + (self.b * (-self.damping)), -self.clip_limit, self.clip_limit)
		if (self.eps_C > 0.0):
			self.C += np.clip((mu.dot(np.transpose(v_hat))
			       -((eps_rel * y).dot(np.transpose(x)))) * (EE * self.eps_C), -self.clip_limit, self.clip_limit)
			if (self.damping > 0.0):
				self.C += np.clip(((self.C_native - self.C) ** 3) * self.damping, -self.clip_limit, self.clip_limit)
			self.h += np.clip((mu * self.harmony - (eps_rel * y)) * (EE * self.eps_C * self.factor_h) , -self.clip_limit, self.clip_limit)
			# if (self.inter_is_teaching and self.gamma > 0.0):
			# 	metric = np.transpose(self.A).dot((Lplus * np.transpose(Lplus))).dot(self.A)  # Order of Lplus multiplication ?
			# 	y = np.copy(self.y_buffer[:, int((self.t - 1) % self.buffer_size)])
			# 	xsi = self.y_teaching - y
			# 	delta = xsi * g_prime
			# 	C += np.clip((metric * delta.dot(np.transpose(x))) * (self.gamma * self.eps_C), -0.05, 0.05)
			# 	h += np.clip((metric * delta) * (self.gamma * self.eps_C * self.factor_h), -0.05, 0.05)
			# 	self.inter_is_teaching = False