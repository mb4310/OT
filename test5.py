def first_and_second_moments(x):
	def g1(x):
		return x[0]

	def g2(x):
		return x[1]

	def g3(x):
		return x[2]

	def g4(x):
		return x[0]*x[1]

	def g5(x): 
		return x[1]*x[2]

	def g6(x): 
		return x[0]*x[2]

	def g7(x):
		return x[0]**2

	def g8(x): 
		return x[1]**2

	def g9(x): 
		return x[2]**2

	def f11(x):
		return 1

	def f12(x):
		return 0

	def f13(x): 
		return 0

	def f21(x):
		return 0

	def f22(x):
		return 1

	def f23(x):
		return 0

	def f31(x):
		return 0

	def f32(x):
		return 0

	def f33(x):
		return 1

	def f41(x):
		return x[1]

	def f42(x):
		return x[0]

	def f43(x):
		return 0

	def f51(x):
		return 0

	def f52(x):
		return x[2]

	def f53(x):
		return x[1]

	def f61(x):
		return x[2]

	def f62(x):
		return 0

	def f63(x):
		return x[0]

	def f71(x):
		return 2*x[0]

	def f72(x):
		return 0

	def f73(x):
		return 0

	def f81(x):
		return 0

	def f82(x):
		return 2*x[1]

	def f83(x):
		return 0

	def f91(x):
		return 0

	def f92(x):
		return 0

	def f93(x):
		return 2*x[2]

	M = np.array([g1, g2, g3, g4, g5, g6, g7, g8, g9])

	J_M = np.array([[f11,f12,f13],[f21,f22,f23],[f31,f32,f33],[f41,f42,f43],[f51,f52,f53],[f61,f62,f63],[f71,f72,f73],[f81,f82,f83],[f91,f92,f93]])














