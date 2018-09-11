import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer

import layers

ap2 = layers.ap2

class ShiftBasedAdaMaxOptimizer(optimizer.Optimizer):
	
	def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
					use_locking=False, name="ShiftBasedAdaMax"):
		super(ShiftBasedAdaMaxOptimizer, self).__init__(use_locking, name)
		
		self.lr = learning_rate
		self.beta1 = beta1
		self.beta2 = beta2
		self.epsilon = epsilon
		
		self._lr_t = None
		self._beta1_t = None
		self._beta2_t = None
		self._eps_t = None

	
	def _prepare(self):
		self._lr_t = ops.convert_to_tensor(self.lr, name="learning_rate")
		self._beta1_t = ops.convert_to_tensor(self.beta1, name="beta1")
		self._beta2_t = ops.convert_to_tensor(self.beta2, name="beta2")
		self._eps_t = ops.convert_to_tensor(self.epsilon, name="epsilon")
	
	
	def _create_slots(self, var_list):
		first_var = min(var_list, key=lambda x: x.name)
		self._create_non_slot_variable(initial_value=self.beta1, name="beta1_power",  colocate_with=first_var)
		
		for v in var_list:
			self._zeros_slot(v, "m", self._name)
			self._zeros_slot(v, "v", self._name)
	
	
	def _get_beta_accumulator(self):
		graph = ops.get_default_graph()
		return self._get_non_slot_variable("beta1_power", graph=graph)
	
	
	def _apply_dense(self, grad, var):
		
		m = self.get_slot(var, "m")
		v = self.get_slot(var, "v")
		
		_beta1_power = self._get_beta_accumulator();
		
		beta1_power = math_ops.cast(_beta1_power, grad.dtype.base_dtype)
		lr = math_ops.cast(self._lr_t, grad.dtype.base_dtype)
		beta1 = math_ops.cast(self._beta1_t, grad.dtype.base_dtype)
		beta2 = math_ops.cast(self._beta2_t, grad.dtype.base_dtype)
		eps = math_ops.cast(self._eps_t, grad.dtype.base_dtype)
		
		v_t = v.assign(beta1 * v + (1. - beta1) * grad)
		m_t = m.assign(tf.maximum(beta2 * m + eps, tf.abs(grad)))
		
		# g_t = v_t / m_t
		# var_update = state_ops.assign_sub(var, (lr / (1 - beta1_power)) * g_t)
		
		# here we apply the shift on the learning rate and on m moment. Since the shift operation
		# is defined for integer values only, we compute them as the approximate power of two
		# of the original operations (lr/(1-beta1^t)) and (v_t / m_t). With dedicated hardware
		# these operations would have been done just by using shift ops.
		lr_c = lr / (1 - beta1_power)
		lr_c = ap2(lr_c)
		g_t = v_t / ap2(m_t)
		
		var_update = state_ops.assign_sub(var, lr_c * g_t)
		
		return control_flow_ops.group(*[var_update, m_t, v_t])
		
		
	def _apply_sparse(self, grad, var):
		raise NotImplementedError("Sparse gradient updates are not supported.")
	
	
	def _finish(self, update_ops, name_scope):
		# Update the power accumulator.
		with ops.control_dependencies(update_ops):
			beta1_power = self._get_beta_accumulator()
			with ops.colocate_with(beta1_power):
				update_beta1 = beta1_power.assign(beta1_power * self._beta1_t, use_locking=self._use_locking)
		return control_flow_ops.group(*update_ops + [update_beta1], name=name_scope)
		
