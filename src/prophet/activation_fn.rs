use num::{Float};
use std::ops::{Deref};

pub type ActivationFn<F> = fn(F) -> F;

pub struct BaseDerivedActivationFn<F: Float> {
	pub base: ActivationFn<F>,
	pub derived: ActivationFn<F>
}

pub fn identity_fn<F: Float>(x: F) -> F { x }
pub fn identity_fn_dx<F: Float>(_: F) -> F { F::one() }

pub fn binary_step_fn<F: Float>(x: F) -> F {
	if x < F::zero() { F::zero() } else { F::one() }
}
pub fn binary_step_fn_dx<F: Float>(x: F) -> F {
	if x != F::zero() { F::zero() } else { F::infinity() }
}

pub fn logistic_fn<F: Float>(x: F) -> F {
	softplus_fn_dx(x)
}
pub fn logistic_fn_dx<F: Float>(x: F) -> F {
	logistic_fn(x) * (F::one() - logistic_fn(x))
}

pub fn tanh_fn<F: Float>(x: F) -> F {
	x.tanh()
}
pub fn tanh_fn_dx<F: Float>(x: F) -> F {
	let fx = tanh_fn(x);
	F::one() - fx*fx
}

pub fn arctan_fn<F: Float>(x: F) -> F {
	x.atan()
}
pub fn arctan_fn_dx<F: Float>(x: F) -> F {
	F::one() / (x*x + F::one())
}

pub fn softsign_fn<F: Float>(x: F) -> F { x / (F::one() + x.abs()) }
pub fn softsign_fn_dx<F: Float>(x: F) -> F { let dx = F::one() + x.abs(); F::one() / (dx*dx) }

pub fn relu_fn<F: Float>(x: F) -> F { if x < F::zero() { F::zero() } else { x } }
pub fn relu_fn_dx<F: Float>(x: F) -> F { if x < F::zero() { F::zero() } else { F::one() } }

pub fn softplus_fn<F: Float>(x: F) -> F {
	x.exp().ln_1p()
}
pub fn softplus_fn_dx<F: Float>(x: F) -> F {
	F::one() / (F::one() + (-x).exp())
}

pub fn bent_identity_fn<F: Float>(x: F) -> F {
	// ((sqrt(x^2 + 1) - 1) / 2) + x
	let two = F::from(2.0).unwrap();
	(((x*x) + F::one()).sqrt() - F::one()) / two + x
}
pub fn bent_identity_fn_dx<F: Float>(x: F) -> F {
	// (x / (2 * sqrt(x^2 + 1))) + 1
	let two = F::from(2.0).unwrap();
	x / (two * ((x * x) + F::one()).sqrt()) + F::one()
}

pub fn sinusoid_fn<F: Float>(x: F) -> F {
	x.sin()
}
pub fn sinusoid_fn_dx<F: Float>(x: F) -> F {
	x.cos()
}

pub fn gaussian_fn<F: Float>(x: F) -> F {
	(-x * x).exp()
}
pub fn gaussian_fn_dx<F: Float>(x: F) -> F {
	let two = F::from(2.0).unwrap();
	-two * x * gaussian_fn(x)
}

impl<F: Float> BaseDerivedActivationFn<F> {
	pub fn new(base: ActivationFn<F>, derived: ActivationFn<F>) -> Self {
		BaseDerivedActivationFn{
			base: base,
			derived: derived
		}
	}

	pub fn identity() -> Self {
		BaseDerivedActivationFn{
			base: identity_fn,
			derived: identity_fn_dx
		}
	}

	pub fn binary_step() -> Self {
		BaseDerivedActivationFn{
			base: binary_step_fn,
			derived: binary_step_fn_dx
		}
	}

	pub fn arctan() -> Self {
		BaseDerivedActivationFn{
			base: arctan_fn,
			derived: arctan_fn_dx
		}
	}

	pub fn tanh() -> Self {
		BaseDerivedActivationFn{
			base: tanh_fn,
			derived: tanh_fn_dx
		}
	}

	pub fn logistic() -> Self {
		BaseDerivedActivationFn{
			base: logistic_fn,
			derived: logistic_fn_dx
		}
	}

	pub fn softsign() -> Self {
		BaseDerivedActivationFn{
			base: softsign_fn,
			derived: softsign_fn_dx
		}
	}

	pub fn relu() -> Self {
		BaseDerivedActivationFn{
			base: relu_fn,
			derived: relu_fn_dx
		}
	}

	pub fn softplus() -> Self {
		BaseDerivedActivationFn{
			base: softplus_fn,
			derived: softplus_fn_dx
		}
	}

	pub fn bent_identity() -> Self {
		BaseDerivedActivationFn{
			base: bent_identity_fn,
			derived: bent_identity_fn_dx
		}
	}

	pub fn sinusoid() -> Self {
		BaseDerivedActivationFn{
			base: sinusoid_fn,
			derived: sinusoid_fn_dx
		}
	}

	pub fn gaussian() -> Self {
		BaseDerivedActivationFn{
			base: gaussian_fn,
			derived: gaussian_fn_dx
		}
	}

	pub fn base(&self, x: F) -> F {
		(self.base)(x)
	}

	pub fn derived(&self, x: F) -> F {
		(self.derived)(x)
	}
}

impl<F: Float> Deref for BaseDerivedActivationFn<F> {
    type Target = ActivationFn<F>;

    fn deref(&self) -> &Self::Target {
    	&self.base
    }
}

#[cfg(test)]
mod tests {
	use num::Float;
	use super::{BaseDerivedActivationFn};

	#[test]
	fn new_base_deriv_act_fn() {
		use super::{logistic_fn, logistic_fn_dx};
		let custom_sigmoid = BaseDerivedActivationFn::<f32>::new(logistic_fn, logistic_fn_dx);
		let predef_sigmoid = BaseDerivedActivationFn::<f32>::logistic();
		assert_eq!(custom_sigmoid.base(-1.0), predef_sigmoid.base(-1.0));
		assert_eq!(custom_sigmoid.base(-0.5), predef_sigmoid.base(-0.5));
		assert_eq!(custom_sigmoid.base( 0.0), predef_sigmoid.base( 0.0));
		assert_eq!(custom_sigmoid.base( 0.5), predef_sigmoid.base( 0.5));
		assert_eq!(custom_sigmoid.base( 1.0), predef_sigmoid.base( 1.0));
	}

	#[test]
	fn activation_fn_deref() {
		let act_fn_pair = BaseDerivedActivationFn::<f32>::tanh();
		assert_eq!(act_fn_pair.base(1.0), act_fn_pair(1.0));
	}

	#[test]
	fn identity_activation_fn() {
		let act_fn_pair = BaseDerivedActivationFn::<f32>::identity();
		assert_eq!(act_fn_pair.base(-1.0), -1.0);
		assert_eq!(act_fn_pair.base(-0.5), -0.5);
		assert_eq!(act_fn_pair.base( 0.0),  0.0);
		assert_eq!(act_fn_pair.base( 0.5),  0.5);
		assert_eq!(act_fn_pair.base( 1.0),  1.0);
		assert_eq!(act_fn_pair.derived(-1.0), 1.0);
		assert_eq!(act_fn_pair.derived(-0.5), 1.0);
		assert_eq!(act_fn_pair.derived( 0.0), 1.0);
		assert_eq!(act_fn_pair.derived( 0.5), 1.0);
		assert_eq!(act_fn_pair.derived( 1.0), 1.0);
	}

	#[test]
	fn binary_step_activation_fn() {
		let act_fn_pair = BaseDerivedActivationFn::<f32>::binary_step();
		assert_eq!(act_fn_pair.base(-1.0), 0.0);
		assert_eq!(act_fn_pair.base(-0.5), 0.0);
		assert_eq!(act_fn_pair.base( 0.0), 1.0);
		assert_eq!(act_fn_pair.base( 0.5), 1.0);
		assert_eq!(act_fn_pair.base( 1.0), 1.0);
		assert_eq!(act_fn_pair.derived(-1.0), 0.0);
		assert_eq!(act_fn_pair.derived(-0.5), 0.0);
		assert_eq!(act_fn_pair.derived( 0.0), <f32>::infinity());
		assert_eq!(act_fn_pair.derived( 0.5), 0.0);
		assert_eq!(act_fn_pair.derived( 1.0), 0.0);
	}

	#[test]
	fn logistic_activation_fn() {
		let act_fn_pair = BaseDerivedActivationFn::<f32>::logistic();
		assert_eq!(act_fn_pair.base(-1.0), 0.26894143);
		assert_eq!(act_fn_pair.base(-0.5), 0.37754068);
		assert_eq!(act_fn_pair.base( 0.0), 0.5);
		assert_eq!(act_fn_pair.base( 0.5), 0.62245935);
		assert_eq!(act_fn_pair.base( 1.0), 0.7310586);
		assert_eq!(act_fn_pair.derived(-1.0), 0.19661194);
		assert_eq!(act_fn_pair.derived(-0.5), 0.23500371);
		assert_eq!(act_fn_pair.derived( 0.0), 0.25);
		assert_eq!(act_fn_pair.derived( 0.5), 0.23500371);
		assert_eq!(act_fn_pair.derived( 1.0), 0.19661193);
	}

	#[test]
	fn arctan_activation_fn() {
		let act_fn_pair = BaseDerivedActivationFn::<f32>::arctan();
		assert_eq!(act_fn_pair.base(-1.0), -0.7853982);
		assert_eq!(act_fn_pair.base(-0.5), -0.4636476);
		assert_eq!(act_fn_pair.base( 0.0),  0.0);
		assert_eq!(act_fn_pair.base( 0.5),  0.4636476);
		assert_eq!(act_fn_pair.base( 1.0),  0.7853982);
		assert_eq!(act_fn_pair.derived(-1.0), 0.5);
		assert_eq!(act_fn_pair.derived(-0.5), 0.8);
		assert_eq!(act_fn_pair.derived( 0.0), 1.0);
		assert_eq!(act_fn_pair.derived( 0.5), 0.8);
		assert_eq!(act_fn_pair.derived( 1.0), 0.5);
	}

	#[test]
	fn tanh_activation_fn() {
		let act_fn_pair = BaseDerivedActivationFn::<f32>::tanh();
		assert_eq!(act_fn_pair.base(-1.0), -0.7615942);
		assert_eq!(act_fn_pair.base(-0.5), -0.46211717);
		assert_eq!(act_fn_pair.base( 0.0),  0.0);
		assert_eq!(act_fn_pair.base( 0.5),  0.46211717);
		assert_eq!(act_fn_pair.base( 1.0),  0.7615942);
		assert_eq!(act_fn_pair.derived(-1.0), 0.41997433);
		assert_eq!(act_fn_pair.derived(-0.5), 0.7864477);
		assert_eq!(act_fn_pair.derived( 0.0), 1.0);
		assert_eq!(act_fn_pair.derived( 0.5), 0.7864477);
		assert_eq!(act_fn_pair.derived( 1.0), 0.41997433);
	}

	#[test]
	fn softsign_activation_fn() {
		let act_fn_pair = BaseDerivedActivationFn::<f32>::softsign();
		assert_eq!(act_fn_pair.base(-1.0), -0.5);
		assert_eq!(act_fn_pair.base(-0.5), -0.33333334);
		assert_eq!(act_fn_pair.base( 0.0),  0.0);
		assert_eq!(act_fn_pair.base( 0.5),  0.33333334);
		assert_eq!(act_fn_pair.base( 1.0),  0.5);
		assert_eq!(act_fn_pair.derived(-1.0), 0.25);
		assert_eq!(act_fn_pair.derived(-0.5), 0.44444445);
		assert_eq!(act_fn_pair.derived( 0.0), 1.0);
		assert_eq!(act_fn_pair.derived( 0.5), 0.44444445);
		assert_eq!(act_fn_pair.derived( 1.0), 0.25);
	}

	#[test]
	fn relu_activation_fn() {
		let act_fn_pair = BaseDerivedActivationFn::<f32>::relu();
		assert_eq!(act_fn_pair.base(-1.0), 0.0);
		assert_eq!(act_fn_pair.base(-0.5), 0.0);
		assert_eq!(act_fn_pair.base( 0.0), 0.0);
		assert_eq!(act_fn_pair.base( 0.5), 0.5);
		assert_eq!(act_fn_pair.base( 1.0), 1.0);
		assert_eq!(act_fn_pair.derived(-1.0), 0.0);
		assert_eq!(act_fn_pair.derived(-0.5), 0.0);
		assert_eq!(act_fn_pair.derived( 0.0), 1.0);
		assert_eq!(act_fn_pair.derived( 0.5), 1.0);
		assert_eq!(act_fn_pair.derived( 1.0), 1.0);
	}

	#[test]
	fn softplus_activation_fn() {
		let act_fn_pair = BaseDerivedActivationFn::<f32>::softplus();
		assert_eq!(act_fn_pair.base(-1.0), 0.3132617);
		assert_eq!(act_fn_pair.base(-0.5), 0.474077);
		assert_eq!(act_fn_pair.base( 0.0), 0.6931472);
		assert_eq!(act_fn_pair.base( 0.5), 0.974077);
		assert_eq!(act_fn_pair.base( 1.0), 1.3132616);
		assert_eq!(act_fn_pair.derived(-1.0), 0.26894143);
		assert_eq!(act_fn_pair.derived(-0.5), 0.37754068);
		assert_eq!(act_fn_pair.derived( 0.0), 0.5);
		assert_eq!(act_fn_pair.derived( 0.5), 0.62245935);
		assert_eq!(act_fn_pair.derived( 1.0), 0.7310586);
	}

	#[test]
	fn bent_identity_activation_fn() {
		let act_fn_pair = BaseDerivedActivationFn::<f32>::bent_identity();
		assert_eq!(act_fn_pair.base(-1.0), -0.79289323);
		assert_eq!(act_fn_pair.base(-0.5), -0.440983);
		assert_eq!(act_fn_pair.base( 0.0),  0.0);
		assert_eq!(act_fn_pair.base( 0.5),  0.559017);
		assert_eq!(act_fn_pair.base( 1.0),  1.2071068);
		assert_eq!(act_fn_pair.derived(-1.0), 0.6464466);
		assert_eq!(act_fn_pair.derived(-0.5), 0.7763932);
		assert_eq!(act_fn_pair.derived( 0.0), 1.0);
		assert_eq!(act_fn_pair.derived( 0.5), 1.2236068);
		assert_eq!(act_fn_pair.derived( 1.0), 1.3535534);
	}

	#[test]
	fn sinusoid_activation_fn() {
		let act_fn_pair = BaseDerivedActivationFn::<f32>::sinusoid();
		assert_eq!(act_fn_pair.base(-1.0), -0.84147096);
		assert_eq!(act_fn_pair.base(-0.5), -0.47942555);
		assert_eq!(act_fn_pair.base( 0.0), 0.0);
		assert_eq!(act_fn_pair.base( 0.5), 0.47942555);
		assert_eq!(act_fn_pair.base( 1.0), 0.84147096);
		assert_eq!(act_fn_pair.derived(-1.0), 0.5403023);
		assert_eq!(act_fn_pair.derived(-0.5), 0.87758255);
		assert_eq!(act_fn_pair.derived( 0.0), 1.0);
		assert_eq!(act_fn_pair.derived( 0.5), 0.87758255);
		assert_eq!(act_fn_pair.derived( 1.0), 0.5403023);
	}

	#[test]
	fn gaussian_activation_fn() {
		let act_fn_pair = BaseDerivedActivationFn::<f32>::gaussian();
		assert_eq!(act_fn_pair.base(-1.0), 0.36787945);
		assert_eq!(act_fn_pair.base(-0.5), 0.7788008);
		assert_eq!(act_fn_pair.base( 0.0), 1.0);
		assert_eq!(act_fn_pair.base( 0.5), 0.7788008);
		assert_eq!(act_fn_pair.base( 1.0), 0.36787945);
		assert_eq!(act_fn_pair.derived(-1.0), 0.7357589);
		assert_eq!(act_fn_pair.derived(-0.5), 0.7788008);
		assert_eq!(act_fn_pair.derived( 0.0), 0.0);
		assert_eq!(act_fn_pair.derived( 0.5), -0.7788008);
		assert_eq!(act_fn_pair.derived( 1.0), -0.7357589);
	}
}
