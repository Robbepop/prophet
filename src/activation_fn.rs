//! Provides utility functionality when working with common activation (or transfer) functions.

use num::{Float};
use std::fmt;
use std::ops::Deref;

macro_rules! fn_ptr_wrapper {
	($(#[$attr:meta])* struct $name:ident) => {
		$(#[$attr])*
		#[derive(Copy, Clone, Debug, PartialEq)]
		pub struct $name<T: Float>{
			fn_ptr: fn(T) -> T
		}

		impl<T: Float> From<fn(T) -> T> for $name<T> {
			fn from(fn_ptr: fn(T) -> T) -> Self {
				$name{
					fn_ptr: fn_ptr
				}
			}
		}

		impl<T: Float> Deref for $name<T> {
			type Target = fn(T) -> T;

			fn deref(&self) -> &Self::Target {
				&self.fn_ptr
			}
		}
	}
}

fn_ptr_wrapper!(
	/// Represents the base function part of an activation function pair.
	struct BaseFn
);
fn_ptr_wrapper!(
	/// Represents the derived function part of an activation function pair.
	struct DerivedFn
);

/// Represents the pair of an activation function and its derivate.
/// 
/// Has some convenience constructors to build some commonly used activation functions
/// with their respective derivate.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ActivationFn<F: Float> {
	/// the base function
	base: BaseFn<F>,
	/// the derivation of the base function
	derived: DerivedFn<F>,
	/// a string representation of the function
	repr: &'static str
}

/// Identity: *ƒ(x) = x*
fn identity_fn<F: Float>(x: F) -> F { x }
/// Derivation of the Identity: *ƒ(x) = 1*
fn identity_fn_dx<F: Float>(_: F) -> F { F::one() }

/// 
/// Binary Step:  
/// *ƒ(x) = 0* **if** *x < 0*  
/// *ƒ(x) = 1* **if** *x ≥ 0*
fn binary_step_fn<F: Float>(x: F) -> F {
	if x < F::zero() { F::zero() } else { F::one() }
}
/// Derivation of Binary Step: *ƒ(x) = 0, x ≠ 0*
fn binary_step_fn_dx<F: Float>(x: F) -> F {
	if x != F::zero() { F::zero() } else { F::infinity() }
}

/// Logistic or Sigmoid
fn logistic_fn<F: Float>(x: F) -> F {
	softplus_fn_dx(x)
}
/// Derivation of Logistic or Sigmoid
fn logistic_fn_dx<F: Float>(x: F) -> F {
	logistic_fn(x) * (F::one() - logistic_fn(x))
}

/// Tangens Hyperbolicus (**tanh**): *ƒ(x) = tanh(x)*
fn tanh_fn<F: Float>(x: F) -> F {
	x.tanh()
}
/// Derivation of Tangens Hyperbolicus (**tanh⁻¹**): *ƒ(x) = 1 - tanh²(x)*
fn tanh_fn_dx<F: Float>(x: F) -> F {
	let fx = tanh_fn(x);
	F::one() - fx*fx
}

/// Arcus Tangens (**atan**): *ƒ(x) = atan(x)*
fn arctan_fn<F: Float>(x: F) -> F {
	x.atan()
}
/// Derivation of Arcus Tangens (**atan⁻¹**): *ƒ(x) = (x² + 1)⁻¹*
fn arctan_fn_dx<F: Float>(x: F) -> F {
	F::one() / (x*x + F::one())
}

/// SoftSign: *ƒ(x) = x ⋅ (1 + |x|)⁻¹*
fn softsign_fn<F: Float>(x: F) -> F { x / (F::one() + x.abs()) }
/// Derivation of SoftSign: *ƒ(x) = ( (1 + |x|)² )⁻¹*
fn softsign_fn_dx<F: Float>(x: F) -> F { let dx = F::one() + x.abs(); F::one() / (dx*dx) }

/// ReLU:  
/// *ƒ(x) = 0* **if** *x < 0*  
/// *ƒ(x) = x* **else**
fn relu_fn<F: Float>(x: F) -> F { if x < F::zero() { F::zero() } else { x } }

/// Derivation of ReLU:  
/// *ƒ(x) = 0* **if** *x < 0*  
/// *ƒ(x) = 1* **else**
fn relu_fn_dx<F: Float>(x: F) -> F { if x < F::zero() { F::zero() } else { F::one() } }

/// SoftPlus: *ƒ(x) = __ln__(1 + eˣ)*
fn softplus_fn<F: Float>(x: F) -> F {
	x.exp().ln_1p()
}
/// Derivation of SoftPlus: *ƒ(x) = (1 + e⁻ˣ)⁻¹*
fn softplus_fn_dx<F: Float>(x: F) -> F {
	F::one() / (F::one() + (-x).exp())
}

/// Bent Identity: *ƒ(x) = ½(__sqrt__(x² + 1) - 1) + x*
fn bent_identity_fn<F: Float>(x: F) -> F {
	let two = F::from(2.0).unwrap();
	(((x*x) + F::one()).sqrt() - F::one()) / two + x
}
/// Derivation of Bent Identity: *ƒ(x) = x ⋅ (2 * __sqrt__(x² + 1))⁻¹ + 1*
fn bent_identity_fn_dx<F: Float>(x: F) -> F {
	let two = F::from(2.0).unwrap();
	x / (two * ((x * x) + F::one()).sqrt()) + F::one()
}

/// Sinusoid: *ƒ(x) = __sin__(x)*
fn sinusoid_fn<F: Float>(x: F) -> F {
	x.sin()
}
/// Derivation of Sinusoid: *ƒ(x) = __cos__(x)*
fn sinusoid_fn_dx<F: Float>(x: F) -> F {
	x.cos()
}

/// Gaussian:  *ƒ(x) = e⁻ˣˣ*
fn gaussian_fn<F: Float>(x: F) -> F {
	(-x * x).exp()
}
/// Derivation of Gaussian:  *ƒ(x) = -2xe⁻ˣˣ*
fn gaussian_fn_dx<F: Float>(x: F) -> F {
	let two = F::from(2.0).unwrap();
	-two * x * gaussian_fn(x)
}

impl<F: Float> ActivationFn<F> {
	/// Creates a new activation function with a base function and its derivation.
	/// 
	/// Similar to ```::custom``` but does not require a ```repr``` field.
	fn from_fn_ptr(base: fn(F) -> F, derived: fn(F) -> F, repr: &'static str) -> Self {
		ActivationFn{
			base: BaseFn{fn_ptr: base},
			derived: DerivedFn{fn_ptr: derived},
			repr: repr
		}
	}

	/// Used to create custom pairs of activation functions for users
	/// who wish to use an activation function that is not already covered by this library.
	pub fn custom(base: fn(F) -> F, derived: fn(F) -> F) -> Self {
		ActivationFn{
			base: BaseFn{fn_ptr: base},
			derived: DerivedFn{fn_ptr: derived},
			repr: "custom"
		}
	}

	/// Returns the base function.
	pub fn base_fn(&self) -> BaseFn<F> {
		self.base
	}

	/// Returns the derivation function.
	pub fn derived_fn(&self) -> DerivedFn<F> {
		self.derived
	}

	/// Forwards `x` to the base function and returns its result.
	pub fn base(&self, x: F) -> F {
		(self.base)(x)
	}

	/// Forwards `x` to the derivation of the base function and returns its result.
	pub fn derived(&self, x: F) -> F {
		(self.derived)(x)
	}

	/// Convenience constructor for the identity activation function.
	pub fn identity() -> Self {
		ActivationFn::from_fn_ptr(identity_fn, identity_fn_dx, "Identity")
	}

	/// Convenience constructor for the binary step activation function.
	pub fn binary_step() -> Self {
		ActivationFn::from_fn_ptr(binary_step_fn, binary_step_fn_dx, "Binary Step")
	}

	/// Convenience constructor for the arcus tangens activation function.
	pub fn arctan() -> Self {
		ActivationFn::from_fn_ptr(arctan_fn, arctan_fn_dx, "Arcus Tangens (arctan)")
	}

	/// Convenience constructor for the tangens hyperbolicus (tanh) activation function.
	pub fn tanh() -> Self {
		ActivationFn::from_fn_ptr(tanh_fn, tanh_fn_dx, "Tangens Hyperbolicus (tanh)")
	}

	/// Convenience constructor for the logistic or sigmoid activation function.
	pub fn logistic() -> Self {
		ActivationFn::from_fn_ptr(logistic_fn, logistic_fn_dx, "Logistic/Sigmoid")
	}

	/// Convenience constructor for the soft sign activation function.
	pub fn softsign() -> Self {
		ActivationFn::from_fn_ptr(softsign_fn, softsign_fn_dx, "SoftSign")
	}

	/// Convenience constructor for the ReLU activation function.
	pub fn relu() -> Self {
		ActivationFn::from_fn_ptr(relu_fn, relu_fn_dx, "ReLU")
	}

	/// Convenience constructor for the soft plus activation function.
	pub fn softplus() -> Self {
		ActivationFn::from_fn_ptr(softplus_fn, softplus_fn_dx, "SoftPlus")
	}

	/// Convenience constructor for the bent identity activation function.
	pub fn bent_identity() -> Self {
		ActivationFn::from_fn_ptr(bent_identity_fn, bent_identity_fn_dx, "Bent Identity")
	}

	/// Convenience constructor for the sinusoid activation function.
	pub fn sinusoid() -> Self {
		ActivationFn::from_fn_ptr(sinusoid_fn, sinusoid_fn_dx, "Sinusoid")
	}

	/// Convenience constructor for the gaussian activation function.
	pub fn gaussian() -> Self {
		ActivationFn::from_fn_ptr(gaussian_fn, gaussian_fn_dx, "Gaussian")
	}
}

impl<F: Float> fmt::Display for ActivationFn<F> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.repr)
    }
}

#[cfg(test)]
mod tests {
	use num::Float;
	use super::*;

	#[test]
	fn new_base_deriv_act_fn() {
		use super::{logistic_fn, logistic_fn_dx};
		let custom_sigmoid = ActivationFn::<f32>::custom(logistic_fn, logistic_fn_dx);
		let predef_sigmoid = ActivationFn::<f32>::logistic();
		assert_eq!(custom_sigmoid.base(-1.0), predef_sigmoid.base(-1.0));
		assert_eq!(custom_sigmoid.base(-0.5), predef_sigmoid.base(-0.5));
		assert_eq!(custom_sigmoid.base( 0.0), predef_sigmoid.base( 0.0));
		assert_eq!(custom_sigmoid.base( 0.5), predef_sigmoid.base( 0.5));
		assert_eq!(custom_sigmoid.base( 1.0), predef_sigmoid.base( 1.0));
	}

	#[test]
	fn identity_activation_fn() {
		let act_fn_pair = ActivationFn::<f32>::identity();
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
		let act_fn_pair = ActivationFn::<f32>::binary_step();
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
		let act_fn_pair = ActivationFn::<f32>::logistic();
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
		let act_fn_pair = ActivationFn::<f32>::arctan();
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
		let act_fn_pair = ActivationFn::<f32>::tanh();
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
		let act_fn_pair = ActivationFn::<f32>::softsign();
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
		let act_fn_pair = ActivationFn::<f32>::relu();
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
		let act_fn_pair = ActivationFn::<f32>::softplus();
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
		let act_fn_pair = ActivationFn::<f32>::bent_identity();
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
		let act_fn_pair = ActivationFn::<f32>::sinusoid();
		relative_eq!(act_fn_pair.base(-1.0), -0.84147096);
		relative_eq!(act_fn_pair.base(-0.5), -0.47942555);
		relative_eq!(act_fn_pair.base( 0.0), 0.0);
		relative_eq!(act_fn_pair.base( 0.5), 0.47942555);
		relative_eq!(act_fn_pair.base( 1.0), 0.84147096);
		relative_eq!(act_fn_pair.derived(-1.0), 0.5403023);
		relative_eq!(act_fn_pair.derived(-0.5), 0.87758255);
		relative_eq!(act_fn_pair.derived( 0.0), 1.0);
		relative_eq!(act_fn_pair.derived( 0.5), 0.87758255);
		relative_eq!(act_fn_pair.derived( 1.0), 0.5403023);
	}

	#[test]
	fn gaussian_activation_fn() {
		let act_fn_pair = ActivationFn::<f32>::gaussian();
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
