//! Provides utility functionality when working with common activation (or transfer) functions.

use ndarray::NdFloat;

/// Represents an activation function.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Activation {
	/// Identity: *ƒ(x) = x*
	Identity,

	/// Binary Step:
	/// *ƒ(x) = 0* **if** *x < 0*
	/// *ƒ(x) = 1* **if** *x ≥ 0*
	BinaryStep,

	/// Sigmoid or Logistic function
	Sigmoid,

	/// Tangens Hyperbolicus (**tanh**): *ƒ(x) = tanh(x)*
	Tanh,

	/// Arcus Tangens (**atan**): *ƒ(x) = atan(x)*
	ArcTan,

	/// SoftSign: *ƒ(x) = x ⋅ (1 + |x|)⁻¹*
	SoftSign,

	/// ReLU:
	/// *ƒ(x) = 0* **if** *x < 0*
	/// *ƒ(x) = x* **else**
	ReLU,

	/// SoftPlus: *ƒ(x) = __ln__(1 + eˣ)*
	SoftPlus,

	/// Bent Identity: *ƒ(x) = ½(__sqrt__(x² + 1) - 1) + x*
	BentIdentity,

	/// Sinusoid: *ƒ(x) = __sin__(x)*
	Sinusoid,

	/// Gaussian:  *ƒ(x) = e⁻ˣˣ*
	Gaussian,
}

impl Activation {
	/// Returns `f(x)` with a given `x` and `f` as the base function.
	pub fn base<F: NdFloat>(self, x: F) -> F {
		use self::details::*;
		use self::Activation::*;
		match self {
			Identity     => identity(x),
			BinaryStep   => binary_step(x),
			Sigmoid      => sigmoid(x),
			Tanh         => tanh(x),
			ArcTan       => arctan(x),
			SoftSign     => softsign(x),
			ReLU         => relu(x),
			SoftPlus     => softplus(x),
			BentIdentity => bent_identity(x),
			Sinusoid     => sinusoid(x),
			Gaussian     => gaussian(x),
		}
	}

	/// Returns `dx(x)` with a given `x` and `dx` as the derived function.
	pub fn derived<F: NdFloat>(self, x: F) -> F {
		use self::details::*;
		use self::Activation::*;
		match self {
			Identity     => identity_dx(x),
			BinaryStep   => binary_step_dx(x),
			Sigmoid      => sigmoid_dx(x),
			Tanh         => tanh_dx(x),
			ArcTan       => arctan_dx(x),
			SoftSign     => softsign_dx(x),
			ReLU         => relu_dx(x),
			SoftPlus     => softplus_dx(x),
			BentIdentity => bent_identity_dx(x),
			Sinusoid     => sinusoid_dx(x),
			Gaussian     => gaussian_dx(x),
		}
	}
}

mod details {
	use ndarray::NdFloat;

	/// Identity: *ƒ(x) = x*
	pub fn identity<F: NdFloat>(x: F) -> F {
		x
	}
	/// Derivation of the Identity: *ƒ(x) = 1*
	pub fn identity_dx<F: NdFloat>(_: F) -> F {
		F::one()
	}

	/// Binary Step:
	/// *ƒ(x) = 0* **if** *x < 0*
	/// *ƒ(x) = 1* **if** *x ≥ 0*
	pub fn binary_step<F: NdFloat>(x: F) -> F {
		if x < F::zero() { F::zero() } else { F::one() }
	}
	/// Derivation of Binary Step: *ƒ(x) = 0, x ≠ 0*
	pub fn binary_step_dx<F: NdFloat>(x: F) -> F {
		if x != F::zero() {
			F::zero()
		} else {
			F::infinity()
		}
	}

	/// Logistic or Sigmoid
	pub fn sigmoid<F: NdFloat>(x: F) -> F {
		softplus_dx(x)
	}
	/// Derivation of Logistic or Sigmoid
	pub fn sigmoid_dx<F: NdFloat>(x: F) -> F {
		sigmoid(x) * (F::one() - sigmoid(x))
	}

	/// Tangens Hyperbolicus (**tanh**): *ƒ(x) = tanh(x)*
	pub fn tanh<F: NdFloat>(x: F) -> F {
		x.tanh()
	}
	/// Derivation of Tangens Hyperbolicus (**tanh⁻¹**): *ƒ(x) = 1 - tanh²(x)*
	pub fn tanh_dx<F: NdFloat>(x: F) -> F {
		let fx = tanh(x);
		F::one() - fx * fx
	}

	/// Arcus Tangens (**atan**): *ƒ(x) = atan(x)*
	pub fn arctan<F: NdFloat>(x: F) -> F {
		x.atan()
	}
	/// Derivation of Arcus Tangens (**atan⁻¹**): *ƒ(x) = (x² + 1)⁻¹*
	pub fn arctan_dx<F: NdFloat>(x: F) -> F {
		F::one() / (x * x + F::one())
	}

	/// SoftSign: *ƒ(x) = x ⋅ (1 + |x|)⁻¹*
	pub fn softsign<F: NdFloat>(x: F) -> F {
		x / (F::one() + x.abs())
	}
	/// Derivation of SoftSign: *ƒ(x) = ( (1 + |x|)² )⁻¹*
	pub fn softsign_dx<F: NdFloat>(x: F) -> F {
		let dx = F::one() + x.abs();
		F::one() / (dx * dx)
	}

	/// ReLU:
	/// *ƒ(x) = 0* **if** *x < 0*
	/// *ƒ(x) = x* **else**
	pub fn relu<F: NdFloat>(x: F) -> F {
		if x < F::zero() { F::zero() } else { x }
	}

	/// Derivation of ReLU:
	/// *ƒ(x) = 0* **if** *x < 0*
	/// *ƒ(x) = 1* **else**
	pub fn relu_dx<F: NdFloat>(x: F) -> F {
		if x < F::zero() { F::zero() } else { F::one() }
	}

	/// SoftPlus: *ƒ(x) = __ln__(1 + eˣ)*
	pub fn softplus<F: NdFloat>(x: F) -> F {
		x.exp().ln_1p()
	}
	/// Derivation of SoftPlus: *ƒ(x) = (1 + e⁻ˣ)⁻¹*
	pub fn softplus_dx<F: NdFloat>(x: F) -> F {
		F::one() / (F::one() + (-x).exp())
	}

	/// Bent Identity: *ƒ(x) = ½(__sqrt__(x² + 1) - 1) + x*
	pub fn bent_identity<F: NdFloat>(x: F) -> F {
		let two = F::from(2.0).unwrap();
		(((x * x) + F::one()).sqrt() - F::one()) / two + x
	}
	/// Derivation of Bent Identity: *ƒ(x) = x ⋅ (2 * __sqrt__(x² + 1))⁻¹ + 1*
	pub fn bent_identity_dx<F: NdFloat>(x: F) -> F {
		let two = F::from(2.0).unwrap();
		x / (two * ((x * x) + F::one()).sqrt()) + F::one()
	}

	/// Sinusoid: *ƒ(x) = __sin__(x)*
	pub fn sinusoid<F: NdFloat>(x: F) -> F {
		x.sin()
	}
	/// Derivation of Sinusoid: *ƒ(x) = __cos__(x)*
	pub fn sinusoid_dx<F: NdFloat>(x: F) -> F {
		x.cos()
	}

	/// Gaussian:  *ƒ(x) = e⁻ˣˣ*
	pub fn gaussian<F: NdFloat>(x: F) -> F {
		(-x * x).exp()
	}
	/// Derivation of Gaussian:  *ƒ(x) = -2xe⁻ˣˣ*
	pub fn gaussian_dx<F: NdFloat>(x: F) -> F {
		let two = F::from(2.0).unwrap();
		-two * x * gaussian(x)
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn identity() {
		let act = Activation::Identity;
		relative_eq!(act.base(-1.0), -1.0);
		relative_eq!(act.base(-0.5), -0.5);
		relative_eq!(act.base(0.0), 0.0);
		relative_eq!(act.base(0.5), 0.5);
		relative_eq!(act.base(1.0), 1.0);
		relative_eq!(act.derived(-1.0), 1.0);
		relative_eq!(act.derived(-0.5), 1.0);
		relative_eq!(act.derived(0.0), 1.0);
		relative_eq!(act.derived(0.5), 1.0);
		relative_eq!(act.derived(1.0), 1.0);
	}

	#[test]
	fn binary_step() {
		use num::Float;
		let act = Activation::BinaryStep;
		relative_eq!(act.base(-1.0), 0.0);
		relative_eq!(act.base(-0.5), 0.0);
		relative_eq!(act.base(0.0), 1.0);
		relative_eq!(act.base(0.5), 1.0);
		relative_eq!(act.base(1.0), 1.0);
		relative_eq!(act.derived(-1.0), 0.0);
		relative_eq!(act.derived(-0.5), 0.0);
		relative_eq!(act.derived(0.0), <f64>::infinity());
		relative_eq!(act.derived(0.5), 0.0);
		relative_eq!(act.derived(1.0), 0.0);
	}

	#[test]
	fn logistic() {
		let act = Activation::Sigmoid;
		relative_eq!(act.base(-1.0), 0.26894143);
		relative_eq!(act.base(-0.5), 0.37754068);
		relative_eq!(act.base(0.0), 0.5);
		relative_eq!(act.base(0.5), 0.62245935);
		relative_eq!(act.base(1.0), 0.7310586);
		relative_eq!(act.derived(-1.0), 0.19661194);
		relative_eq!(act.derived(-0.5), 0.23500371);
		relative_eq!(act.derived(0.0), 0.25);
		relative_eq!(act.derived(0.5), 0.23500371);
		relative_eq!(act.derived(1.0), 0.19661193);
	}

	#[test]
	fn arctan() {
		let act = Activation::ArcTan;
		relative_eq!(act.base(-1.0), -0.7853982);
		relative_eq!(act.base(-0.5), -0.4636476);
		relative_eq!(act.base(0.0), 0.0);
		relative_eq!(act.base(0.5), 0.4636476);
		relative_eq!(act.base(1.0), 0.7853982);
		relative_eq!(act.derived(-1.0), 0.5);
		relative_eq!(act.derived(-0.5), 0.8);
		relative_eq!(act.derived(0.0), 1.0);
		relative_eq!(act.derived(0.5), 0.8);
		relative_eq!(act.derived(1.0), 0.5);
	}

	#[test]
	fn tanh() {
		let act = Activation::Tanh;
		relative_eq!(act.base(-1.0), -0.7615942);
		relative_eq!(act.base(-0.5), -0.46211717);
		relative_eq!(act.base(0.0), 0.0);
		relative_eq!(act.base(0.5), 0.46211717);
		relative_eq!(act.base(1.0), 0.7615942);
		relative_eq!(act.derived(-1.0), 0.41997433);
		relative_eq!(act.derived(-0.5), 0.7864477);
		relative_eq!(act.derived(0.0), 1.0);
		relative_eq!(act.derived(0.5), 0.7864477);
		relative_eq!(act.derived(1.0), 0.41997433);
	}

	#[test]
	fn softsign() {
		let act = Activation::SoftSign;
		relative_eq!(act.base(-1.0), -0.5);
		relative_eq!(act.base(-0.5), -0.33333334);
		relative_eq!(act.base(0.0), 0.0);
		relative_eq!(act.base(0.5), 0.33333334);
		relative_eq!(act.base(1.0), 0.5);
		relative_eq!(act.derived(-1.0), 0.25);
		relative_eq!(act.derived(-0.5), 0.44444445);
		relative_eq!(act.derived(0.0), 1.0);
		relative_eq!(act.derived(0.5), 0.44444445);
		relative_eq!(act.derived(1.0), 0.25);
	}

	#[test]
	fn relu() {
		let act = Activation::ReLU;
		relative_eq!(act.base(-1.0), 0.0);
		relative_eq!(act.base(-0.5), 0.0);
		relative_eq!(act.base(0.0), 0.0);
		relative_eq!(act.base(0.5), 0.5);
		relative_eq!(act.base(1.0), 1.0);
		relative_eq!(act.derived(-1.0), 0.0);
		relative_eq!(act.derived(-0.5), 0.0);
		relative_eq!(act.derived(0.0), 1.0);
		relative_eq!(act.derived(0.5), 1.0);
		relative_eq!(act.derived(1.0), 1.0);
	}

	#[test]
	fn softplus() {
		let act = Activation::SoftPlus;
		relative_eq!(act.base(-1.0), 0.3132617);
		relative_eq!(act.base(-0.5), 0.474077);
		relative_eq!(act.base(0.0), 0.6931472);
		relative_eq!(act.base(0.5), 0.974077);
		relative_eq!(act.base(1.0), 1.3132616);
		relative_eq!(act.derived(-1.0), 0.26894143);
		relative_eq!(act.derived(-0.5), 0.37754068);
		relative_eq!(act.derived(0.0), 0.5);
		relative_eq!(act.derived(0.5), 0.62245935);
		relative_eq!(act.derived(1.0), 0.7310586);
	}

	#[test]
	fn bent_identity() {
		let act = Activation::BentIdentity;
		relative_eq!(act.base(-1.0), -0.79289323);
		relative_eq!(act.base(-0.5), -0.440983);
		relative_eq!(act.base(0.0), 0.0);
		relative_eq!(act.base(0.5), 0.559017);
		relative_eq!(act.base(1.0), 1.2071068);
		relative_eq!(act.derived(-1.0), 0.6464466);
		relative_eq!(act.derived(-0.5), 0.7763932);
		relative_eq!(act.derived(0.0), 1.0);
		relative_eq!(act.derived(0.5), 1.2236068);
		relative_eq!(act.derived(1.0), 1.3535534);
	}

	#[test]
	fn sinusoid() {
		let act = Activation::Sinusoid;
		relative_eq!(act.base(-1.0), -0.84147096);
		relative_eq!(act.base(-0.5), -0.47942555);
		relative_eq!(act.base(0.0), 0.0);
		relative_eq!(act.base(0.5), 0.47942555);
		relative_eq!(act.base(1.0), 0.84147096);
		relative_eq!(act.derived(-1.0), 0.5403023);
		relative_eq!(act.derived(-0.5), 0.87758255);
		relative_eq!(act.derived(0.0), 1.0);
		relative_eq!(act.derived(0.5), 0.87758255);
		relative_eq!(act.derived(1.0), 0.5403023);
	}

	#[test]
	fn gaussian() {
		let act = Activation::Gaussian;
		relative_eq!(act.base(-1.0), 0.36787945);
		relative_eq!(act.base(-0.5), 0.7788008);
		relative_eq!(act.base(0.0), 1.0);
		relative_eq!(act.base(0.5), 0.7788008);
		relative_eq!(act.base(1.0), 0.36787945);
		relative_eq!(act.derived(-1.0), 0.7357589);
		relative_eq!(act.derived(-0.5), 0.7788008);
		relative_eq!(act.derived(0.0), 0.0);
		relative_eq!(act.derived(0.5), -0.7788008);
		relative_eq!(act.derived(1.0), -0.7357589);
	}
}
