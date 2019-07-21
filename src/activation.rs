//! Provides utility functionality when working with common activation (or transfer) functions.

use ndarray::NdFloat;

/// Represents an activation function.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub enum Activation {
	/// Identity: *ƒ(x) = x*
	Identity,

	/// Binary Step:
	/// *ƒ(x) = 0* **if** *x < 0*
	/// *ƒ(x) = 1* **if** *x ≥ 0*
	BinaryStep,

	/// Logistic function
	Logistic,

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
			Identity => identity(x),
			BinaryStep => binary_step(x),
			Logistic => logistic(x),
			Tanh => tanh(x),
			ArcTan => arctan(x),
			SoftSign => softsign(x),
			ReLU => relu(x),
			SoftPlus => softplus(x),
			BentIdentity => bent_identity(x),
			Sinusoid => sinusoid(x),
			Gaussian => gaussian(x),
		}
	}

	/// Returns `dx(x)` with a given `x` and `dx` as the derived function.
	pub fn derived<F: NdFloat>(self, x: F) -> F {
		use self::details::*;
		use self::Activation::*;
		match self {
			Identity => identity_dx(x),
			BinaryStep => binary_step_dx(x),
			Logistic => logistic_dx(x),
			Tanh => tanh_dx(x),
			ArcTan => arctan_dx(x),
			SoftSign => softsign_dx(x),
			ReLU => relu_dx(x),
			SoftPlus => softplus_dx(x),
			BentIdentity => bent_identity_dx(x),
			Sinusoid => sinusoid_dx(x),
			Gaussian => gaussian_dx(x),
		}
	}
}

mod details {
	use ndarray::NdFloat;

	/// `Identity`: *ƒ(x) = x*
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
		if x < F::zero() {
			F::zero()
		} else {
			F::one()
		}
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
	pub fn logistic<F: NdFloat>(x: F) -> F {
		softplus_dx(x)
	}
	/// Derivation of Logistic or Sigmoid
	pub fn logistic_dx<F: NdFloat>(x: F) -> F {
		logistic(x) * (F::one() - logistic(x))
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

	/// `SoftSign`: *ƒ(x) = x ⋅ (1 + |x|)⁻¹*
	pub fn softsign<F: NdFloat>(x: F) -> F {
		x / (F::one() + x.abs())
	}
	/// Derivation of `SoftSign`: *ƒ(x) = ( (1 + |x|)² )⁻¹*
	pub fn softsign_dx<F: NdFloat>(x: F) -> F {
		let dx = F::one() + x.abs();
		F::one() / (dx * dx)
	}

	/// `ReLU`:
	/// *ƒ(x) = 0* **if** *x < 0*
	/// *ƒ(x) = x* **else**
	pub fn relu<F: NdFloat>(x: F) -> F {
		if x < F::zero() {
			F::zero()
		} else {
			x
		}
	}

	/// Derivation of `ReLU`:
	/// *ƒ(x) = 0* **if** *x < 0*
	/// *ƒ(x) = 1* **else**
	pub fn relu_dx<F: NdFloat>(x: F) -> F {
		if x < F::zero() {
			F::zero()
		} else {
			F::one()
		}
	}

	/// `SoftPlus`: *ƒ(x) = __ln__(1 + eˣ)*
	pub fn softplus<F: NdFloat>(x: F) -> F {
		x.exp().ln_1p()
	}
	/// Derivation of `SoftPlus`: *ƒ(x) = (1 + e⁻ˣ)⁻¹*
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

	const EPSILON: f64 = 0.0000001;

	#[test]
	fn identity() {
		let act = Activation::Identity;
		assert_relative_eq!(act.base(-1.0), -1.0);
		assert_relative_eq!(act.base(-0.5), -0.5);
		assert_relative_eq!(act.base(0.0), 0.0);
		assert_relative_eq!(act.base(0.5), 0.5);
		assert_relative_eq!(act.base(1.0), 1.0);
		assert_relative_eq!(act.derived(-1.0), 1.0);
		assert_relative_eq!(act.derived(-0.5), 1.0);
		assert_relative_eq!(act.derived(0.0), 1.0);
		assert_relative_eq!(act.derived(0.5), 1.0);
		assert_relative_eq!(act.derived(1.0), 1.0);
	}

	#[test]
	fn binary_step() {
		use num::Float;
		let act = Activation::BinaryStep;
		assert_relative_eq!(act.base(-1.0), 0.0);
		assert_relative_eq!(act.base(-0.5), 0.0);
		assert_relative_eq!(act.base(0.0), 1.0);
		assert_relative_eq!(act.base(0.5), 1.0);
		assert_relative_eq!(act.base(1.0), 1.0);
		assert_relative_eq!(act.derived(-1.0), 0.0);
		assert_relative_eq!(act.derived(-0.5), 0.0);
		assert_relative_eq!(act.derived(0.0), <f64>::infinity());
		assert_relative_eq!(act.derived(0.5), 0.0);
		assert_relative_eq!(act.derived(1.0), 0.0);
	}

	#[test]
	fn logistic() {
		let act = Activation::Logistic;
		assert_relative_eq!(act.base(-1.0), 0.26894143, epsilon = EPSILON);
		assert_relative_eq!(act.base(-0.5), 0.37754068, epsilon = EPSILON);
		assert_relative_eq!(act.base(0.0), 0.5);
		assert_relative_eq!(act.base(0.5), 0.62245935, epsilon = EPSILON);
		assert_relative_eq!(act.base(1.0), 0.7310586, epsilon = EPSILON);
		assert_relative_eq!(act.derived(-1.0), 0.19661194, epsilon = EPSILON);
		assert_relative_eq!(act.derived(-0.5), 0.23500371, epsilon = EPSILON);
		assert_relative_eq!(act.derived(0.0), 0.25);
		assert_relative_eq!(act.derived(0.5), 0.23500371, epsilon = EPSILON);
		assert_relative_eq!(act.derived(1.0), 0.19661193, epsilon = EPSILON);
	}

	#[test]
	fn arctan() {
		let act = Activation::ArcTan;
		assert_relative_eq!(act.base(-1.0), -0.7853982, epsilon = EPSILON);
		assert_relative_eq!(act.base(-0.5), -0.4636476, epsilon = EPSILON);
		assert_relative_eq!(act.base(0.0), 0.0);
		assert_relative_eq!(act.base(0.5), 0.4636476, epsilon = EPSILON);
		assert_relative_eq!(act.base(1.0), 0.7853982, epsilon = EPSILON);
		assert_relative_eq!(act.derived(-1.0), 0.5);
		assert_relative_eq!(act.derived(-0.5), 0.8);
		assert_relative_eq!(act.derived(0.0), 1.0);
		assert_relative_eq!(act.derived(0.5), 0.8);
		assert_relative_eq!(act.derived(1.0), 0.5);
	}

	#[test]
	fn tanh() {
		let act = Activation::Tanh;
		assert_relative_eq!(act.base(-1.0), -0.7615942, epsilon = EPSILON);
		assert_relative_eq!(act.base(-0.5), -0.46211717, epsilon = EPSILON);
		assert_relative_eq!(act.base(0.0), 0.0);
		assert_relative_eq!(act.base(0.5), 0.46211717, epsilon = EPSILON);
		assert_relative_eq!(act.base(1.0), 0.7615942, epsilon = EPSILON);
		assert_relative_eq!(act.derived(-1.0), 0.41997433, epsilon = EPSILON);
		assert_relative_eq!(act.derived(-0.5), 0.7864477, epsilon = EPSILON);
		assert_relative_eq!(act.derived(0.0), 1.0);
		assert_relative_eq!(act.derived(0.5), 0.7864477, epsilon = EPSILON);
		assert_relative_eq!(act.derived(1.0), 0.41997433, epsilon = EPSILON);
	}

	#[test]
	fn softsign() {
		let act = Activation::SoftSign;
		assert_relative_eq!(act.base(-1.0), -0.5);
		assert_relative_eq!(act.base(-0.5), -0.33333334, epsilon = EPSILON);
		assert_relative_eq!(act.base(0.0), 0.0);
		assert_relative_eq!(act.base(0.5), 0.33333334, epsilon = EPSILON);
		assert_relative_eq!(act.base(1.0), 0.5);
		assert_relative_eq!(act.derived(-1.0), 0.25);
		assert_relative_eq!(act.derived(-0.5), 0.44444445, epsilon = EPSILON);
		assert_relative_eq!(act.derived(0.0), 1.0);
		assert_relative_eq!(act.derived(0.5), 0.44444445, epsilon = EPSILON);
		assert_relative_eq!(act.derived(1.0), 0.25);
	}

	#[test]
	fn relu() {
		let act = Activation::ReLU;
		assert_relative_eq!(act.base(-1.0), 0.0);
		assert_relative_eq!(act.base(-0.5), 0.0);
		assert_relative_eq!(act.base(0.0), 0.0);
		assert_relative_eq!(act.base(0.5), 0.5);
		assert_relative_eq!(act.base(1.0), 1.0);
		assert_relative_eq!(act.derived(-1.0), 0.0);
		assert_relative_eq!(act.derived(-0.5), 0.0);
		assert_relative_eq!(act.derived(0.0), 1.0);
		assert_relative_eq!(act.derived(0.5), 1.0);
		assert_relative_eq!(act.derived(1.0), 1.0);
	}

	#[test]
	fn softplus() {
		let act = Activation::SoftPlus;
		assert_relative_eq!(act.base(-1.0), 0.3132617, epsilon = EPSILON);
		assert_relative_eq!(act.base(-0.5), 0.474077, epsilon = EPSILON);
		assert_relative_eq!(act.base(0.0), 0.6931472, epsilon = EPSILON);
		assert_relative_eq!(act.base(0.5), 0.974077, epsilon = EPSILON);
		assert_relative_eq!(act.base(1.0), 1.3132616, epsilon = EPSILON);
		assert_relative_eq!(act.derived(-1.0), 0.26894143, epsilon = EPSILON);
		assert_relative_eq!(act.derived(-0.5), 0.37754068, epsilon = EPSILON);
		assert_relative_eq!(act.derived(0.0), 0.5);
		assert_relative_eq!(act.derived(0.5), 0.62245935, epsilon = EPSILON);
		assert_relative_eq!(act.derived(1.0), 0.7310586, epsilon = EPSILON);
	}

	#[test]
	fn bent_identity() {
		let act = Activation::BentIdentity;
		assert_relative_eq!(act.base(-1.0), -0.79289323, epsilon = EPSILON);
		assert_relative_eq!(act.base(-0.5), -0.440983, epsilon = EPSILON);
		assert_relative_eq!(act.base(0.0), 0.0);
		assert_relative_eq!(act.base(0.5), 0.559017, epsilon = EPSILON);
		assert_relative_eq!(act.base(1.0), 1.2071068, epsilon = EPSILON);
		assert_relative_eq!(act.derived(-1.0), 0.6464466, epsilon = EPSILON);
		assert_relative_eq!(act.derived(-0.5), 0.7763932, epsilon = EPSILON);
		assert_relative_eq!(act.derived(0.0), 1.0);
		assert_relative_eq!(act.derived(0.5), 1.2236068, epsilon = EPSILON);
		assert_relative_eq!(act.derived(1.0), 1.3535534, epsilon = EPSILON);
	}

	#[test]
	fn sinusoid() {
		let act = Activation::Sinusoid;
		assert_relative_eq!(act.base(-1.0), -0.84147096, epsilon = EPSILON);
		assert_relative_eq!(act.base(-0.5), -0.47942555, epsilon = EPSILON);
		assert_relative_eq!(act.base(0.0), 0.0);
		assert_relative_eq!(act.base(0.5), 0.47942555, epsilon = EPSILON);
		assert_relative_eq!(act.base(1.0), 0.84147096, epsilon = EPSILON);
		assert_relative_eq!(act.derived(-1.0), 0.5403023, epsilon = EPSILON);
		assert_relative_eq!(act.derived(-0.5), 0.87758255, epsilon = EPSILON);
		assert_relative_eq!(act.derived(0.0), 1.0);
		assert_relative_eq!(act.derived(0.5), 0.87758255, epsilon = EPSILON);
		assert_relative_eq!(act.derived(1.0), 0.5403023, epsilon = EPSILON);
	}

	#[test]
	fn gaussian() {
		let act = Activation::Gaussian;
		assert_relative_eq!(act.base(-1.0), 0.36787945, epsilon = EPSILON);
		assert_relative_eq!(act.base(-0.5), 0.7788008, epsilon = EPSILON);
		assert_relative_eq!(act.base(0.0), 1.0);
		assert_relative_eq!(act.base(0.5), 0.7788008, epsilon = EPSILON);
		assert_relative_eq!(act.base(1.0), 0.36787945, epsilon = EPSILON);
		assert_relative_eq!(act.derived(-1.0), 0.7357589, epsilon = EPSILON);
		assert_relative_eq!(act.derived(-0.5), 0.7788008, epsilon = EPSILON);
		assert_relative_eq!(act.derived(0.0), 0.0);
		assert_relative_eq!(act.derived(0.5), -0.7788008, epsilon = EPSILON);
		assert_relative_eq!(act.derived(1.0), -0.7357589, epsilon = EPSILON);
	}
}
