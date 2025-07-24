use std::{fmt, ops::{Add, Div, Index, Mul, Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, Sub}};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vector<const N: usize> {
    components: [f64; N],
}

impl<const N: usize> fmt::Display for Vector<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Start with an opening bracket
        write!(f, "[")?;
        
        // Iterate through components, formatting each one
        for (i, component) in self.components.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;  // Add comma separator between elements
            }
            write!(f, "{:.6}", component)?;  // Format with 6 decimal places
        }
        
        // Close with a bracket
        write!(f, "]")
    }
}

// ===== 2D Vector Methods =====
impl Vector<2> {
    #[inline]
    pub fn x(&self) -> f64 {
        self.components[0] // Returns the first component (x-coordinate).
    }

    #[inline]
    pub fn y(&self) -> f64 {
        self.components[1] // Returns the second component (y-coordinate).
    }

    /// Returns the 2D cross product (scalar/pseudo-scalar).
    /// For 2D vectors, the cross product is a scalar value representing the signed area of the parallelogram formed by the two vectors.
    #[inline]
    pub fn cross(&self, other: &Self) -> f64 {
        self.x() * other.y() - self.y() * other.x()
    }
}
// Example Usage:
// Input:
// let v1 = Vector::new([1.0, 2.0]);
// let v2 = Vector::new([3.0, 4.0]);
// Output:
// v1.x(): 1.0
// v1.y(): 2.0
// v1.cross(&v2): -2.0 (calculated as (1.0 * 4.0) - (2.0 * 3.0) = 4.0 - 6.0 = -2.0)

// ===== 3D Vector Methods =====
impl Vector<3> {
    #[inline]
    pub fn x(&self) -> f64 {
        self.components[0] // Returns the first component (x-coordinate).
    }
    #[inline]
    pub fn y(&self) -> f64 {
        self.components[1] // Returns the second component (y-coordinate).
    }
    #[inline]
    pub fn z(&self) -> f64 {
        self.components[2] // Returns the third component (z-coordinate).
    }

    /// Returns the 3D cross product (vector).
    /// The 3D cross product results in a new vector that is perpendicular to both input vectors.
    #[inline]
    pub fn cross(&self, other: &Self) -> Self {
        Self::new([
            self.y() * other.z() - self.z() * other.y(),
            self.z() * other.x() - self.x() * other.z(),
            self.x() * other.y() - self.y() * other.x(),
        ])
    }
}
// Example Usage:
// Input:
// let v1 = Vector::new([1.0, 0.0, 0.0]); // i-hat
// let v2 = Vector::new([0.0, 1.0, 0.0]); // j-hat
// Output:
// v1.x(): 1.0
// v1.y(): 0.0
// v1.z(): 0.0
// v1.cross(&v2): Vector { components: [0.0, 0.0, 1.0] } (k-hat)

// ===== Generic Vector Methods (N-dimensional) =====
impl<const N: usize> Vector<N> {
    /// Creates a new vector from an array of components.
    #[inline]
    pub fn new(components: [f64; N]) -> Self {
        Self { components }
    }

    /// Creates a new vector with all components set to 0.0.
    #[inline]
    pub fn zeros() -> Self {
        Self {
            components: [0.0; N],
        }
    }

     /// Creates a new vector with all components set to 1.0.
    #[inline]
    pub fn ones() -> Self {
        Self {
            components: [1.0; N],
        }
    }

    #[inline]
    pub fn apply<F>(&self, fun: F) -> Self
    where
        F: Fn(f64) -> f64, // 'F' must be a function/closure that takes an f64 and returns an f64
    {
        Vector {
            components: std::array::from_fn(|i| fun(self.components[i])),
        }
    }

    /// Returns the component at a specific index.
    #[inline]
    pub fn component(&self, index: usize) -> f64 {
        self.components[index]
    }

    /// Calculates the magnitude (length) of the vector.
    /// Uses the Euclidean norm: sqrt(sum(x_i^2)).
    #[inline]
    pub fn magnitude(&self) -> f64 {
        self.components
            .iter()
            .fold(0.0, |acc, x| acc + x * x)
            .sqrt()
    }

    /// Calculates the magnitude (length) of the vector.
    /// Uses the Euclidean norm: sqrt(sum(x_i^2)).
    #[inline]
    pub fn normalize(&self) -> Self {
        self.divide_vector(self.magnitude())
    }

    // Existing methods that will now be replaced by operator overloads or used internally
    /// Adds two vectors component-wise.
    #[inline]
    pub fn add_vectors(self, other: Self) -> Self {
        Vector {
            components: std::array::from_fn(|i| self.components[i] + other.components[i]),
        }
    }

    /// Subtracts one vector from another component-wise.
    #[inline]
    pub fn subtract_vectors(self, other: Self) -> Self {
        Vector {
            components: std::array::from_fn(|i| self.components[i] - other.components[i]),
        }
    }

    /// Performs a Hadamard (element-wise) product of two vectors.
    #[inline]
    pub fn hadamard_product_vectors(self, other: Self) -> Self {
        Vector {
            components: std::array::from_fn(|i| self.components[i] * other.components[i]),
        }
    }

    /// Performs a Hadamard (element-wise) division of two vectors.
    #[inline]
    pub fn hadamard_division_vectors(self, other: Self) -> Self {
        Vector {
            components: std::array::from_fn(|i| self.components[i] / other.components[i]),
        }
    }

    /// Scales a vector by a scalar value (multiplies each component by the scalar).
    #[inline]
    pub fn scale_vector(self, scalar: f64) -> Self {
        Vector {
            components: std::array::from_fn(|i| self.components[i] * scalar),
        }
    }

    /// Divides a vector by a scalar value (divides each component by the scalar).
    #[inline]
    pub fn divide_vector(self, scalar: f64) -> Self {
        Vector {
            components: std::array::from_fn(|i| self.components[i] / scalar),
        }
    }

    /// Calculates the dot product (scalar product) of two vectors.
    /// Sum of the products of corresponding components
    #[inline]
    pub fn dot(&self, other: &Self) -> f64 {
        self.components
            .iter()
            .zip(other.components.iter())
            .map(|(a, b)| a * b)
            .sum()
    }
}
// Example Usage:
// Input:
// let v = Vector::new([3.0, 4.0, 0.0]);
// let zero_vec = Vector::<3>::zero();
// let v_add = Vector::new([1.0, 2.0, 3.0]);
// let v_sub = Vector::new([4.0, 2.0, 1.0]);
// let v_had_prod_v1 = Vector::new([2.0, 3.0, 4.0]);
// let v_had_prod_v2 = Vector::new([1.0, 2.0, 0.5]);
// let v_scale = Vector::new([1.0, 2.0, 3.0]);
// let v_dot1 = Vector::new([1.0, 2.0, 3.0]);
// let v_dot2 = Vector::new([4.0, 5.0, 6.0]);
// Output:
// Vector::new([1.0, 2.0, 3.0]): Vector { components: [1.0, 2.0, 3.0] }
// Vector::<3>::zero(): Vector { components: [0.0, 0.0, 0.0] }
// v.component(0): 3.0
// v.magnitude(): 5.0 (sqrt(3*3 + 4*4 + 0*0) = sqrt(9 + 16) = sqrt(25) = 5.0)
// v_add.add_vectors(Vector::new([1.0, 1.0, 1.0])): Vector { components: [2.0, 3.0, 4.0] }
// v_sub.subtract_vectors(Vector::new([1.0, 1.0, 1.0])): Vector { components: [3.0, 1.0, 0.0] }
// v_had_prod_v1.hadamard_product_vectors(v_had_prod_v2): Vector { components: [2.0, 6.0, 2.0] }
// v_scale.scale_vector(2.0): Vector { components: [2.0, 4.0, 6.0] }
// v_scale.divide_vector(2.0): Vector { components: [0.5, 1.0, 1.5] }
// v_dot1.dot(&v_dot2): 32.0 (1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32)

impl<const N: usize> Add for Vector<N> {
    type Output = Self;

    #[inline]
    fn add(self, other: Self) -> Self::Output {
        self.add_vectors(other)
    }
}
// Example Usage:
// Input:
// let v1 = Vector::new([1.0, 2.0, 3.0]);
// let v2 = Vector::new([4.0, 5.0, 6.0]);
// Output:
// v1 + v2: Vector { components: [5.0, 7.0, 9.0] }

impl<const N: usize> Sub for Vector<N> {
    type Output = Self;

    #[inline]
    fn sub(self, other: Self) -> Self::Output {
        self.subtract_vectors(other)
    }
}
// Example Usage:
// Input:
// let v1 = Vector::new([5.0, 7.0, 9.0]);
// let v2 = Vector::new([4.0, 5.0, 6.0]);
// Output:
// v1 - v2: Vector { components: [1.0, 2.0, 3.0] }

impl<const N: usize> Mul for Vector<N> {
    type Output = Self;

    #[inline]
    fn mul(self, other: Self) -> Self::Output {
        self.hadamard_product_vectors(other)
    }
}
// Example Usage:
// Input:
// let v1 = Vector::new([1.0, 2.0, 3.0]);
// let v2 = Vector::new([4.0, 5.0, 6.0]);
// Output:
// v1 * v2: Vector { components: [4.0, 10.0, 18.0] } // Element-wise multiplication

impl<const N: usize> Mul<f64> for Vector<N> {
    type Output = Self;

    #[inline]
    fn mul(self, scalar: f64) -> Self::Output {
        self.scale_vector(scalar)
    }
}
// Example Usage:
// Input:
// let v = Vector::new([1.0, 2.0, 3.0]);
// Output:
// v * 2.0: Vector { components: [2.0, 4.0, 6.0] }

impl<const N: usize> Div for Vector<N> {
    type Output = Self;

    #[inline]
    fn div(self, other: Self) -> Self::Output {
        self.hadamard_division_vectors(other)
    }
}
// Example Usage:
// Input:
// let v1 = Vector::new([4.0, 10.0, 18.0]);
// let v2 = Vector::new([1.0, 2.0, 3.0]);
// Output:
// v1 / v2: Vector { components: [4.0, 5.0, 6.0] } // Element-wise division

impl<const N: usize> Div<f64> for Vector<N> {
    type Output = Self;

    #[inline]
    fn div(self, scalar: f64) -> Self::Output {
        self.divide_vector(scalar)
    }
}
// Example Usage:
// Input:
// let v = Vector::new([2.0, 4.0, 6.0]);
// Output:
// v / 2.0: Vector { components: [1.0, 2.0, 3.0] }

impl<const N: usize> Index<usize> for Vector<N> {
    type Output = f64;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.components[index]
    }
}

// --- Indexing for ranges (returns a slice & [f64]) ---
impl<const N: usize> Index<Range<usize>> for Vector<N> {
    type Output = [f64];
    #[inline]
    fn index(&self, index: Range<usize>) -> &Self::Output {
        &self.components[index]
    }
}
// Example Usage:
// Input:
// let v = Vector::new([10.0, 20.0, 30.0]);
// Output:
// v[0]: 10.0
// v[1]: 20.0

impl<const N: usize> Index<RangeTo<usize>> for Vector<N> {
    type Output = [f64];
    #[inline]
    fn index(&self, index: RangeTo<usize>) -> &Self::Output {
        &self.components[index]
    }
}

impl<const N: usize> Index<RangeFrom<usize>> for Vector<N> {
    type Output = [f64];
    #[inline]
    fn index(&self, index: RangeFrom<usize>) -> &Self::Output {
        &self.components[index]
    }
}

impl<const N: usize> Index<RangeFull> for Vector<N> {
    type Output = [f64];
    #[inline]
    fn index(&self, index: RangeFull) -> &Self::Output {
        &self.components[index]
    }
}

impl<const N: usize> Index<RangeInclusive<usize>> for Vector<N> {
    type Output = [f64];
    #[inline]
    fn index(&self, index: RangeInclusive<usize>) -> &Self::Output {
        &self.components[index]
    }
}
// Example Usage:
// Input:
// let v = Vector::new([1.0, 2.0, 3.0, 4.0, 5.0]);
// Output:
// v[1..4]: [2.0, 3.0, 4.0]
// v[..3]: [1.0, 2.0, 3.0]
// v[2..]: [3.0, 4.0, 5.0]
// v[..]: [1.0, 2.0, 3.0, 4.0, 5.0]
// v[1..=3]: [2.0, 3.0, 4.0]

impl<const N: usize> From<&[f64]> for Vector<N> {
    /// Converts a slice of f64 into a Vector.
    /// Panics if the slice length does not match the vector's dimension N.
    fn from(slice: &[f64]) -> Self {
        if slice.len() != N {
            panic!(
                "Cannot create Vector<{}> from slice of length {}",
                N,
                slice.len()
            );
        }
        let mut components = [0.0; N];
        components.copy_from_slice(slice);
        Vector { components }
    }
}

impl<const N: usize> From<&[f64; N]> for Vector<N> {
    fn from(arr: &[f64; N]) -> Self {
        Vector { components: *arr }
    }
}
// Example Usage:
// Input:
// let slice_data = [10.0, 20.0, 30.0];
// let v_from_slice = Vector::<3>::from(&slice_data);
// Output:
// v_from_slice: Vector { components: [10.0, 20.0, 30.0] }

// Input (will panic):
// let wrong_slice = [1.0, 2.0];
// let v_wrong = Vector::<3>::from(&wrong_slice);
// Output:
// Panics with "Cannot create Vector<3> from slice of length 2"