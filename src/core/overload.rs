use std::ops::{Add, Div, Mul, Sub, AddAssign, SubAssign, MulAssign, DivAssign, Neg, Index, IndexMut};
use crate::core::Matrix;

// ===== Addition Operators =====

/// Implements matrix addition (`Matrix + Matrix`)
///
/// # Examples
/// ```
/// use nelab::Matrix;
/// let a = Matrix::ones(2, 2);
/// let b = Matrix::identity(2);
/// let sum = &a + &b;
/// ```
///
/// # Errors
/// Returns `Err` if matrices have different dimensions
impl Add<&Matrix> for &Matrix {
    type Output = Result<Matrix, String>;

    fn add(self, rhs: &Matrix) -> Self::Output {
        self.addition(rhs)
    }
}

/// Implements compound matrix addition assignment (`Matrix += Matrix`)
///
/// # Panics
/// Panics if matrices have different dimensions
///
/// # Examples
/// ```
/// use nelab::Matrix;
/// let mut a = Matrix::ones(2, 2);
/// let b = Matrix::identity(2);
/// a += &b;  // a now contains [[2, 1], [1, 2]]
/// ```
impl AddAssign<&Matrix> for Matrix {
    fn add_assign(&mut self, rhs: &Matrix) {
        *self = (self as &Matrix + rhs).unwrap();
    }
}

/// Implements matrix-scalar addition (`Matrix + f64`)
///
/// # Examples
/// ```
/// use nelab::Matrix;
/// let m = Matrix::ones(2, 2);
/// let result = &m + 3.0;  // [[4, 4], [4, 4]]
/// ```
impl Add<f64> for &Matrix {
    type Output = Matrix;

    fn add(self, rhs: f64) -> Self::Output {
        self.scalar_addition(rhs)
    }
}

/// Implements scalar-matrix addition (`f64 + Matrix`)
///
/// # Examples
/// ```
/// use nelab::Matrix;
/// let m = Matrix::ones(2, 2);
/// let result = 3.0 + &m;  // [[4, 4], [4, 4]]
/// ```
impl Add<&Matrix> for f64 {
    type Output = Matrix;

    fn add(self, rhs: &Matrix) -> Self::Output {
        rhs.scalar_addition(self)
    }
}

// ===== Subtraction Operators =====

/// Implements matrix subtraction (`Matrix - Matrix`)
///
/// # Examples
/// ```
/// use nelab::Matrix;
/// let a = Matrix::ones(2, 2);
/// let b = Matrix::identity(2);
/// let diff = &a - &b;  // [[0, 1], [1, 0]]
/// ```
///
/// # Errors
/// Returns `Err` if matrices have different dimensions
impl Sub<&Matrix> for &Matrix {
    type Output = Result<Matrix, String>;

    fn sub(self, rhs: &Matrix) -> Self::Output {
        self.subtraction(rhs)
    }
}

/// Implements compound matrix subtraction assignment (`Matrix -= Matrix`)
///
/// # Panics
/// Panics if matrices have different dimensions
///
/// # Examples
/// ```
/// use nelab::Matrix;
/// let mut a = Matrix::ones(2, 2);
/// let b = Matrix::identity(2);
/// a -= &b;  // a now contains [[0, 1], [1, 0]]
/// ```
impl SubAssign<&Matrix> for Matrix {
    fn sub_assign(&mut self, rhs: &Matrix) {
        *self = (self as &Matrix - rhs).unwrap();
    }
}

/// Implements matrix-scalar subtraction (`Matrix - f64`)
///
/// # Examples
/// ```
/// use nelab::Matrix;
/// let m = Matrix::ones(2, 2);
/// let result = &m - 1.0;  // [[0, 0], [0, 0]]
/// ```
impl Sub<f64> for &Matrix {
    type Output = Matrix;

    fn sub(self, rhs: f64) -> Self::Output {
        self.scalar_subtraction(rhs)
    }
}

/// Implements scalar-matrix subtraction (`f64 - Matrix`)
///
/// # Examples
/// ```
/// use nelab::Matrix;
/// let m = Matrix::ones(2, 2);
/// let result = 3.0 - &m;  // [[2, 2], [2, 2]]
/// ```
impl Sub<&Matrix> for f64 {
    type Output = Matrix;

    fn sub(self, rhs: &Matrix) -> Self::Output {
        rhs.scalar_operation(self, |a, b| b - a)
    }
}

// ===== Multiplication Operators =====

/// Implements Hadamard (element-wise) product (`Matrix * Matrix`)
///
/// # Examples
/// ```
/// use nelab::Matrix;
/// let a = Matrix::diagonal(&[1.0, 2.0]);
/// let b = Matrix::ones(2, 2);
/// let product = &a * &b;  // [[1, 0], [0, 2]]
/// ```
///
/// # Errors
/// Returns `Err` if matrices have different dimensions
impl Mul<&Matrix> for &Matrix {
    type Output = Result<Matrix, String>;

    fn mul(self, rhs: &Matrix) -> Self::Output {
        self.hadamard_multiplication(rhs)
    }
}

/// Implements matrix-scalar multiplication (`Matrix * f64`)
///
/// # Examples
/// ```
/// use nelab::Matrix;
/// let m = Matrix::diagonal(&[1.0, 2.0]);
/// let result = &m * 3.0;  // [[3, 0], [0, 6]]
/// ```
impl Mul<f64> for &Matrix {
    type Output = Matrix;

    fn mul(self, rhs: f64) -> Self::Output {
        self.scalar_multiplication(rhs)
    }
}

/// Implements scalar-matrix multiplication (`f64 * Matrix`)
///
/// # Examples
/// ```
/// use nelab::Matrix;
/// let m = Matrix::diagonal(&[1.0, 2.0]);
/// let result = 3.0 * &m;  // [[3, 0], [0, 6]]
/// ```
impl Mul<&Matrix> for f64 {
    type Output = Matrix;

    fn mul(self, rhs: &Matrix) -> Self::Output {
        rhs.scalar_multiplication(self)
    }
}

/// Implements compound scalar multiplication assignment (`Matrix *= f64`)
///
/// # Examples
/// ```
/// use nelab::Matrix;
/// let mut m = Matrix::diagonal(&[1.0, 2.0]);
/// m *= 3.0;  // m now contains [[3, 0], [0, 6]]
/// ```
impl MulAssign<f64> for Matrix {
    fn mul_assign(&mut self, rhs: f64) {
        *self = (self as &Matrix) * rhs;
    }
}

// ===== Division Operators =====

/// Implements matrix-scalar division (`Matrix / f64`)
///
/// # Examples
/// ```
/// use nelab::Matrix;
/// let m = Matrix::diagonal(&[2.0, 4.0]);
/// let result = &m / 2.0;  // [[1, 0], [0, 2]]
/// ```
impl Div<f64> for &Matrix {
    type Output = Matrix;

    fn div(self, rhs: f64) -> Self::Output {
        self.scalar_division(rhs)
    }
}

/// Implements scalar-matrix division (`f64 / Matrix`)
///
/// # Examples
/// ```
/// use nelab::Matrix;
/// let m = Matrix::diagonal(&[2.0, 4.0]);
/// let result = 4.0 / &m;  // [[2, 0], [0, 1]]
/// ```
impl Div<&Matrix> for f64 {
    type Output = Matrix;

    fn div(self, rhs: &Matrix) -> Self::Output {
        rhs.scalar_operation(self, |a, b| b / a)
    }
}

/// Implements compound scalar division assignment (`Matrix /= f64`)
///
/// # Examples
/// ```
/// use nelab::Matrix;
/// let mut m = Matrix::diagonal(&[2.0, 4.0]);
/// m /= 2.0;  // m now contains [[1, 0], [0, 2]]
/// ```
impl DivAssign<f64> for Matrix {
    fn div_assign(&mut self, rhs: f64) {
        *self = (self as &Matrix) / rhs;
    }
}

/// Implements Hadamard (element-wise) division (`Matrix / Matrix`)
///
/// # Examples
/// ```
/// use nelab::Matrix;
/// let a = Matrix::diagonal(&[4.0, 9.0]);
/// let b = Matrix::diagonal(&[2.0, 3.0]);
/// let quotient = a / b;  // [[2, 0], [0, 3]]
/// ```
///
/// # Errors
/// Returns `Err` if matrices have different dimensions
impl Div<Matrix> for Matrix {
    type Output = Result<Matrix, String>;

    fn div(self, rhs: Matrix) -> Self::Output {
        self.hadamard_division(&rhs)
    }
}

// ===== Unary Operators =====

/// Implements matrix negation (`-Matrix`)
///
/// # Examples
/// ```
/// use nelab::Matrix;
/// let m = Matrix::diagonal(&[1.0, -2.0]);
/// let neg = -&m;  // [[-1, 0], [0, 2]]
/// ```
impl Neg for &Matrix {
    type Output = Matrix;

    fn neg(self) -> Self::Output {
        self.scalar_multiplication(-1.0)
    }
}

// ===== Indexing Operators =====

/// Implements row indexing (`Matrix[row]`)
///
/// Returns a slice of the row's elements
///
/// # Examples
/// ```
/// use nelab::Matrix;
/// let m = Matrix::diagonal(&[1.0, 2.0]);
/// assert_eq!(m[0], [1.0, 0.0]);
/// assert_eq!(m[1], [0.0, 2.0]);
/// ```
///
/// # Panics
/// Panics if row index is out of bounds
impl Index<usize> for Matrix {
    type Output = [f64];

    fn index(&self, row: usize) -> &Self::Output {
        let start = row * self.columns;
        &self.data[start..start + self.columns]
    }
}

/// Implements mutable row indexing (`Matrix[row] = ...`)
///
/// # Examples
/// ```
/// use nelab::Matrix;
/// let mut m = Matrix::zeros(2, 2);
/// m[0][0] = 1.0;
/// m[1][1] = 1.0;
/// assert_eq!(m.data, [1.0, 0.0, 0.0, 1.0]);
/// ```
///
/// # Panics
/// Panics if row index is out of bounds
impl IndexMut<usize> for Matrix {
    fn index_mut(&mut self, row: usize) -> &mut Self::Output {
        let start = row * self.columns;
        &mut self.data[start..start + self.columns]
    }
}