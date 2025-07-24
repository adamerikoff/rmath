use std::fmt;

use rand::Rng;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Matrix<const N: usize> {
    data: [f64; N],
    rows: usize,
    columns: usize,
}

impl<const N: usize> fmt::Display for Matrix<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Matrix<{}x{}> [", self.rows, self.columns)?;

        for row_idx in 0..self.rows {
            write!(f, "  ")?;
            for col_idx in 0..self.columns {
                write!(f, "{:8.4}", self.get(row_idx, col_idx))?;
                if col_idx < self.columns - 1 {
                    write!(f, ", ")?;
                }
            }
            writeln!(f)?;
        }
        write!(f, "]")
    }
}

impl<const N: usize> Matrix<N> {
    pub fn new(rows: usize, columns: usize) -> Self {
        Self::validate_dimensions(rows, columns);

        Self {
            data: Self::random_array(),
            rows,
            columns,
        }
    }

    /// Creates a new matrix filled with zeros
    pub fn zeros(rows: usize, columns: usize) -> Self {
        Self::validate_dimensions(rows, columns);

        Self {
            data: [0.0; N],
            rows,
            columns,
        }
    }

    /// Creates a new matrix filled with ones
    pub fn ones(rows: usize, columns: usize) -> Self {
        Self::validate_dimensions(rows, columns);

        Self {
            data: [1.0; N],
            rows,
            columns,
        }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn columns(&self) -> usize {
        self.columns
    }

    pub fn size(&self) -> usize {
        N
    }

    pub fn shape(&self) -> [usize; 2] {
    [self.rows, self.columns]
}

    /// Validates that the matrix dimensions match the storage size
    fn validate_dimensions(rows: usize, columns: usize) {
        if rows * columns != N {
            panic!(
                "Matrix dimensions {}x{} don't match storage for {} elements",
                rows, columns, N
            );
        }
    }

    // Helper function to check dimensions
    fn check_dimensions_match(&self, other: &Matrix<N>) {
        if self.columns != other.columns || self.rows != other.rows {
            panic!("Matrix dimensions must match for this operation");
        }
    }

    /// Generates an array of random values between 0.0 and 1.0
    fn random_array() -> [f64; N] {
        let mut arr = [0.0; N];
        let mut rng = rand::rng();
        for item in &mut arr {
            *item = rng.random_range(0.0..1.0);
        }

        arr
    }

    pub fn get(&self, row: usize, column: usize) -> f64 {
        self.data[row * self.columns + column]
    }

    pub fn set<T: Into<f64>>(&mut self, row: usize, column: usize, value: T) {
        self.data[row * self.columns + column] = value.into();
    }

    pub fn get_data(&self) -> [f64; N] {
        self.data
    }

    pub fn set_data(&mut self, data: [f64; N]) {
        self.data = data;
    }

    // Matrix addition
    pub fn add_matrix(&self, other: &Matrix<N>) -> Self {
        self.check_dimensions_match(other);
        Self {
            data: std::array::from_fn(|i| self.data[i] + other.data[i]),
            columns: self.columns,
            rows: self.rows,
        }
    }

    // Addition Scalar
    pub fn add_scalar<T: Into<f64>>(&self, scalar: T) -> Self {
        let scalar = scalar.into();
        Self {
            data: std::array::from_fn(|i| self.data[i] + scalar),
            columns: self.columns,
            rows: self.rows,
        }
    }

    // Matrix subtraction
    pub fn sub_matrix(&self, other: &Matrix<N>) -> Self {
        self.check_dimensions_match(other);
        Self {
            data: std::array::from_fn(|i| self.data[i] - other.data[i]),
            columns: self.columns,
            rows: self.rows,
        }
    }

    // Scalar subtraction
    pub fn sub_scalar<T: Into<f64>>(&self, scalar: T) -> Self {
        let scalar = scalar.into();
        Self {
            data: std::array::from_fn(|i| self.data[i] - scalar),
            columns: self.columns,
            rows: self.rows,
        }
    }

    // Hadamard (element-wise) product
    pub fn hadamard_product(&self, other: &Matrix<N>) -> Self {
        self.check_dimensions_match(other);
        Self {
            data: std::array::from_fn(|i| self.data[i] * other.data[i]),
            columns: self.columns,
            rows: self.rows,
        }
    }

    // Hadamard (element-wise) division
    pub fn hadamard_division(&self, other: &Matrix<N>) -> Self {
        self.check_dimensions_match(other);
        Self {
            data: std::array::from_fn(|i| self.data[i] / other.data[i]),
            columns: self.columns,
            rows: self.rows,
        }
    }

    // Scalar multiplication
    pub fn mul_scalar<T: Into<f64>>(&self, scalar: T) -> Self {
        let scalar = scalar.into();
        Self {
            data: std::array::from_fn(|i| self.data[i] * scalar),
            columns: self.columns,
            rows: self.rows,
        }
    }

    // Scalar division
    pub fn div_scalar<T: Into<f64>>(&self, scalar: T) -> Self {
        let scalar = scalar.into();
        Self {
            data: std::array::from_fn(|i| self.data[i] / scalar),
            columns: self.columns,
            rows: self.rows,
        }
    }

    // Matrix multiplication
    pub fn mul(&self, other: &Matrix<N>) -> Self {
        if self.columns != other.rows {
            panic!(
                "Matrix dimensions mismatch: cannot multiply {}x{} by {}x{}",
                self.rows, self.columns, other.rows, other.columns
            );
        }

        let result_rows = self.rows;
        let result_columns = other.columns;
        let common_dimension = self.columns;

        let mut result = Matrix::zeros(result_rows, result_columns);

        for i in 0..result_rows { // Iterate through rows of the result matrix (and self)
            for j in 0..result_columns { // Iterate through columns of the result matrix (and other)
                let mut sum = 0.0;
                for k in 0..common_dimension { // Iterate through the common dimension
                    // C_ij = Sum(A_ik * B_kj)
                    sum += self.get(i, k) * other.get(k, j);
                }
                result.set(i, j, sum);
            }
        }
        result
    }
}

#[macro_export]
macro_rules! mtx {
    ($rows:expr, $columns:expr) => {{
        const ROWS: usize = $rows;
        const COLS: usize = $columns;
        const N: usize = ROWS * COLS;
        Matrix::<N>::new(ROWS, COLS)
    }};
}
