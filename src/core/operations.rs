use crate::core::Matrix;

impl Matrix {
    /// Applies a function to each element of the matrix, returning a new matrix
    ///
    /// # Arguments
    /// * `func` - Function that takes an f64 and returns an f64
    ///
    /// # Examples
    /// ```
    /// use nelab::Matrix;
    /// let m = Matrix::diagonal(&[1.0, 2.0]);
    /// let squared = m.apply(|x| x * x);
    /// assert_eq!(squared.data, vec![1.0, 0.0, 0.0, 4.0]);
    /// ```
    pub fn apply<F>(self, func: F) -> Self
    where
        F: Fn(f64) -> f64,
    {
        Matrix {
            data: self.data.into_iter().map(func).collect(),
            rows: self.rows,
            columns: self.columns,
        }
    }

    /// Performs matrix addition
    ///
    /// # Arguments
    /// * `other` - Matrix to add
    ///
    /// # Returns
    /// - `Ok(Matrix)` if dimensions match
    /// - `Err` if dimensions mismatch
    ///
    /// # Examples
    /// ```
    /// use nelab::Matrix;
    /// let a = Matrix::ones(2, 2);
    /// let b = Matrix::identity(2);
    /// let sum = a.addition(&b).unwrap();
    /// assert_eq!(sum.data, vec![2.0, 1.0, 1.0, 2.0]);
    /// ```
    pub fn addition(&self, other: &Matrix) -> Result<Self, String> {
        self.elementwise_operation(other, |a, b| a + b)
    }

    /// Performs matrix subtraction
    ///
    /// # Arguments
    /// * `other` - Matrix to subtract
    ///
    /// # Returns
    /// - `Ok(Matrix)` if dimensions match
    /// - `Err` if dimensions mismatch
    ///
    /// # Examples
    /// ```
    /// use nelab::Matrix;
    /// let a = Matrix::ones(2, 2);
    /// let b = Matrix::identity(2);
    /// let diff = a.subtraction(&b).unwrap();
    /// assert_eq!(diff.data, vec![0.0, 1.0, 1.0, 0.0]);
    /// ```
    pub fn subtraction(&self, other: &Matrix) -> Result<Self, String> {
        self.elementwise_operation(other, |a, b| a - b)
    }

    /// Performs element-wise (Hadamard) multiplication
    ///
    /// # Arguments
    /// * `other` - Matrix to multiply with
    ///
    /// # Returns
    /// - `Ok(Matrix)` if dimensions match
    /// - `Err` if dimensions mismatch
    ///
    /// # Examples
    /// ```
    /// use nelab::Matrix;
    /// let a = Matrix::diagonal(&[1.0, 2.0]);
    /// let b = Matrix::ones(2, 2);
    /// let product = a.hadamard_multiplication(&b).unwrap();
    /// assert_eq!(product.data, vec![1.0, 0.0, 0.0, 2.0]);
    /// ```
    pub fn hadamard_multiplication(&self, other: &Matrix) -> Result<Self, String> {
        self.elementwise_operation(other, |a, b| a * b)
    }

    /// Performs element-wise (Hadamard) division
    ///
    /// # Arguments
    /// * `other` - Matrix to divide by
    ///
    /// # Returns
    /// - `Ok(Matrix)` if dimensions match
    /// - `Err` if dimensions mismatch
    ///
    /// # Examples
    /// ```
    /// use nelab::Matrix;
    /// let a = Matrix::diagonal(&[4.0, 9.0]);
    /// let b = Matrix::diagonal(&[2.0, 3.0]);
    /// let quotient = a.hadamard_division(&b).unwrap();
    /// assert_eq!(quotient.data, vec![2.0, 0.0, 0.0, 3.0]);
    /// ```
    pub fn hadamard_division(&self, other: &Matrix) -> Result<Self, String> {
        self.elementwise_operation(other, |a, b| a / b)
    }

    /// Adds a scalar to each element of the matrix
    ///
    /// # Arguments
    /// * `scalar` - Value to add
    ///
    /// # Examples
    /// ```
    /// use nelab::Matrix;
    /// let m = Matrix::ones(2, 2);
    /// let result = m.scalar_addition(3.0);
    /// assert_eq!(result.data, vec![4.0, 4.0, 4.0, 4.0]);
    /// ```
    pub fn scalar_addition(&self, scalar: f64) -> Self {
        self.scalar_operation(scalar, |a, b| a + b)
    }

    /// Subtracts a scalar from each element of the matrix
    ///
    /// # Arguments
    /// * `scalar` - Value to subtract
    ///
    /// # Examples
    /// ```
    /// use nelab::Matrix;
    /// let m = Matrix::ones(2, 2);
    /// let result = m.scalar_subtraction(1.0);
    /// assert_eq!(result.data, vec![0.0, 0.0, 0.0, 0.0]);
    /// ```
    pub fn scalar_subtraction(&self, scalar: f64) -> Self {
        self.scalar_operation(scalar, |a, b| a - b)
    }

    /// Multiplies each element of the matrix by a scalar
    ///
    /// # Arguments
    /// * `scalar` - Value to multiply by
    ///
    /// # Examples
    /// ```
    /// use nelab::Matrix;
    /// let m = Matrix::diagonal(&[1.0, 2.0]);
    /// let result = m.scalar_multiplication(3.0);
    /// assert_eq!(result.data, vec![3.0, 0.0, 0.0, 6.0]);
    /// ```
    pub fn scalar_multiplication(&self, scalar: f64) -> Self {
        self.scalar_operation(scalar, |a, b| a * b)
    }

    /// Divides each element of the matrix by a scalar
    ///
    /// # Arguments
    /// * `scalar` - Value to divide by
    ///
    /// # Examples
    /// ```
    /// use nelab::Matrix;
    /// let m = Matrix::diagonal(&[2.0, 4.0]);
    /// let result = m.scalar_division(2.0);
    /// assert_eq!(result.data, vec![1.0, 0.0, 0.0, 2.0]);
    /// ```
    pub fn scalar_division(&self, scalar: f64) -> Self {
        self.scalar_operation(scalar, |a, b| a / b)
    }

    /// Computes the dot product of two vectors
    ///
    /// # Arguments
    /// * `other` - Vector to compute dot product with
    ///
    /// # Returns
    /// - `Ok(f64)` if both are vectors of same length
    /// - `Err` if inputs aren't vectors or lengths mismatch
    ///
    /// # Examples
    /// ```
    /// use nelab::Matrix;
    /// let a = Matrix::new(3, 1, vec![1.0, 2.0, 3.0]);
    /// let b = Matrix::new(3, 1, vec![4.0, 5.0, 6.0]);
    /// let dot = a.dot_product(&b).unwrap();
    /// assert_eq!(dot, 32.0);
    /// ```
    pub fn dot_product(&self, other: &Matrix) -> Result<f64, String> {
        if !self.check_vector() || !other.check_vector() {
            return Err("Both matrices must be vectors (1xN or Nx1)".to_string());
        }
        self.check_dimensions_match(other)?;

        Ok(self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a * b)
            .sum())
    }

    /// Computes the cross product of two 3D vectors
    ///
    /// # Arguments
    /// * `other` - Vector to compute cross product with
    ///
    /// # Returns
    /// - `Ok(Matrix)` containing the cross product if both are 3D vectors
    /// - `Err` if inputs aren't 3D vectors
    ///
    /// # Examples
    /// ```
    /// use nelab::Matrix;
    /// let a = Matrix::new(3, 1, vec![1.0, 0.0, 0.0]);
    /// let b = Matrix::new(3, 1, vec![0.0, 1.0, 0.0]);
    /// let cross = a.cross_product(&b).unwrap();
    /// assert_eq!(cross.data, vec![0.0, 0.0, 1.0]);
    /// ```
    pub fn cross_product(&self, other: &Matrix) -> Result<Self, String> {
        if !self.check_3d_vector() || !other.check_3d_vector() {
            return Err("Cross product only defined for 3D vectors (length 3)".to_string());
        }

        let (a, b) = (&self.data, &other.data);

        let data = vec![
            a[1] * b[2] - a[2] * b[1], // x component
            a[2] * b[0] - a[0] * b[2], // y component
            a[0] * b[1] - a[1] * b[0], // z component
        ];

        Ok(Self {
            data,
            rows: 3,
            columns: 1,
        })
    }

    /// Computes the magnitude (length) of a vector
    ///
    /// # Returns
    /// - `Ok(f64)` containing the magnitude if matrix is a vector
    /// - `Err` if matrix is not a vector
    ///
    /// # Examples
    /// ```
    /// use nelab::Matrix;
    /// use std::f64::consts::SQRT_2;
    /// let v = Matrix::new(2, 1, vec![1.0, 1.0]);
    /// let mag = v.magnitude().unwrap();
    /// assert!((mag - SQRT_2).abs() < 1e-10);
    /// ```
    pub fn magnitude(&self) -> Result<f64, String> {
        if !self.check_vector() {
            return Err("Matrix must be a vector to calculate magnitude".to_string());
        }

        self.dot_product(self).map(|dot| dot.sqrt())
    }

    /// Computes the unit vector in the same direction
    ///
    /// # Returns
    /// - `Ok(Matrix)` containing unit vector if matrix is a non-zero vector
    /// - `Err` if matrix is not a vector or has zero magnitude
    ///
    /// # Examples
    /// ```
    /// use nelab::Matrix;
    /// let v = Matrix::new(2, 1, vec![3.0, 4.0]);
    /// let unit = v.unit_vector().unwrap();
    /// assert!((unit.magnitude().unwrap() - 1.0).abs() < 1e-10);
    /// ```
    pub fn unit_vector(&self) -> Result<Self, String> {
        let mag = self.magnitude()?;

        if mag == 0.0 {
            return Err("Cannot normalize zero-length vector".to_string());
        }

        Ok(self.scalar_operation(mag, |a, m| a / m))
    }

    /// Normalizes the vector in-place to unit length
    ///
    /// # Returns
    /// - `Ok(())` if successful
    /// - `Err` if matrix is not a vector or has zero magnitude
    ///
    /// # Examples
    /// ```
    /// use nelab::Matrix;
    /// let mut v = Matrix::new(2, 1, vec![3.0, 4.0]);
    /// v.normalize().unwrap();
    /// assert!((v.magnitude().unwrap() - 1.0).abs() < 1e-10);
    /// ```
    pub fn normalize(&mut self) -> Result<(), String> {
        let mag = self.magnitude()?;

        if mag == 0.0 {
            return Err("Cannot normalize zero-length vector".to_string());
        }

        for item in &mut self.data {
            *item /= mag;
        }

        Ok(())
    }

    /// Computes the scalar projection of this vector onto another
    ///
    /// # Arguments
    /// * `other` - Vector to project onto
    ///
    /// # Returns
    /// - `Ok(f64)` containing the projection length if both are vectors
    /// - `Err` if inputs aren't vectors or lengths mismatch
    ///
    /// # Examples
    /// ```
    /// use nelab::Matrix;
    /// let a = Matrix::new(2, 1, vec![1.0, 0.0]);
    /// let b = Matrix::new(2, 1, vec![1.0, 1.0]);
    /// let proj = a.scalar_projection_onto(&b).unwrap();
    /// assert!((proj - 1.0/f64::sqrt(2.0)).abs() < 1e-10);
    /// ```
    pub fn scalar_projection_onto(&self, other: &Matrix) -> Result<f64, String> {
        let dot_product = self.dot_product(other)?; // a·b
        let other_mag = other.magnitude()?; // ‖b‖
        Ok(dot_product / other_mag) // (a·b)/‖b‖
    }

    /// Computes the vector projection of this vector onto another
    ///
    /// # Arguments
    /// * `other` - Vector to project onto
    ///
    /// # Returns
    /// - `Ok(Matrix)` containing the projection vector if both are vectors
    /// - `Err` if inputs aren't vectors or lengths mismatch
    ///
    /// # Examples
    /// ```
    /// use nelab::Matrix;
    /// let a = Matrix::new(2, 1, vec![1.0, 0.0]);
    /// let b = Matrix::new(2, 1, vec![1.0, 1.0]);
    /// let proj = a.vector_projection_onto(&b).unwrap();
    /// assert_eq!(proj.data, vec![0.5, 0.5]);
    /// ```
    pub fn vector_projection_onto(&self, other: &Matrix) -> Result<Self, String> {
        let dot_product = self.dot_product(other)?; // a·b
        let other_mag_squared = other.dot_product(other)?; // ‖b‖²
        let scale = dot_product / other_mag_squared; // (a·b)/‖b‖²
        Ok(other.scalar_operation(scale, |x, s| x * s)) // scale * b
    }

    /// Computes the transpose of the matrix
    ///
    /// # Returns
    /// New matrix that is the transpose of the original
    ///
    /// # Examples
    /// ```
    /// use nelab::Matrix;
    /// let m = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// let transposed = m.transpose();
    /// assert_eq!(transposed.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    /// assert_eq!(transposed.rows, 3);
    /// assert_eq!(transposed.columns, 2);
    /// ```
    pub fn transpose(&self) -> Self {
        let mut data = vec![0.0; self.rows * self.columns];

        for i in 0..self.rows {
            for j in 0..self.columns {
                data[j * self.rows + i] = self.data[i * self.columns + j];
            }
        }

        Matrix {
            data,
            rows: self.columns,
            columns: self.rows,
        }
    }

    /// Performs standard matrix multiplication
    ///
    /// # Arguments
    /// * `other` - Matrix to multiply with
    ///
    /// # Returns
    /// - `Ok(Matrix)` if dimensions are compatible
    /// - `Err` if dimensions are incompatible
    ///
    /// # Examples
    /// ```
    /// use nelab::Matrix;
    /// let a = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// let b = Matrix::new(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    /// let product = a.multiply(&b).unwrap();
    /// assert_eq!(product.data, vec![58.0, 64.0, 139.0, 154.0]);
    /// ```
    pub fn multiply(&self, other: &Matrix) -> Result<Self, String> {
        self.check_multiplication_compatible(other)?;

        let mut data = vec![0.0; self.rows * other.columns];

        for i in 0..self.rows {
            for j in 0..other.columns {
                let mut sum = 0.0;
                for k in 0..self.columns {
                    sum += self.data[i * self.columns + k] * other.data[k * other.columns + j];
                }
                data[i * other.columns + j] = sum;
            }
        }

        Ok(Matrix {
            data,
            rows: self.rows,
            columns: other.columns,
        })
    }

    /// Computes the minor matrix (used for determinant calculation)
    ///
    /// # Arguments
    /// * `row` - Row to exclude (0-based)
    /// * `col` - Column to exclude (0-based)
    ///
    /// # Returns
    /// - `Ok(Matrix)` containing the minor if indices are valid
    /// - `Err` if matrix isn't square or indices are out of bounds
    fn minor(&self, row: usize, col: usize) -> Result<Self, String> {
        self.check_square()?;
        
        if row >= self.rows || col >= self.columns {
            return Err(format!(
                "Row {} or column {} out of bounds for {}x{} matrix",
                row, col, self.rows, self.columns
            ));
        }

        let size = self.rows - 1;
        let mut data = Vec::with_capacity(size * size);

        for i in 0..self.rows {
            if i == row {
                continue;
            }
            for j in 0..self.columns {
                if j == col {
                    continue;
                }
                data.push(self.data[i * self.columns + j]);
            }
        }

        Ok(Matrix {
            data,
            rows: size,
            columns: size,
        })
    }

    /// Computes the determinant of the matrix
    ///
    /// # Returns
    /// - `Ok(f64)` containing the determinant if matrix is square
    /// - `Err` if matrix isn't square
    ///
    /// # Examples
    /// ```
    /// use nelab::Matrix;
    /// let m = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    /// let det = m.determinant().unwrap();
    /// assert_eq!(det, -2.0);
    /// ```
    pub fn determinant(&self) -> Result<f64, String> {
        self.check_square()?;

        match self.rows {
            1 => Ok(self.data[0]),
            2 => Ok(self.data[0] * self.data[3] - self.data[1] * self.data[2]),
            3 => {
                // Rule of Sarrus for 3x3
                let a = &self.data;
                Ok(a[0] * a[4] * a[8] + 
                   a[1] * a[5] * a[6] + 
                   a[2] * a[3] * a[7] -
                   a[2] * a[4] * a[6] -
                   a[1] * a[3] * a[8] -
                   a[0] * a[5] * a[7])
            }
            _ => {
                // Laplace expansion for NxN
                let mut det = 0.0;
                for col in 0..self.columns {
                    let minor = self.minor(0, col)?;
                    let sign = if col % 2 == 0 { 1.0 } else { -1.0 };
                    det += self.data[col] * sign * minor.determinant()?;
                }
                Ok(det)
            }
        }
    }

    /// Computes the matrix inverse
    ///
    /// # Returns
    /// - `Ok(Matrix)` containing the inverse if matrix is invertible
    /// - `Err` if matrix isn't square or is singular
    ///
    /// # Examples
    /// ```
    /// use nelab::Matrix;
    /// let m = Matrix::new(2, 2, vec![4.0, 7.0, 2.0, 6.0]);
    /// let inv = m.inverse().unwrap();
    /// // Verify it's the inverse by multiplying
    /// let product = m.multiply(&inv).unwrap();
    /// assert!((product.data[0] - 1.0).abs() < 1e-10);
    /// assert!((product.data[3] - 1.0).abs() < 1e-10);
    /// ```
    pub fn inverse(&self) -> Result<Self, String> {
        self.check_square()?;
        let det = self.determinant()?;
        
        if det.abs() < f64::EPSILON {
            return Err("Matrix is singular (determinant = 0), cannot invert".to_string());
        }

        let size = self.rows;
        let mut cofactors = Matrix::ones(size, size);

        // Calculate cofactor matrix
        for row in 0..size {
            for col in 0..size {
                let minor = self.minor(row, col)?;
                let sign = if (row + col) % 2 == 0 { 1.0 } else { -1.0 };
                cofactors.data[col * size + row] = sign * minor.determinant()?;
            }
        }

        // Transpose and divide by determinant
        Ok(cofactors.scalar_operation(det, |x, d| x / d))
    }

    /// Computes the trace of the matrix (sum of diagonal elements)
    ///
    /// # Returns
    /// - `Ok(f64)` containing the trace if matrix is square
    /// - `Err` if matrix isn't square
    ///
    /// # Examples
    /// ```
    /// use nelab::Matrix;
    /// let m = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    /// let trace = m.trace().unwrap();
    /// assert_eq!(trace, 5.0);
    /// ```
    pub fn trace(&self) -> Result<f64, String> {
        self.check_square()?;
        Ok((0..self.rows)
            .map(|i| self.data[i * self.columns + i])
            .sum())
    }

    /// Computes the rank of the matrix (number of linearly independent rows/columns)
    ///
    /// # Returns
    /// The rank as a usize
    ///
    /// # Examples
    /// ```
    /// use nelab::Matrix;
    /// let m = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0]);
    /// assert_eq!(m.rank(), 1);
    /// ```
    pub fn rank(&self) -> usize {
        // Convert to row echelon form and count non-zero rows
        let mut matrix = self.clone();
        let mut rank = 0;
        let n = self.rows.min(self.columns);

        for col in 0..n {
            // Find pivot row
            if let Some(pivot_row) = (rank..self.rows)
                .find(|&row| (matrix.data[row * self.columns + col]).abs() > f64::EPSILON)
            {
                // Swap rows if needed
                if pivot_row != rank {
                    matrix.swap_rows(pivot_row, rank);
                }

                // Eliminate this column in other rows
                for row in (0..self.rows).filter(|&r| r != rank) {
                    let factor = matrix.data[row * self.columns + col] / 
                                matrix.data[rank * self.columns + col];
                    
                    if factor.abs() > f64::EPSILON {
                        for c in col..self.columns {
                            let target_idx = row * self.columns + c;
                            let source_idx = rank * self.columns + c;
                            matrix.data[target_idx] -= factor * matrix.data[source_idx];
                        }
                    }
                }

                rank += 1;
            }
        }

        rank
    }
}