use crate::core::Matrix;

impl Matrix {
    /// Checks if matrices are compatible for multiplication
    ///
    /// # Arguments
    /// * `other` - Matrix to check compatibility with
    ///
    /// # Returns
    /// - `Ok(())` if matrices can be multiplied (self.columns == other.rows)
    /// - `Err` with descriptive message if incompatible
    ///
    /// # Examples
    /// ```
    /// use nelab::Matrix;
    /// let a = Matrix::zeros(2, 3);  // 2x3
    /// let b = Matrix::zeros(3, 4);  // 3x4
    /// assert!(a.check_multiplication_compatible(&b).is_ok());
    ///
    /// let c = Matrix::zeros(2, 2);  // 2x2
    /// assert!(a.check_multiplication_compatible(&c).is_err());
    /// ```
    pub(crate) fn check_multiplication_compatible(&self, other: &Matrix) -> Result<(), String> {
        if self.columns != other.rows {
            return Err(format!(
                "Matrix dimensions incompatible for multiplication: {}x{} vs {}x{}",
                self.rows, self.columns, other.rows, other.columns
            ));
        }
        Ok(())
    }

    /// Verifies matrix has specific dimensions
    ///
    /// # Arguments
    /// * `rows` - Expected row count
    /// * `columns` - Expected column count
    ///
    /// # Returns
    /// - `Ok(())` if dimensions match
    /// - `Err` with descriptive message if mismatch
    ///
    /// # Examples
    /// ```
    /// use nelab::Matrix;
    /// let m = Matrix::identity(2);
    /// assert!(m.check_dimensions(2, 2).is_ok());
    /// assert!(m.check_dimensions(3, 2).is_err());
    /// ```
    pub(crate) fn check_dimensions(&self, rows: usize, columns: usize) -> Result<(), String> {
        if self.rows != rows || self.columns != columns {
            return Err(format!(
                "Expected {}x{} matrix, got {}x{}",
                rows, columns, self.rows, self.columns
            ));
        }
        Ok(())
    }

    /// Checks if two matrices have identical dimensions
    ///
    /// # Arguments
    /// * `other` - Matrix to compare against
    ///
    /// # Returns
    /// - `Ok(())` if dimensions match
    /// - `Err` with descriptive message if mismatch
    ///
    /// # Examples
    /// ```
    /// use nelab::Matrix;
    /// let a = Matrix::zeros(2, 3);
    /// let b = Matrix::zeros(2, 3);
    /// assert!(a.check_dimensions_match(&b).is_ok());
    ///
    /// let c = Matrix::zeros(3, 2);
    /// assert!(a.check_dimensions_match(&c).is_err());
    /// ```
    pub(crate) fn check_dimensions_match(&self, other: &Matrix) -> Result<(), String> {
        if self.rows != other.rows || self.columns != other.columns {
            return Err(format!(
                "Dimension mismatch: self is {}x{}, other is {}x{}",
                self.rows, self.columns, other.rows, other.columns
            ));
        }
        Ok(())
    }

    /// Verifies matrix is square (rows == columns)
    ///
    /// # Returns
    /// - `Ok(())` for square matrices
    /// - `Err` with descriptive message for non-square
    ///
    /// # Examples
    /// ```
    /// use nelab::Matrix;
    /// let square = Matrix::identity(3);
    /// assert!(square.check_square().is_ok());
    ///
    /// let rect = Matrix::zeros(2, 3);
    /// assert!(rect.check_square().is_err());
    /// ```
    pub(crate) fn check_square(&self) -> Result<(), String> {
        if self.rows != self.columns {
            return Err(format!(
                "Matrix is not square: {}x{}",
                self.rows, self.columns
            ));
        }
        Ok(())
    }

    /// Checks if matrix is a vector (either row or column vector)
    ///
    /// # Returns
    /// `true` if matrix is 1×n or n×1, `false` otherwise
    ///
    /// # Examples
    /// ```
    /// use nelab::Matrix;
    /// let row_vec = Matrix::zeros(1, 3);
    /// let col_vec = Matrix::zeros(3, 1);
    /// let non_vec = Matrix::zeros(2, 2);
    ///
    /// assert!(row_vec.check_vector());
    /// assert!(col_vec.check_vector());
    /// assert!(!non_vec.check_vector());
    /// ```
    pub(crate) fn check_vector(&self) -> bool {
        self.rows == 1 || self.columns == 1
    }

    /// Checks if matrix is a 3D vector (either 3×1 or 1×3)
    ///
    /// # Returns
    /// `true` if matrix is valid 3D vector, `false` otherwise
    ///
    /// # Examples
    /// ```
    /// use nelab::Matrix;
    /// let valid = Matrix::zeros(3, 1);
    /// let invalid_size = Matrix::zeros(3, 2);
    /// let invalid_dims = Matrix::zeros(2, 2);
    ///
    /// assert!(valid.check_3d_vector());
    /// assert!(!invalid_size.check_3d_vector());
    /// assert!(!invalid_dims.check_3d_vector());
    /// ```
    pub(crate) fn check_3d_vector(&self) -> bool {
        (self.rows == 3 && self.columns == 1) || (self.rows == 1 && self.columns == 3) && self.data.len() == 3
    }

    /// Performs element-wise operation between two matrices
    ///
    /// # Arguments
    /// * `other` - Matrix to operate with
    /// * `op` - Operation function (e.g., |a, b| a + b)
    ///
    /// # Returns
    /// - `Ok(Matrix)` with operation results if dimensions match
    /// - `Err` if dimensions mismatch
    ///
    /// # Examples
    /// ```
    /// use nelab::Matrix;
    /// let a = Matrix::ones(2, 2);
    /// let b = Matrix::diagonal(&[1.0, 2.0]);
    ///
    /// let sum = a.elementwise_operation(&b, |x, y| x + y).unwrap();
    /// assert_eq!(sum.data, vec![2.0, 1.0, 1.0, 3.0]);
    /// ```
    pub(crate) fn elementwise_operation<F>(&self, other: &Matrix, op: F) -> Result<Self, String>
    where
        F: Fn(f64, f64) -> f64,
    {
        self.check_dimensions_match(other)?;
        let data: Vec<f64> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| op(a, b))
            .collect();
        Ok(Matrix {
            data,
            rows: self.rows,
            columns: self.columns,
        })
    }

    /// Applies scalar operation to each matrix element
    ///
    /// # Arguments
    /// * `scalar` - Scalar value to operate with
    /// * `op` - Operation function (e.g., |a, s| a * s)
    ///
    /// # Returns
    /// New matrix with operation results
    ///
    /// # Examples
    /// ```
    /// use nelab::Matrix;
    /// let m = Matrix::diagonal(&[1.0, 2.0]);
    /// let scaled = m.scalar_operation(2.0, |x, s| x * s);
    /// assert_eq!(scaled.data, vec![2.0, 0.0, 0.0, 4.0]);
    /// ```
    pub(crate) fn scalar_operation<F>(&self, scalar: f64, op: F) -> Self
    where
        F: Fn(f64, f64) -> f64,
    {
        Matrix {
            data: self.data.iter().map(|&a| op(a, scalar)).collect(),
            columns: self.columns,
            rows: self.rows,
        }
    }

    /// Swaps two rows in the matrix
    ///
    /// # Arguments
    /// * `row1` - First row index (0-based)
    /// * `row2` - Second row index (0-based)
    ///
    /// # Notes
    /// - Silently returns if either index is out of bounds
    /// - Modifies matrix in-place
    ///
    /// # Examples
    /// ```
    /// use nelab::Matrix;
    /// let mut m = Matrix::diagonal(&[1.0, 2.0]);
    /// m.swap_rows(0, 1);
    /// assert_eq!(m.data, vec![0.0, 2.0, 1.0, 0.0]);
    /// ```
    pub(crate) fn swap_rows(&mut self, row1: usize, row2: usize) {
        if row1 >= self.rows || row2 >= self.rows {
            return;
        }
        
        for col in 0..self.columns {
            let idx1 = row1 * self.columns + col;
            let idx2 = row2 * self.columns + col;
            self.data.swap(idx1, idx2);
        }
    }
}