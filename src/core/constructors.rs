use rand::Rng;
use crate::core::Matrix;

impl Matrix {
    /// Creates a new matrix with random values between 0.0 and 1.0
    ///
    /// # Arguments
    /// * `rows` - Number of rows in the matrix
    /// * `columns` - Number of columns in the matrix
    ///
    /// # Examples
    /// ```
    /// use nelab::Matrix;
    /// let random_matrix = Matrix::new(2, 2);
    /// // Contains random values like:
    /// // [[0.1234, 0.5678]
    /// //  [0.9012, 0.3456]]
    /// ```
    pub fn new(rows: usize, columns: usize) -> Self {
        let mut rng = rand::thread_rng();
        Matrix {
            data: (0..rows * columns).map(|_| rng.gen_range(0.0..1.0)).collect(),
            rows,
            columns,
        }
    }

    /// Creates a matrix filled with ones
    ///
    /// # Arguments
    /// * `rows` - Number of rows
    /// * `columns` - Number of columns
    ///
    /// # Examples
    /// ```
    /// use nelab::Matrix;
    /// let ones = Matrix::ones(2, 3);
    /// assert_eq!(ones.data, vec![1.0; 6]);
    /// ```
    pub fn ones(rows: usize, columns: usize) -> Self {
        Matrix {
            data: vec![1.0; rows * columns],
            rows,
            columns,
        }
    }

    /// Creates a matrix filled with zeros
    ///
    /// # Arguments
    /// * `rows` - Number of rows
    /// * `columns` - Number of columns
    ///
    /// # Examples
    /// ```
    /// use nelab::Matrix;
    /// let zeros = Matrix::zeros(3, 2);
    /// assert_eq!(zeros.data, vec![0.0; 6]);
    /// ```
    pub fn zeros(rows: usize, columns: usize) -> Self {
        Matrix {
            data: vec![0.0; rows * columns],
            rows,
            columns,
        }
    }

    /// Creates an identity matrix (square matrix with 1s on diagonal and 0s elsewhere)
    ///
    /// # Arguments
    /// * `size` - Dimension of the square matrix
    ///
    /// # Examples
    /// ```
    /// use nelab::Matrix;
    /// let id = Matrix::identity(3);
    /// assert_eq!(id.data, vec![1.0, 0.0, 0.0,
    ///                          0.0, 1.0, 0.0,
    ///                          0.0, 0.0, 1.0]);
    /// ```
    pub fn identity(size: usize) -> Self {
        let mut data = vec![0.0; size * size];
        for i in 0..size {
            data[i * size + i] = 1.0;
        }
        Matrix {
            data,
            rows: size,
            columns: size,
        }
    }

    /// Creates a diagonal matrix from a vector of values
    ///
    /// # Arguments
    /// * `values` - Vector of values to place on the diagonal
    ///
    /// # Examples
    /// ```
    /// use nelab::Matrix;
    /// let diag = Matrix::diagonal(&[1.0, 2.0, 3.0]);
    /// assert_eq!(diag.data, vec![1.0, 0.0, 0.0,
    ///                            0.0, 2.0, 0.0,
    ///                            0.0, 0.0, 3.0]);
    /// ```
    ///
    /// # Panics
    /// The input slice must not be empty.
    pub fn diagonal(values: &[f64]) -> Self {
        assert!(!values.is_empty(), "Cannot create diagonal matrix from empty slice");
        let size = values.len();
        let mut data = vec![0.0; size * size];
        for i in 0..size {
            data[i * size + i] = values[i];
        }
        Matrix {
            data,
            rows: size,
            columns: size,
        }
    }
}