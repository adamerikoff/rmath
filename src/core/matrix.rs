use std::fmt;

/// A matrix representation with floating-point elements.
///
/// # Fields
/// - `data`: Vector storing matrix elements in row-major order
/// - `rows`: Number of rows in the matrix
/// - `columns`: Number of columns in the matrix
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    pub data: Vec<f64>,
    pub rows: usize,
    pub columns: usize,
}

impl fmt::Display for Matrix {
    /// Formats the matrix for display, showing a clear 2D structure.
    ///
    /// The output format uses nested brackets to clearly represent the matrix structure:
    /// - Outer brackets `[]` enclose the entire matrix
    /// - Each row is enclosed in its own `[]` brackets
    /// - Elements are separated by commas
    /// - Rows are separated by newlines for readability
    ///
    /// # Examples
    /// ```
    /// use nelab::Matrix;
    /// use std::fmt::Write;
    ///
    /// let m = Matrix {
    ///     data: vec![1.0, 2.0, 3.0, 4.0],
    ///     rows: 2,
    ///     columns: 2,
    /// };
    ///
    /// assert_eq!(format!("{}", m), "[[1, 2]\n [3, 4]]");
    ///
    /// let empty = Matrix {
    ///     data: vec![],
    ///     rows: 0,
    ///     columns: 0,
    /// };
    ///
    /// assert_eq!(format!("{}", empty), "[]");
    /// ```
    ///
    /// # Output Format
    /// For a 2x2 matrix [[1, 2], [3, 4]], the output will be:
    /// ```text
    /// [[1, 2]
    ///  [3, 4]]
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Handle empty matrix case
        if self.data.is_empty() {
            return write!(f, "[]");
        }

        write!(f, "[")?; // Start outer brackets

        for row in 0..self.rows {
            if row != 0 {
                write!(f, "\n ")?; // Newline and space for alignment
            }
            write!(f, "[")?; // Start row bracket
            
            // Format each element in the row
            for col in 0..self.columns {
                let index = row * self.columns + col;
                write!(f, "{}", self.data[index])?;
                
                // Add separator unless it's the last element
                if col != self.columns - 1 {
                    write!(f, ", ")?;
                }
            }

            write!(f, "]")?; // Close row bracket
        }

        write!(f, "]")?; // Close outer brackets
        Ok(())
    }
}