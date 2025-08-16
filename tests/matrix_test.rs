use nelab::*;

use approx::assert_relative_eq;

// Helper function to create a test matrix
fn test_matrix() -> Matrix {
    Matrix {
        data: vec![1.0, 2.0, 3.0, 4.0],
        rows: 2,
        columns: 2,
    }
}

#[test]
fn test_matrix_creation() {
    let zeros = Matrix::zeros(2, 2);
    assert_eq!(zeros.data, vec![0.0, 0.0, 0.0, 0.0]);

    let ones = Matrix::ones(2, 2);
    assert_eq!(ones.data, vec![1.0, 1.0, 1.0, 1.0]);

    let identity = Matrix::identity(2);
    assert_eq!(identity.data, vec![1.0, 0.0, 0.0, 1.0]);

    let diagonal = Matrix::diagonal(&[1.0, 2.0]);
    assert_eq!(diagonal.data, vec![1.0, 0.0, 0.0, 2.0]);
}

#[test]
fn test_addition() {
    let m1 = test_matrix();
    let m2 = Matrix::ones(2, 2);

    let sum = (&m1 + &m2).unwrap();
    assert_eq!(sum.data, vec![2.0, 3.0, 4.0, 5.0]);

    // Test incompatible dimensions
    let m3 = Matrix::ones(3, 3);
    assert!(m1.addition(&m3).is_err());
}

#[test]
fn test_scalar_operations() {
    let m = test_matrix();

    let added = &m + 1.0;
    assert_eq!(added.data, vec![2.0, 3.0, 4.0, 5.0]);

    let multiplied = &m * 2.0;
    assert_eq!(multiplied.data, vec![2.0, 4.0, 6.0, 8.0]);

    let divided = &m / 2.0;
    assert_eq!(divided.data, vec![0.5, 1.0, 1.5, 2.0]);
}

#[test]
fn test_multiplication() {
    let m1 = test_matrix();
    let m2 = Matrix {
        data: vec![5.0, 6.0, 7.0, 8.0],
        rows: 2,
        columns: 2,
    };

    let product = m1.multiply(&m2).unwrap();
    assert_eq!(product.data, vec![19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn test_transpose() {
    let m = Matrix {
        data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        rows: 2,
        columns: 3,
    };

    let transposed = m.transpose();
    assert_eq!(transposed.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    assert_eq!(transposed.rows, 3);
    assert_eq!(transposed.columns, 2);
}

#[test]
fn test_determinant() {
    let m = test_matrix();
    assert_eq!(m.determinant().unwrap(), -2.0);

    let m3x3 = Matrix {
        data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        rows: 3,
        columns: 3,
    };
    assert_eq!(m3x3.determinant().unwrap(), 0.0); // Singular matrix
}

#[test]
fn test_inverse() {
    let m = test_matrix();
    let inv = m.inverse().unwrap();

    // Verify it's actually the inverse
    let product = m.multiply(&inv).unwrap();
    assert_relative_eq!(product.data[0], 1.0, epsilon = 1e-10);
    assert_relative_eq!(product.data[3], 1.0, epsilon = 1e-10);
    assert_relative_eq!(product.data[1], 0.0, epsilon = 1e-10);
    assert_relative_eq!(product.data[2], 0.0, epsilon = 1e-10);

    // Test non-invertible matrix
    let singular = Matrix {
        data: vec![1.0, 2.0, 2.0, 4.0],
        rows: 2,
        columns: 2,
    };
    assert!(singular.inverse().is_err());
}

#[test]
fn test_vector_operations() {
    let v1 = Matrix {
        data: vec![1.0, 2.0, 3.0],
        rows: 3,
        columns: 1,
    };
    let v2 = Matrix {
        data: vec![4.0, 5.0, 6.0],
        rows: 3,
        columns: 1,
    };

    // Dot product
    assert_eq!(v1.dot_product(&v2).unwrap(), 32.0);

    // Cross product
    let cross = v1.cross_product(&v2).unwrap();
    assert_eq!(cross.data, vec![-3.0, 6.0, -3.0]);

    // Magnitude
    assert_relative_eq!(v1.magnitude().unwrap(), (14.0f64).sqrt());

    // Normalization
    let unit = v1.unit_vector().unwrap();
    assert_relative_eq!(unit.magnitude().unwrap(), 1.0, epsilon = 1e-10);
}

#[test]
fn test_rank() {
    let full_rank = test_matrix();
    assert_eq!(full_rank.rank(), 2);

    let rank1 = Matrix {
        data: vec![1.0, 2.0, 2.0, 4.0],
        rows: 2,
        columns: 2,
    };
    assert_eq!(rank1.rank(), 1);
}

#[test]
fn test_operator_overloading() {
    let m1 = test_matrix();
    let m2 = Matrix::ones(2, 2);

    // Addition
    let sum = &m1 + &m2;
    assert_eq!(sum.unwrap().data, vec![2.0, 3.0, 4.0, 5.0]);

    // Scalar multiplication
    let scaled = &m1 * 2.0;
    assert_eq!(scaled.data, vec![2.0, 4.0, 6.0, 8.0]);

    // Negation
    let neg = -&m1;
    assert_eq!(neg.data, vec![-1.0, -2.0, -3.0, -4.0]);
}

#[test]
fn test_indexing() {
    let mut m = test_matrix();

    // Read access
    assert_eq!(m[0][0], 1.0);
    assert_eq!(m[1][1], 4.0);

    // Write access
    m[0][0] = 10.0;
    assert_eq!(m.data[0], 10.0);
}

#[test]
fn test_display_floats() {
    let m = Matrix {
        data: vec![1.23, 4.567, 8.901, 2.345],
        rows: 2,
        columns: 2,
    };
    
    let expected = "[[1.23, 4.567]\n [8.901, 2.345]]";
    assert_eq!(format!("{}", m), expected);
}

#[test]
fn test_elementwise_operations() {
    let m1 = Matrix {
        data: vec![1.0, 2.0, 3.0, 4.0],
        rows: 2,
        columns: 2,
    };
    let m2 = Matrix {
        data: vec![5.0, 6.0, 7.0, 8.0],
        rows: 2,
        columns: 2,
    };

    // Hadamard product
    let hadamard = m1.hadamard_multiplication(&m2).unwrap();
    assert_eq!(hadamard.data, vec![5.0, 12.0, 21.0, 32.0]);

    // Hadamard division
    let hadamard_div = m1.hadamard_division(&m2).unwrap();
    assert_relative_eq!(hadamard_div.data[0], 0.2, epsilon = 1e-10);
    assert_relative_eq!(hadamard_div.data[1], 2.0/6.0, epsilon = 1e-10);
}

#[test]
fn test_matrix_power() {
    let m = Matrix {
        data: vec![1.0, 2.0, 3.0, 4.0],
        rows: 2,
        columns: 2,
    };

    // Square of matrix
    let squared = m.multiply(&m).unwrap();
    assert_eq!(squared.data, vec![7.0, 10.0, 15.0, 22.0]);

    // Multiply by identity should return original
    let identity = Matrix::identity(2);
    let product = m.multiply(&identity).unwrap();
    assert_eq!(product.data, m.data);
}

#[test]
fn test_special_matrices() {
    // Zero matrix multiplication
    let zero = Matrix::zeros(2, 2);
    let m = test_matrix();
    let product = zero.multiply(&m).unwrap();
    assert_eq!(product.data, vec![0.0; 4]);

    // Diagonal matrix operations
    let diag = Matrix::diagonal(&[2.0, 3.0]);
    let result = diag.multiply(&diag).unwrap();
    assert_eq!(result.data, vec![4.0, 0.0, 0.0, 9.0]);
}

#[test]
fn test_vector_special_cases() {
    // Zero vector
    let zero_vec = Matrix::zeros(3, 1);
    let vec = Matrix {
        data: vec![1.0, 2.0, 3.0],
        rows: 3,
        columns: 1,
    };

    // Dot with zero vector
    assert_eq!(zero_vec.dot_product(&vec).unwrap(), 0.0);

    // Cross with zero vector
    let cross = zero_vec.cross_product(&vec).unwrap();
    assert_eq!(cross.data, vec![0.0; 3]);

    // Normalization of zero vector should fail
    assert!(zero_vec.unit_vector().is_err());
}

#[test]
fn test_matrix_properties() {
    let m = test_matrix();
    let transposed = m.transpose();

    // (A^T)^T = A
    assert_eq!(transposed.transpose().data, m.data);

    // Test trace
    assert_eq!(m.trace().unwrap(), 5.0); // 1 + 4

    // Test determinant properties
    let det = m.determinant().unwrap();
    let scaled = &m * 2.0;
    assert_eq!(scaled.determinant().unwrap(), det * 4.0); // det(kA) = k^n det(A)
}

#[test]
fn test_assignment_operators() {
    let mut m1 = test_matrix();
    let m2 = Matrix::ones(2, 2);

    // AddAssign
    m1 += &m2;
    assert_eq!(m1.data, vec![2.0, 3.0, 4.0, 5.0]);

    // MulAssign scalar
    m1 *= 2.0;
    assert_eq!(m1.data, vec![4.0, 6.0, 8.0, 10.0]);

    // SubAssign
    m1 -= &m2;
    assert_eq!(m1.data, vec![3.0, 5.0, 7.0, 9.0]);
}

#[test]
fn test_error_handling() {
    // Non-square matrix for operations requiring square
    let non_square = Matrix {
        data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        rows: 2,
        columns: 3,
    };

    assert!(non_square.determinant().is_err());
    assert!(non_square.inverse().is_err());
    assert!(non_square.trace().is_err());

    // Dimension mismatches
    let m1 = Matrix::ones(2, 3);
    let m2 = Matrix::ones(3, 2);
    assert!(m1.addition(&m2).is_err());
    assert!(m1.hadamard_multiplication(&m2).is_err());
}

#[test]
fn test_apply_function() {
    let m = test_matrix();
    let squared = m.apply(|x| x * x);
    assert_eq!(squared.data, vec![1.0, 4.0, 9.0, 16.0]);

    let abs = Matrix {
        data: vec![-1.0, 2.0, -3.0, 4.0],
        rows: 2,
        columns: 2,
    }.apply(f64::abs);
    assert_eq!(abs.data, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_large_matrices() {
    // 4x4 matrix test
    let m = Matrix {
        data: vec![1.0, 2.0, 3.0, 4.0,
                   5.0, 6.0, 7.0, 8.0,
                   9.0, 10.0, 11.0, 12.0,
                   13.0, 14.0, 15.0, 16.0],
        rows: 4,
        columns: 4,
    };

    // Rank should be 2 (rows are linearly dependent)
    assert_eq!(m.rank(), 2);

    // Determinant should be 0 for singular matrix
    assert_eq!(m.determinant().unwrap(), 0.0);
}

#[test]
fn test_scalar_projection() {
    let v1 = Matrix {
        data: vec![1.0, 0.0],
        rows: 2,
        columns: 1,
    };
    let v2 = Matrix {
        data: vec![3.0, 4.0],
        rows: 2,
        columns: 1,
    };

    // Projection of (3,4) onto x-axis
    let proj = v2.scalar_projection_onto(&v1).unwrap();
    assert_relative_eq!(proj, 3.0, epsilon = 1e-10);

    // Vector projection
    let vec_proj = v2.vector_projection_onto(&v1).unwrap();
    assert_relative_eq!(vec_proj.data[0], 3.0, epsilon = 1e-10);
    assert_relative_eq!(vec_proj.data[1], 0.0, epsilon = 1e-10);
}