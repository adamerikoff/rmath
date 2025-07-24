use rmath::*;

// Helper function to create a matrix with specific data for testing
fn create_test_matrix<const N: usize>(rows: usize, columns: usize, data: [f64; N]) -> Matrix<N> {
    let mut matrix = Matrix::zeros(rows, columns);
    matrix.set_data(data);
    matrix
}

#[test]
fn test_new_matrix() {
    // Test successful creation of a 2x2 matrix
    let matrix = Matrix::<4>::new(2, 2);
    assert_eq!(matrix.rows(), 2);
    assert_eq!(matrix.columns(), 2);
    // Data is random, so we can't assert specific values, but we can check length
    assert_eq!(matrix.size(), 4);

    // Test creation of a 3x1 matrix
    let matrix_3x1 = Matrix::<3>::new(3, 1);
    assert_eq!(matrix_3x1.rows(), 3);
    assert_eq!(matrix_3x1.columns(), 1);
    assert_eq!(matrix_3x1.size(), 3);
}

#[test]
#[should_panic(expected = "Matrix dimensions 2x3 don't match storage for 4 elements")]
fn test_new_matrix_panic_dimensions() {
    // Test panic when dimensions don't match N
    let _ = Matrix::<4>::new(2, 3);
}

#[test]
fn test_zeros_matrix() {
    let matrix = Matrix::<9>::zeros(3, 3);
    assert_eq!(matrix.rows(), 3);
    assert_eq!(matrix.columns(), 3);
    // All elements should be 0.0
    for &val in matrix.get_data().iter() {
        assert_eq!(val, 0.0);
    }
}

#[test]
fn test_ones_matrix() {
    let matrix = Matrix::<4>::ones(2, 2);
    assert_eq!(matrix.rows(), 2);
    assert_eq!(matrix.columns(), 2);
    // All elements should be 1.0
    for &val in matrix.get_data().iter() {
        assert_eq!(val, 1.0);
    }
}

#[test]
fn test_get_set() {
    let mut matrix = Matrix::<4>::zeros(2, 2); // Start with a zero matrix

    // Set a value
    matrix.set(0, 0, 10.0);
    assert_eq!(matrix.get(0, 0), 10.0);

    // Set another value
    matrix.set(1, 1, 20.0);
    assert_eq!(matrix.get(1, 1), 20.0);

    // Check other values are still zero
    assert_eq!(matrix.get(0, 1), 0.0);
    assert_eq!(matrix.get(1, 0), 0.0);

    // Test setting with different numeric types
    matrix.set(0, 1, 5); // int
    assert_eq!(matrix.get(0, 1), 5.0);
    matrix.set(1, 0, 3.14f32); // f32
    assert_eq!(matrix.get(1, 0), 3.14f64);
}

#[test]
#[should_panic]
fn test_get_out_of_bounds() {
    let matrix = Matrix::<4>::zeros(2, 2);
    // Attempt to access out of bounds, should panic
    let _ = matrix.get(2, 0);
}

#[test]
#[should_panic]
fn test_set_out_of_bounds() {
    let mut matrix = Matrix::<4>::zeros(2, 2);
    // Attempt to set out of bounds, should panic
    println!("{}", matrix);
    matrix.set(0, 2, 1.0);
    println!("{}", matrix);
}

#[test]
fn test_add_matrix() {
    let m1 = create_test_matrix(2, 2, [1.0, 2.0, 3.0, 4.0]);
    let m2 = create_test_matrix(2, 2, [5.0, 6.0, 7.0, 8.0]);
    let result = m1.add_matrix(&m2);
    let expected = create_test_matrix(2, 2, [6.0, 8.0, 10.0, 12.0]);
    assert_eq!(result, expected);
}

#[test]
fn test_add_scalar() {
    let m = create_test_matrix(2, 2, [1.0, 2.0, 3.0, 4.0]);
    let result = m.add_scalar(10.0);
    let expected = create_test_matrix(2, 2, [11.0, 12.0, 13.0, 14.0]);
    assert_eq!(result, expected);

    // Test with integer scalar
    let result_int = m.add_scalar(5);
    let expected_int = create_test_matrix(2, 2, [6.0, 7.0, 8.0, 9.0]);
    assert_eq!(result_int, expected_int);
}

#[test]
fn test_sub_matrix() {
    let m1 = create_test_matrix(2, 2, [10.0, 20.0, 30.0, 40.0]);
    let m2 = create_test_matrix(2, 2, [1.0, 2.0, 3.0, 4.0]);
    let result = m1.sub_matrix(&m2);
    let expected = create_test_matrix(2, 2, [9.0, 18.0, 27.0, 36.0]);
    assert_eq!(result, expected);
}

#[test]
fn test_sub_scalar() {
    let m = create_test_matrix(2, 2, [10.0, 20.0, 30.0, 40.0]);
    let result = m.sub_scalar(5.0);
    let expected = create_test_matrix(2, 2, [5.0, 15.0, 25.0, 35.0]);
    assert_eq!(result, expected);

    // Test with integer scalar
    let result_int = m.sub_scalar(10);
    let expected_int = create_test_matrix(2, 2, [0.0, 10.0, 20.0, 30.0]);
    assert_eq!(result_int, expected_int);
}

#[test]
fn test_hadamard_product() {
    let m1 = create_test_matrix(2, 2, [1.0, 2.0, 3.0, 4.0]);
    let m2 = create_test_matrix(2, 2, [2.0, 3.0, 4.0, 5.0]);
    let result = m1.hadamard_product(&m2);
    let expected = create_test_matrix(2, 2, [2.0, 6.0, 12.0, 20.0]);
    assert_eq!(result, expected);
}

#[test]
fn test_hadamard_division() {
    let m1 = create_test_matrix(2, 2, [10.0, 20.0, 30.0, 40.0]);
    let m2 = create_test_matrix(2, 2, [2.0, 4.0, 5.0, 8.0]);
    let result = m1.hadamard_division(&m2);
    let expected = create_test_matrix(2, 2, [5.0, 5.0, 6.0, 5.0]);
    assert_eq!(result, expected);
}

#[test]
fn test_mul_scalar() {
    let m = create_test_matrix(2, 2, [1.0, 2.0, 3.0, 4.0]);
    let result = m.mul_scalar(2.0);
    let expected = create_test_matrix(2, 2, [2.0, 4.0, 6.0, 8.0]);
    assert_eq!(result, expected);

    // Test with integer scalar
    let result_int = m.mul_scalar(3);
    let expected_int = create_test_matrix(2, 2, [3.0, 6.0, 9.0, 12.0]);
    assert_eq!(result_int, expected_int);
}

#[test]
fn test_div_scalar() {
    let m = create_test_matrix(2, 2, [10.0, 20.0, 30.0, 40.0]);
    let result = m.div_scalar(5.0);
    let expected = create_test_matrix(2, 2, [2.0, 4.0, 6.0, 8.0]);
    assert_eq!(result, expected);

    // Test with integer scalar
    let result_int = m.div_scalar(10);
    let expected_int = create_test_matrix(2, 2, [1.0, 2.0, 3.0, 4.0]);
    assert_eq!(result_int, expected_int);
}

#[test]
fn test_matrix_multiplication_2x2_by_2x2() {
    let m1 = create_test_matrix(2, 2, [1.0, 2.0, 3.0, 4.0]);
    let m2 = create_test_matrix(2, 2, [5.0, 6.0, 7.0, 8.0]);
    let result = m1.mul(&m2);
    // Expected:
    // (1*5 + 2*7) = 5 + 14 = 19
    // (1*6 + 2*8) = 6 + 16 = 22
    // (3*5 + 4*7) = 15 + 28 = 43
    // (3*6 + 4*8) = 18 + 32 = 50
    let expected = create_test_matrix(2, 2, [19.0, 22.0, 43.0, 50.0]);
    assert_eq!(result, expected);
}

// #[test]
// fn test_matrix_multiplication_2x3_by_3x2() {
//     let m1 = create_test_matrix(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
//     let m2 = create_test_matrix(3, 2, [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
//     let result = m1.mul(&m2);
//     // Expected result is a 2x2 matrix
//     // [ (1*7 + 2*9 + 3*11), (1*8 + 2*10 + 3*12) ]
//     // [ (4*7 + 5*9 + 6*11), (4*8 + 5*10 + 6*12) ]
//     //
//     // [ (7 + 18 + 33), (8 + 20 + 36) ]
//     // [ (28 + 45 + 66), (32 + 50 + 72) ]
//     //
//     // [ 58, 64 ]
//     // [ 139, 154 ]
//     let expected = create_test_matrix(2, 2, [58.0, 64.0, 139.0, 154.0]);
//     assert_eq!(result, expected);
// }

#[test]
fn test_mtx_macro() {
    let matrix = mtx!(2, 2); // This should create a Matrix<4>
    assert_eq!(matrix.rows(), 2);
    assert_eq!(matrix.columns(), 2);
    assert_eq!(matrix.size(), 4);
}

#[test]
#[should_panic(expected = "Matrix dimensions 1x5 don't match storage for 4 elements")]
fn test_mtx_macro_panic_dimensions() {
    // This should panic because 1 * 5 = 5, but N is 4
    let _ = mtx!(1, 5);
}
