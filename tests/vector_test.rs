use approx::assert_relative_eq;
use rmath::*;

// ===== Test Helper Functions =====
fn assert_vectors_equal<const N: usize>(actual: Vector<N>, expected: Vector<N>) {
    for i in 0..N {
        assert_relative_eq!(actual[i], expected[i], epsilon = 1e-10);
    }
}

// ===== 2D Vector Tests =====
#[test]
fn test_2d_vector_components() {
    let v = Vector::new([1.0, 2.0]);
    assert_eq!(v.x(), 1.0);
    assert_eq!(v.y(), 2.0);
}

#[test]
fn test_2d_cross_product() {
    let v1 = Vector::new([1.0, 2.0]);
    let v2 = Vector::new([3.0, 4.0]);
    assert_relative_eq!(v1.cross(&v2), -2.0);

    // Test orthogonal vectors
    let i = Vector::new([1.0, 0.0]);
    let j = Vector::new([0.0, 1.0]);
    assert_relative_eq!(i.cross(&j), 1.0);

    // Test parallel vectors
    let v = Vector::new([2.0, 3.0]);
    let scaled_v = Vector::new([4.0, 6.0]);
    assert_relative_eq!(v.cross(&scaled_v), 0.0);
}

// ===== 3D Vector Tests =====
#[test]
fn test_3d_vector_components() {
    let v = Vector::new([1.0, 2.0, 3.0]);
    assert_eq!(v.x(), 1.0);
    assert_eq!(v.y(), 2.0);
    assert_eq!(v.z(), 3.0);
}

#[test]
fn test_3d_cross_product() {
    // Standard basis vectors
    let i = Vector::new([1.0, 0.0, 0.0]);
    let j = Vector::new([0.0, 1.0, 0.0]);
    let k = Vector::new([0.0, 0.0, 1.0]);

    assert_vectors_equal(i.cross(&j), k);
    assert_vectors_equal(j.cross(&k), i);
    assert_vectors_equal(k.cross(&i), j);

    // Anti-commutative property
    assert_vectors_equal(j.cross(&i), Vector::new([0.0, 0.0, -1.0]));

    // Test with arbitrary vectors
    let v1 = Vector::new([1.0, 2.0, 3.0]);
    let v2 = Vector::new([4.0, 5.0, 6.0]);
    let expected = Vector::new([-3.0, 6.0, -3.0]);
    assert_vectors_equal(v1.cross(&v2), expected);
}

// ===== Generic Vector Tests =====
#[test]
fn test_vector_construction() {
    // Test new()
    let v = Vector::new([1.0, 2.0, 3.0, 4.0]);
    assert_eq!(v[0], 1.0);
    assert_eq!(v[1], 2.0);
    assert_eq!(v[2], 3.0);
    assert_eq!(v[3], 4.0);

    // Test zeros()
    let zero_vec = Vector::<5>::zeros();
    for i in 0..5 {
        assert_eq!(zero_vec[i], 0.0);
    }

    // Test ones()
    let ones_vec = Vector::<4>::ones();
    for i in 0..4 {
        assert_eq!(ones_vec[i], 1.0);
    }
}

#[test]
fn test_component_access() {
    let v = Vector::new([10.0, 20.0, 30.0]);
    assert_eq!(v.component(0), 10.0);
    assert_eq!(v.component(1), 20.0);
    assert_eq!(v.component(2), 30.0);
}

#[test]
#[should_panic]
fn test_component_access_out_of_bounds() {
    let v = Vector::new([1.0, 2.0]);
    let _ = v.component(2); // Should panic for 2D vector
}

#[test]
fn test_magnitude() {
    // 2D vector
    let v2 = Vector::new([3.0, 4.0]);
    assert_relative_eq!(v2.magnitude(), 5.0);

    // 3D vector
    let v3 = Vector::new([1.0, 2.0, 2.0]);
    assert_relative_eq!(v3.magnitude(), 3.0);

    // 4D vector
    let v4 = Vector::new([1.0, 1.0, 1.0, 1.0]);
    assert_relative_eq!(v4.magnitude(), 2.0);
}

#[test]
fn test_dot_product() {
    // 2D case
    let v1 = Vector::new([1.0, 2.0]);
    let v2 = Vector::new([3.0, 4.0]);
    assert_relative_eq!(v1.dot(&v2), 11.0);

    // 3D case
    let v3 = Vector::new([1.0, 2.0, 3.0]);
    let v4 = Vector::new([4.0, 5.0, 6.0]);
    assert_relative_eq!(v3.dot(&v4), 32.0);

    // Orthogonal vectors
    let orth1 = Vector::new([1.0, 0.0]);
    let orth2 = Vector::new([0.0, 1.0]);
    assert_relative_eq!(orth1.dot(&orth2), 0.0);
}

// ===== Operator Overload Tests =====
#[test]
fn test_vector_addition() {
    let v1 = Vector::new([1.0, 2.0, 3.0]);
    let v2 = Vector::new([4.0, 5.0, 6.0]);
    let expected = Vector::new([5.0, 7.0, 9.0]);
    assert_vectors_equal(&v1 + &v2, expected);

    // Test commutative property
    assert_vectors_equal(&v2 + &v1, expected);

    // Test with zeros
    let zero = Vector::<3>::zeros();
    assert_vectors_equal(&v1 + &zero, v1);
}

#[test]
fn test_vector_subtraction() {
    let v1 = Vector::new([5.0, 7.0, 9.0]);
    let v2 = Vector::new([4.0, 5.0, 6.0]);
    let expected = Vector::new([1.0, 2.0, 3.0]);
    assert_vectors_equal(&v1 - &v2, expected);

    // Test with zeros
    let zero = Vector::<3>::zeros();
    assert_vectors_equal(&v1 - &zero, v1);
}

#[test]
fn test_hadamard_product() {
    let v1 = Vector::new([1.0, 2.0, 3.0]);
    let v2 = Vector::new([4.0, 5.0, 6.0]);
    let expected = Vector::new([4.0, 10.0, 18.0]);
    assert_vectors_equal(&v1 * &v2, expected);

    // Test with ones
    let ones = Vector::<3>::ones();
    assert_vectors_equal(&v1 * &ones, v1);
}

#[test]
fn test_scalar_multiplication() {
    let v = Vector::new([1.0, 2.0, 3.0]);
    let scalar = 2.0;
    let expected = Vector::new([2.0, 4.0, 6.0]);
    assert_vectors_equal(&v * scalar, expected);
    // Test with zero
    assert_vectors_equal(&v * 0.0, Vector::<3>::zeros());
}

#[test]
fn test_hadamard_division() {
    let v1 = Vector::new([4.0, 10.0, 18.0]);
    let v2 = Vector::new([1.0, 2.0, 3.0]);
    let expected = Vector::new([4.0, 5.0, 6.0]);
    assert_vectors_equal(&v1 / &v2, expected);
}

#[test]
fn test_scalar_division() {
    let v = Vector::new([2.0, 4.0, 6.0]);
    let scalar = 2.0;
    let expected = Vector::new([1.0, 2.0, 3.0]);
    assert_vectors_equal(&v / scalar, expected);
}

#[test]
fn test_division_by_zero_scalar() {
    let v = Vector::new([1.0, 2.0]);
    let r = &v / 0.0;
    assert_eq!(r, Vector::new([f64::INFINITY, f64::INFINITY]));
}

#[test]
fn test_hadamard_division_by_zero_vector() {
    let v1 = Vector::new([1.0, 2.0]);
    let v2 = Vector::new([1.0, 0.0]);
    let v3 = &v1 / &v2;
    assert_eq!(v3, Vector::new([1.0, f64::INFINITY]));
}

// ===== Indexing Tests =====
#[test]
fn test_indexing() {
    let v = Vector::new([10.0, 20.0, 30.0]);
    assert_eq!(v[0], 10.0);
    assert_eq!(v[1], 20.0);
    assert_eq!(v[2], 30.0);
}

#[test]
#[should_panic]
fn test_index_out_of_bounds() {
    let v = Vector::new([1.0, 2.0]);
    let _ = v[2];
}

// ===== From Slice Tests =====
#[test]
fn test_from_slice() {
    let slice = &[1.0, 2.0, 3.0][..];
    let v = Vector::<3>::from(slice);
    assert_eq!(v[0], 1.0);
    assert_eq!(v[1], 2.0);
    assert_eq!(v[2], 3.0);
}

#[test]
#[should_panic(expected = "Cannot create Vector<3> from slice of length 2")]
fn test_from_slice_wrong_length() {
    let slice = &[1.0, 2.0][..];
    let _ = Vector::<3>::from(slice);
}

// ===== Complex Operation Tests =====
#[test]
fn test_vector_operations_chain() {
    // Test a complex chain of operations: (v1 + v2) * v3 - v4 / scalar
    let v1 = Vector::new([1.0, 2.0, 3.0]);
    let v2 = Vector::new([4.0, 5.0, 6.0]);
    let v3 = Vector::new([2.0, 3.0, 4.0]);
    let v4 = Vector::new([10.0, 20.0, 30.0]);
    let scalar = 2.0;

    let v5 = &(&v1 + &v2) * &v3;
    let v6 = &v4 / scalar;
    let result = &v5 - &v6;
    let expected = Vector::new([5.0 * 2.0 - 5.0, 7.0 * 3.0 - 10.0, 9.0 * 4.0 - 15.0]);
    assert_vectors_equal(result, expected);
}

#[test]
fn test_geometric_properties() {
    // Test that cross product is orthogonal to both vectors
    let v1 = Vector::new([1.0, 2.0, 3.0]);
    let v2 = Vector::new([4.0, 5.0, 6.0]);
    let cross = v1.cross(&v2);

    // Dot product with original vectors should be zero (orthogonal)
    assert_relative_eq!(cross.dot(&v1), 0.0);
    assert_relative_eq!(cross.dot(&v2), 0.0);

    // Test magnitude of cross product equals area of parallelogram
    let area = v1.magnitude() * v2.magnitude() * v1.normalize().dot(&v2.normalize()).acos().sin();
    assert_relative_eq!(cross.magnitude(), area, epsilon = 1e-10);
}

#[test]
fn test_vector_normalization() {
    let v = Vector::new([3.0, 4.0]);
    let normalized = &v / v.magnitude();
    assert_relative_eq!(normalized.magnitude(), 1.0);

    // Test that normalized vector points in same direction
    let dot_product = normalized.dot(&(&v / v.magnitude()));
    assert_relative_eq!(dot_product, 1.0);
}
