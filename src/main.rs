use rmath::Vector;

fn main() {
    // Create a 5-dimensional vector with all components set to 1.0, then scale by 2.3
    let v2 = Vector::<5>::ones() * 2.3;
    println!("v2: {}", v2);

    // Apply the cosine function to each component of v2
    // We use f64::cos as the components are f64, and it's part of the standard library
    let v3 = v2.apply(f64::cos);
    println!("v3 (cosine applied): {}", v3);

    // Demonstrate division by zero for floating-point numbers.
    // This will result in `inf` (infinity) for each component.
    println!("v2 / 0.0: {}", v2 / 0.0);
}