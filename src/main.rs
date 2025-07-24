use rmath::{mtx, Matrix};

fn main() {
    // Create 3x2 matrix (2 columns, 3 rows)
    let m = mtx![2, 3];

    println!("{}", m);
    /* Output:
    Matrix<3x2> [
      1.0000, 2.0000, 3.0000
      4.0000, 5.0000, 6.0000
    ]
    */
}