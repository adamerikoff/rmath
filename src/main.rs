use rmath::Vector;



fn main() {
    let v1 = Vector::<5>::zeros();
    let v2 = Vector::<5>::ones() * 2.2;

    println!("{:?}", v2 / 0.0);
}
