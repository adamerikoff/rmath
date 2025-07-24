use std::{
    fmt,
    ops::{Add, Div, Index, Mul, Sub},
};

use rand::Rng;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vector<const N: usize> {
    components: [f64; N],
}

impl<const N: usize> fmt::Display for Vector<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;

        for (i, component) in self.components.iter().enumerate() {
            if i > 0 {
                writeln!(f, "")?;
                write!(f, " ")?;
            }
            write!(f, "{:8.4}", component)?;
        }
        if N > 0 {
            writeln!(f)?;
        }
        write!(f, "]")
    }
}

impl Vector<2> {
    pub fn x(&self) -> f64 {
        self.components[0]
    }

    pub fn y(&self) -> f64 {
        self.components[1]
    }

    pub fn cross(&self, other: &Self) -> f64 {
        self.x() * other.y() - self.y() * other.x()
    }
}

impl Vector<3> {
    pub fn x(&self) -> f64 {
        self.components[0]
    }
    pub fn y(&self) -> f64 {
        self.components[1]
    }
    pub fn z(&self) -> f64 {
        self.components[2]
    }

    pub fn cross(&self, other: &Self) -> Self {
        Self::new([
            self.y() * other.z() - self.z() * other.y(),
            self.z() * other.x() - self.x() * other.z(),
            self.x() * other.y() - self.y() * other.x(),
        ])
    }
}

impl<const N: usize> Vector<N> {
    pub fn new(components: [f64; N]) -> Self {
        Self { components }
    }

    pub fn zeros() -> Self {
        Self {
            components: [0.0; N],
        }
    }

    pub fn ones() -> Self {
        Self {
            components: [1.0; N],
        }
    }

    pub fn randomized() -> Self {
        let mut arr = [0.0; N];
        let mut rng = rand::rng();
        for item in &mut arr {
            *item = rng.random_range(0.0..1.0);
        }
        Self { components: arr }
    }

    pub fn apply<F>(&self, fun: F) -> Self
    where
        F: Fn(f64) -> f64,
    {
        Self {
            components: std::array::from_fn(|i| fun(self.components[i])),
        }
    }

    pub fn component(&self, index: usize) -> f64 {
        self.components[index]
    }

    pub fn magnitude(&self) -> f64 {
        self.components
            .iter()
            .fold(0.0, |acc, x| acc + x * x)
            .sqrt()
    }

    pub fn normalize(&self) -> Self {
        self.divide(self.magnitude())
    }

    pub fn add_vectors(&self, other: &Self) -> Self {
        Self {
            components: std::array::from_fn(|i| self.components[i] + other.components[i]),
        }
    }

    pub fn subtract_vectors(&self, other: &Self) -> Self {
        Self {
            components: std::array::from_fn(|i| self.components[i] - other.components[i]),
        }
    }

    pub fn hadamard_product(&self, other: &Self) -> Self {
        Self {
            components: std::array::from_fn(|i| self.components[i] * other.components[i]),
        }
    }

    pub fn hadamard_division(&self, other: &Self) -> Self {
        Self {
            components: std::array::from_fn(|i| self.components[i] / other.components[i]),
        }
    }

    pub fn scale(&self, scalar: f64) -> Self {
        Self {
            components: std::array::from_fn(|i| self.components[i] * scalar),
        }
    }

    pub fn divide(&self, scalar: f64) -> Self {
        Self {
            components: std::array::from_fn(|i| self.components[i] / scalar),
        }
    }

    pub fn dot(&self, other: &Self) -> f64 {
        self.components
            .iter()
            .zip(other.components.iter())
            .map(|(a, b)| a * b)
            .sum()
    }
}

impl<const N: usize> Add for &Vector<N> {
    type Output = Vector<N>;
    fn add(self, other: Self) -> Vector<N> {
        self.add_vectors(other)
    }
}

impl<const N: usize> Sub for &Vector<N> {
    type Output = Vector<N>;
    fn sub(self, other: Self) -> Vector<N> {
        self.subtract_vectors(other)
    }
}

impl<const N: usize> Mul for &Vector<N> {
    type Output = Vector<N>;
    fn mul(self, other: Self) -> Vector<N> {
        self.hadamard_product(other)
    }
}

impl<const N: usize> Mul<f64> for &Vector<N> {
    type Output = Vector<N>;
    fn mul(self, scalar: f64) -> Vector<N> {
        self.scale(scalar)
    }
}

impl<const N: usize> Div for &Vector<N> {
    type Output = Vector<N>;
    fn div(self, other: Self) -> Vector<N> {
        self.hadamard_division(other)
    }
}

impl<const N: usize> Div<f64> for &Vector<N> {
    type Output = Vector<N>;
    fn div(self, scalar: f64) -> Vector<N> {
        self.divide(scalar)
    }
}

impl<const N: usize> Index<usize> for Vector<N> {
    type Output = f64;
    fn index(&self, index: usize) -> &f64 {
        &self.components[index]
    }
}

impl<const N: usize> From<&[f64]> for Vector<N> {
    fn from(slice: &[f64]) -> Self {
        if slice.len() != N {
            panic!(
                "Cannot create Vector<{}> from slice of length {}",
                N,
                slice.len()
            );
        }
        let mut components = [0.0; N];
        components.copy_from_slice(slice);
        Vector { components }
    }
}

impl<const N: usize> From<&[f64; N]> for Vector<N> {
    fn from(arr: &[f64; N]) -> Self {
        Vector { components: *arr }
    }
}
