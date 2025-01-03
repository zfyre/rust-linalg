# Linear Algebra Library in Rust

## Features

- **Matrix Operations**: Perform basic matrix operations such as addition, subtraction, and multiplication.
- **Vector Operations**: Support for vector addition, subtraction, dot product, and cross product.
- **Determinant Calculation**: Compute the determinant of a matrix.
- **Inverse Matrix**: Find the inverse of a matrix if it exists.
- **Eigenvalues and Eigenvectors**: Calculate eigenvalues and eigenvectors for square matrices.
- **LU Decomposition**: Perform LU decomposition of matrices.
- **QR Decomposition**: Perform QR decomposition of matrices.

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
linalg = "0.1.0"
```

## Usage

```rust
extern crate linalg;

use linalg::matrix::Matrix;
use linalg::vector::Vector;

fn main() {
    // Example usage
    let a = Matrix::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    let b = Matrix::new(vec![vec![5.0, 6.0], vec![7.0, 8.0]]);
    
    let c = a + b;
    println!("Matrix C: {:?}", c);
}
```

## License

This project is licensed under the MIT License.