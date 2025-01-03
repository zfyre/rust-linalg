use std::ops::{Add, Div, Mul, Sub};

trait DType {
    fn zero() -> Self;
    fn one() -> Self;
    // fn inf() -> Self;
}
macro_rules! impl_DType {
    ($($t:ty)*) => {$(
            impl DType for $t {
                fn zero() -> Self {
                    0 as $t
                }
                fn one() -> Self {
                    1 as $t
                }
            }
        )*
    };
}
// Implement dtype for all primitive numerical types
impl_DType!(usize u8 u16 u32 u64 i8 i16 i32 i64 f32 f64);

#[derive(Debug, Clone)]
struct Matrix<T>
{
    data: Vec<Vec<T>>,
    shape: (usize, usize),
    rank: (usize, bool),
    trace: Option<T>,
}

impl<T> Matrix<T> 
where
T: Clone 
    + std::fmt::Display 
    + Copy 
    + std::iter::Sum
    + Div<Output = T>
    + Mul<T, Output = T> 
    + Add<Output = T>
    + Sub<Output = T>
    + Div<Output = T>
    + std::cmp::PartialEq
    + DType

// Vec<T>: FromIterator<()>
{
    fn new(shape: (usize, usize), data: &[T]) -> Self {
        let total_ele = shape.0  * shape.1;
        if shape.0 == 0 || shape.1 == 0 {
            panic!("shape {:?} cannot be of 0 size", total_ele);
        }
        if data.len() != total_ele {
            panic!("expected {} elements for shape {:?}, found {}", total_ele, shape, data.len());
        }
        // let mut matrix: Vec<Vec<T>> = vec![vec![data[0].clone(); shape.0]; shape.1];
        // let mut k = 0;
        // for i in 0..shape.0 {
        //     for j in 0..shape.1 {
        //         matrix[i][j] = data[k].clone();
        //         k += 1;
        //     }
        // }
        let matrix: Vec<Vec<T>> = data.chunks(shape.1).map(|row| row.to_vec()).collect(); // Done in one line!!
        Self {
            data: matrix,
            shape: shape,
            rank: (usize::zero(), false),
            trace: None
        }
    }
    fn identity(shape: usize) -> Self {
        if shape == 0 {
            panic!("shape {:?} cannot be of 0 size", shape);
        }
        let mut mat = vec![vec![T::zero(); shape]; shape];
        for i in 0..shape {
            mat[i][i] = T::one();
        }
        Self {
            data: mat,
            shape: (shape, shape),
            rank: (usize::zero(), false),
            trace: None
        }
    }
    fn zeros(shape: (usize, usize)) -> Self {
        if shape.0 == 0 || shape.1 == 0 {
            panic!("shape {:?} cannot be of 0 size", shape);
        }
        let mat = vec![vec![T::zero(); shape.1]; shape.0];
        Self {
            data: mat,
            shape: shape,
            rank: (usize::zero(), false),
            trace: None
        }
    }

    fn trace(&mut self) -> T {
        match self.trace {
            Some(x) => x,
            None => {
                let (n, m) = self.shape;
                let res = (0..n).map(|i| self.data[i][i]).sum();
                self.trace = Some(res);
                res
            }
        }
    }

    fn print(&self) {

        println!("Matrix with shape: {:?} & rank: {:?}", self.shape, self.rank);
        self.data.iter().for_each(|row | {
            print!("[");
            row.iter().for_each(|e| {
                print!(" {} ", e);
            });
            print!("]\n");
        });
    }

    fn mm(&self, other: &Self) -> Self {
        // (n x k) @ (k x m) -> (n x m)
        let (r1,c1) = self.shape;
        let (r2,c2) = other.shape;

        if c1 != r2 {
            panic!(
                "Matrix Mul not possible: shapes {:?} and {:?} do not fit (n x k) @ (k x m)",
                self.shape, other.shape
            );
        }else {
            // println!("r1 = {}, c1 = {}, r2 = {}, c2 = {}", r1, c1, r2, c2);
            let mut res: Vec<Vec<T>> = vec![vec![T::zero(); c2]; r1];
            for i in 0..r1 {
                for j in 0..c2 {
                    for l in 0..c1 {
                        // println!("i: {}, j: {}, l: {}", i, j, l);
                        res[i][j] = res[i][j] + self.data[i][l] * other.data[l][j];
                    }
                }
            }
            
            Self {
                data: res,
                shape: (r1, c2),
                rank: (usize::zero(), false),
                trace: None
            }
        }
    }

    fn t(&self) -> Self {
        let (n, m) = self.shape;
        
        // let mut res: Vec<Vec<T>> = vec![vec![T::zero(); n]; m];
        // for i in 0..n {
        //     for j in 0..m {
        //         res[i][j] = self.data[j][i];
        //     }
        // }
        // OR
        let res = (0..m)
                            .map(|i| (0..n).map(|j| self.data[j][i]).collect())
                            .collect();

        Self {
            data: res,
            shape: (m, n),
            rank: self.rank,
            trace: None
        }
    }
    fn concat(&self, other: &Self, dim: usize) -> Self {
        let (r1, c1) = self.shape;
        let (r2, c2) = other.shape;

        if dim > 1 {
            panic!("dim {:?} should be either 0 or 1", dim);
        }
        if (dim == 0 && c1 != c2) || (dim == 1 && r1 != r2) {
            panic!("shapes {:?} & {:?} cannot be concanated along dim {:?}",self.shape, other.shape, dim);
        }

        let mut mat = self.data.clone();
        let matshape;
        if dim == 1 {
            for i in 0..self.shape.0 {
                mat[i].extend(other.data[i].iter());
            }
            matshape = (r1, c1 + c2);
        }
        else {
            mat.extend(other.data.clone().into_iter());
            matshape = (r1 + r2, c1);
        }
        Self {
            data: mat,
            shape: matshape,
            rank: (usize::zero(), false),
            trace: None
        }
    }

    fn gauss_elm(&self) -> (Self, Vec<usize>) {

        let (n, m) = self.shape;

        let mut augmented = self.concat(&Self::identity(n), 1);
        
        let mut pivot_colidx = Vec::<usize>::new();
        let mut rank: usize = 0;
        
        let mut col: usize = 0;
        let mut fin = false;
        
        for row in 0..n {

            let mut pivot = augmented.data[row][col];

            while pivot == T::zero() { 
                let mut swapped = false;
                for i in row+1..n {
                    if augmented.data[i][col] != T::zero() {
                        augmented.data.swap(i, row);
                        swapped = true;
                        break;
                    }
                }
                if !swapped {
                    col += 1;
                    if col >= m {
                        fin = true;
                        break;
                    }
                    pivot = augmented.data[row][col];
                }
            }
            if fin { break; }
            
            // Updating the rank and pivot column index
            rank += 1;
            pivot_colidx.push(col);

            // Making the pivot 1
            for i in (col)..(m+n) {
                augmented.data[row][i] = augmented.data[row][i] / pivot;
            }

            // Making the other elements in the column zero
            for i in (row+1)..n {
                let factor = augmented.data[i][col];
                for j in (col)..(m+n) {
                    augmented.data[i][j] = augmented.data[i][j] - factor * augmented.data[row][j];
                }
            }
        }
        // Making Row Reduced Echelon Form
        for row in (0..rank).rev() {
            for i in 0..row {
                let factor = augmented.data[i][pivot_colidx[row]];
                for j in 0..m+n {
                    augmented.data[i][j] = augmented.data[i][j] - factor * augmented.data[row][j];
                }
            }

        }
        augmented.rank = (rank, fin);
        return (augmented, pivot_colidx);
    }



}



impl<T> Add for Matrix<T> 
where T: std::ops::Add<Output = T> + Clone
{
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {

        if self.shape != rhs.shape {
            panic!(
                "Matrix addition not possible: shapes {:?} and {:?} do not match",
                self.shape, rhs.shape
            );
        }

        let out = self.data.iter().zip(rhs.data.iter()).map(|(row1, row2)| {
            row1.iter().zip(row2.iter()).map(|(a, b)| (*a).clone() + (*b).clone()).collect()
        }).collect();

        Self {
            data : out,
            shape: self.shape,
            rank: (usize::zero(), false),
            trace: None
        }
    }
}

// for Matrix * C

impl<T> Mul<T> for Matrix<T> // element wise multiplicaiton!
where 
    T: Mul<Output = T> + Clone, 
{
    type Output = Matrix<T>;
    fn mul(self, rhs: T) -> Self::Output {

        let out = self.data.iter().map(|row| {
            row.iter().map(|a| (*a).clone() * rhs.clone()).collect()
        }).collect();

        Matrix {
            data : out,
            shape: self.shape,
            rank: (usize::zero(), false),
            trace: None
        }
    }
}

impl<T> Sub for Matrix<T> 
where T: std::ops::Sub<Output = T> + Clone
{
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {

        if self.shape != rhs.shape {
            panic!(
                "Matrix addition not possible: shapes {:?} and {:?} do not match",
                self.shape, rhs.shape
            );
        }

        let out = self.data.iter().zip(rhs.data.iter()).map(|(row1, row2)| {
            row1.iter().zip(row2.iter()).map(|(a, b)| (*a).clone() - (*b).clone()).collect()
        }).collect();

        Self {
            data : out,
            shape: self.shape,
            rank: (usize::zero(), false),
            trace: None
        }
    }
}

impl<T> Div for Matrix<T> 
where T: std::ops::Div<Output = T> + Clone
{
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {

        if self.shape != rhs.shape {
            panic!(
                "Matrix addition not possible: shapes {:?} and {:?} do not match",
                self.shape, rhs.shape
            );
        }

        let out = self.data.iter().zip(rhs.data.iter()).map(|(row1, row2)| {
            row1.iter().zip(row2.iter()).map(|(a, b)| (*a).clone() / (*b).clone()).collect()
        }).collect();

        Self {
            data : out,
            shape: self.shape,
            rank: (usize::zero(), false),
            trace: None
        }
    }
}

fn main() { 
    println!("Hello World!");

    let m1 = Matrix::new((3, 2), &[1, 2, 3, 4, 5, 6]);
    let m2 = Matrix::new((3, 2), &[2, 3, 4, 5, 6, 7]);
    m1.print();

    let m3 = m1.clone() + m2.clone();
    let m4 = m1.clone() - m2.clone();

    m3.print();
    m4.print();


    let m5 = m3 + m1;
    let m6 = m5.t();
    let mut m7 = m5.mm(&m6);
    m5.print();
    m6.print();
    m7.print();    

    println!("Trace of m7: {}", m7.trace());

}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_vector_concat() {
        let mut a= vec![[1], [2], [3]];
        let b= vec![[4], [5], [6]];
        println!("a: {:?}, b: {:?}", a, b);
        a.extend(b.iter());
        println!("a: {:?}, b: {:?}", a, b);
    }
    #[test]
    fn check_matrix_const_mul(){
        let m1 = Matrix::new((3, 2), &[1, 2, 3, 4, 5, 6]);
        m1.print();
        let m2 = m1 * 2;
        m2.print();
    }
    #[test]
    fn check_matrix_concat() {      
        let mut m1 = Matrix::new((3, 2), &[1, 2, 3, 4, 5, 6]);
        let m2 = Matrix::new((3, 2), &[2, 3, 4, 5, 6, 7]);

        let m3 = m1.concat(&m2, 1);
        let m4 = m1.concat(&m2, 0);

        m3.print();
        m4.print();
    }
    #[test]
    fn check_gauss_elm() {
        let mut m1: Matrix<f64> = Matrix::new((3, 4), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.0, 9.0, 10.0, 11.0, 12.0]);
        m1.print();
        let (g, col_idx) = m1.gauss_elm();
        g.print();
        println!("pivot columns {:?}", col_idx);
    }
    #[test]
    fn check_inverse() {
        let mut m1: Matrix<f64> = Matrix::new((3, 3), &[1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0]);
        m1.print();
        let (g, col_idx) = m1.gauss_elm();
        g.print();
        println!("pivot columns {:?}", col_idx);

        let m2 = m1.mm(&g);
        m2.print();
    }
    #[test]
    fn check_chunk() {
        // TODO: Implement chunking for matrix along axis/ dim 0 and 1 as axis = (row, col)
        // return a vector of matrices
    }
}
