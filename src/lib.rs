// n次正方行列の固有値(Eigenvalue)・固有ベクトル(Eigenvector)を求める
// Jacobian法，べき乗法，逆べき乗法
//
// 参考
// Jacobi法-[物理のかぎしっぽ]
//     http://hooktail.org/computer/index.php?Jacobi%CB%A1
//
// Jacobi法による固有値解析プログラム
//     http://www-in.aut.ac.jp/~minemura/pub/Csimu/C/Jacobi.html
//
// 5 Jacobi法
//    http://nalab.mind.meiji.ac.jp/~mk/labo/text/eigenvalues-add/node19.html
//
// ちゃんと動いたら行列の扱い方を変える（メモリアクセスの効率の関係）．
// 今は二次元のベクタで表現してるけど，構造体にまとめて一次元ベクタとする．
// NxN行列なら i*N + j みたいな感じ？

pub type Matrix<T> = Vec<Vec<T>>;

const EPSILON: f64 = 1e-10;  // 収束条件
const LOOP_MAX: usize = 500;  // いくらループしてもここまで
const FRAC_1_SQRT_2: f64 = std::f64::consts::FRAC_1_SQRT_2;  // 1 / √2


/// Jacobi法により実対称行列の固有値・固有ベクトルを求める．
/// return: (eigenvalue[λ1, λ2...], eigenvector[x1, x2...])
#[inline]
pub fn jacobi(mut a: Matrix<f64>) -> (Vec<f64>, Matrix<f64>) {
    let n = a.len();  // 正方行列のサイズ
    let mut lambda = Vec::with_capacity(n);  // 固有値
    let mut r = Vec::with_capacity(n);  // 直交行列（最終的に固有ベクトルとなる）

    // 直交行列を初期化（単位行列）
    for i in 0..n {
        let mut line = Vec::with_capacity(n);
        for j in 0..n {
            line.push( if i == j {1.0} else {0.0} );
        }
        r.push(line);
    }
    
    // 反復計算
    for _ in 0..LOOP_MAX {
        let (a_max, [p, q]) = search_max_index(&a);

        // 対角化が終わったことを判断するには，非対角行列成分の最大値を見れば良い．
        if a_max <= EPSILON {
            break;
        } else {
            // 更新途中で値が変わらないように変数に確保しておく．
            let a_pp = a[p][p];
            let a_qq = a[q][q];
            let a_pq = a[p][q];

            // 回転角
            let [sin, cos, sin_pow2, cos_pow2] = calc_sin_cos(a_pp, a_qq, a_pq);

            // ---------- 実対称行列Aを対角化 ---------- //
            // pp, qq, pq, qp更新
            let tmp = 2.0 * a_pq * sin * cos;
            a[p][p] = a_pp * cos_pow2 + a_qq * sin_pow2 + tmp;
            a[q][q] = a_pp * sin_pow2 + a_qq * cos_pow2 - tmp;
            // ここが0になるようにthetaを計算しているので，計算せずに0を代入してしまう．
            a[p][q] = 0.0;
            a[q][p] = 0.0;

            // pj, qj, ip, iq 更新
            for i in 0..n {
                if (i != p) && (i != q) {
                    let a_ip = a[i][p];
                    let a_iq = a[i][q];
                    a[i][p] =  a_ip * cos + a_iq * sin;
                    a[i][q] = -a_ip * sin + a_iq * cos;
                    // 対称行列なので入る値は同じ
                    a[p][i] = a[i][p];
                    a[q][i] = a[i][q];
                }
            }

            // ------------ 直交行列Rを更新 ------------ //
            for i in 0..n {
                let r_ip = r[i][p];
                let r_iq = r[i][q];
                r[i][p] =  r_ip * cos + r_iq * sin;
                r[i][q] = -r_ip * sin + r_iq * cos;
            }
        }
    }

    // 固有値を取り出す
    for i in 0..n {
        lambda.push(a[i][i]);
    }

    // 直交行列の縦ベクトルが固有ベクトルになるので，行ベクトルに入れ直す．
    let mut x = Vec::with_capacity(n);
    for i in 0..n {
        let mut line = Vec::with_capacity(n);
        for j in 0..n {
            line.push(r[j][i]);  // 縦ベクトル読み出し
        }
        x.push(line);
    }

    (lambda, x)
}

/// 回転平面を張るための軸，p, qを見つける．
/// 非対角行列成分の中から絶対値が最大となる値の場所を探す．
/// return: (max_num, [p, q])
#[inline]
fn search_max_index(a: &Matrix<f64>) -> (f64, [usize; 2]) {
    let mut p_q = [0, 0];
    let mut max_num = 0.0;

    let n = a.len();
    for i in 0..n {
        for j in 0..n {
            // 非対角成分のみ探索
            if i != j {
                // 絶対値が最大なら更新
                let val = a[i][j].abs();
                if val > max_num {
                    max_num = val;
                    p_q = [i, j];
                }
            }
        }
    }
    (max_num, p_q)
}

/// a_pq, a_qpが0となるような回転角を求める．
/// return: [sinθ, cosθ, sinθ^2, cosθ^2]
#[inline]
fn calc_sin_cos(a_pp: f64, a_qq: f64, a_pq: f64) -> [f64; 4] {
    let tmp = a_pp - a_qq;
    let (sin, cos) = if tmp.abs() < EPSILON {
        (FRAC_1_SQRT_2, FRAC_1_SQRT_2)
    } else {
        let theta = 0.5 * ( (2.0 * a_pq) / tmp ).atan();
        ( theta.sin(), theta.cos() )
    };

    [sin, cos, sin*sin, cos*cos]
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jacobi() {
        // 実対称行列
        let a = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2.0, 5.0, 4.0, 0.0],
            vec![3.0, 4.0, 1.0, 1.0],
            vec![4.0, 0.0, 1.0, 2.0],
        ];

        let (eigenval, eigenvec) = jacobi(a);
        println!("eigenvalue: {:?}", eigenval);

        println!("eigenvector: ");
        for i in 0..eigenvec.len() {
            println!("{:?}", eigenvec[i]);
        }

        //assert!(false);
    }
}
