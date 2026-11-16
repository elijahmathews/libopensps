#include <cstdlib>
#include <hip/hip_runtime.h>
#include <stdint.h>



template <typename T>
inline __device__ void kahan_add(T term, T &sum, T &c) {
    T y = term - c;
    T t = sum + y;
    c = (t - sum) - y;
    sum = t;
}

template <>
inline __device__ void kahan_add<double>(double term, double &sum, double &c) {
    sum += term;
}

#define TILE_E 32
#define TILE_M 8
#define TILE_K 32

template <typename T>
__global__ void __launch_bounds__(256, 4) adj_template_bar_kernel(
    int nspec,
    int n_eff,
    int ky,
    int nwhl,
    int n_mod,
    int m0_0based,
    int j0_0based,
    int n_run_models,
    const int* __restrict__ d_perm,
    const T* __restrict__ d_etpl_t,
    const T* __restrict__ d_gw,
    int n_eff_max,
    int nc_max,
    double* __restrict__ d_vb
) {
    __shared__ T s_gwy[TILE_K][TILE_M];
    __shared__ T s_gwo[TILE_K][TILE_M];

    int tx = threadIdx.x; // 0..31 (e)
    int ty = threadIdx.y; // 0..7  (model)

    int e = blockIdx.x * TILE_E + tx;
    int j_local = blockIdx.y * TILE_M + ty;

    int model_idx = -1;
    if (j_local < n_run_models) {
        int jj = j0_0based + j_local;
        model_idx = d_perm[m0_0based + jj] - 1;
    }

    T ey = 0;
    T cey = 0;
    T eo = 0;
    T ceo = 0;
    T sy = 0;
    T csy = 0;
    T so = 0;
    T cso = 0;

    __shared__ T s_etpl[TILE_K][TILE_E];

    for (int k0 = 0; k0 < nspec; k0 += TILE_K) {
        int k = k0 + tx;
        if (j_local < n_run_models && k < nspec) {
            s_gwy[tx][ty] = d_gw[(size_t)model_idx * nspec + k];
            s_gwo[tx][ty] = d_gw[(size_t)(n_mod + model_idx) * nspec + k];
        } else {
            s_gwy[tx][ty] = 0;
            s_gwo[tx][ty] = 0;
        }

        // Cooperatively load TILE_K x TILE_E of d_etpl_t into shared LDS
        // Block has TILE_E * TILE_M = 32 * 8 = 256 threads.
        // Array size is TILE_K * TILE_E = 32 * 32 = 1024 elements (4 KB).
        // Each thread loads 4 elements.
        int tid = ty * TILE_E + tx;
        #pragma unroll
        for (int item = 0; item < 4; ++item) {
            int elem = tid + item * 256;
            int r_k = elem / TILE_E;
            int r_e = elem % TILE_E;
            int global_k = k0 + r_k;
            int global_e = blockIdx.x * TILE_E + r_e;
            if (global_k < nspec && global_e < n_eff) {
                s_etpl[r_k][r_e] = d_etpl_t[(size_t)global_k * n_eff + global_e];
            } else {
                s_etpl[r_k][r_e] = 0;
            }
        }
        __syncthreads();

        int k_end = (k0 + TILE_K < nspec) ? (k0 + TILE_K) : nspec;
        if (e < n_eff && j_local < n_run_models) {
            for (int curr_k = k0; curr_k < k_end; ++curr_k) {
                int k_sub = curr_k - k0;
                T ev = s_etpl[k_sub][tx];
                T gwy = s_gwy[k_sub][ty];
                T gwo = s_gwo[k_sub][ty];

                if (curr_k < nwhl) {
                    kahan_add(ev * gwy, ey, cey);
                    kahan_add(ev * gwo, eo, ceo);
                } else {
                    kahan_add(ev * gwy, sy, csy);
                    kahan_add(ev * gwo, so, cso);
                }
            }
        }
        __syncthreads();
    }

    if (e < n_eff && j_local < n_run_models) {
        int jj = j0_0based + j_local;
        size_t m_offset = (size_t)jj * 4 * n_eff_max;

        if (e + 1 <= ky) {
            d_vb[m_offset + 0 * n_eff_max + e] = (double)sy;
            d_vb[m_offset + 2 * n_eff_max + e] = (double)ey;
        } else {
            d_vb[m_offset + 0 * n_eff_max + e] = 0.0;
            d_vb[m_offset + 2 * n_eff_max + e] = 0.0;
        }
        d_vb[m_offset + 1 * n_eff_max + e] = (double)so;
        d_vb[m_offset + 3 * n_eff_max + e] = (double)eo;
    }
}

extern "C" {

void opensps_hip_adj_template_bar_sp(
    int nspec,
    int n_eff,
    int ky,
    int nwhl,
    int n_mod,
    int m0,
    int j0,
    int jr,
    const int* d_perm,
    const float* d_etpl_t,
    const float* d_gw,
    int n_eff_max,
    int nc_max,
    double* d_vb,
    void* stream
) {
    int n_run_models = jr - j0 + 1;
    if (n_run_models <= 0 || n_eff <= 0) return;

    dim3 block(TILE_E, TILE_M, 1);
    dim3 grid((n_eff + TILE_E - 1) / TILE_E, (n_run_models + TILE_M - 1) / TILE_M, 1);

    hipStream_t s = (hipStream_t)stream;

    adj_template_bar_kernel<float><<<grid, block, 0, s>>>(
        nspec, n_eff, ky, nwhl, n_mod,
        m0 - 1, j0 - 1, n_run_models,
        d_perm, d_etpl_t, d_gw,
        n_eff_max, nc_max, d_vb
    );
}

void opensps_hip_adj_template_bar_dp(
    int nspec,
    int n_eff,
    int ky,
    int nwhl,
    int n_mod,
    int m0,
    int j0,
    int jr,
    const int* d_perm,
    const double* d_etpl_t,
    const double* d_gw,
    int n_eff_max,
    int nc_max,
    double* d_vb,
    void* stream
) {
    int n_run_models = jr - j0 + 1;
    if (n_run_models <= 0 || n_eff <= 0) return;

    dim3 block(TILE_E, TILE_M, 1);
    dim3 grid((n_eff + TILE_E - 1) / TILE_E, (n_run_models + TILE_M - 1) / TILE_M, 1);

    hipStream_t s = (hipStream_t)stream;

    adj_template_bar_kernel<double><<<grid, block, 0, s>>>(
        nspec, n_eff, ky, nwhl, n_mod,
        m0 - 1, j0 - 1, n_run_models,
        d_perm, d_etpl_t, d_gw,
        n_eff_max, nc_max, d_vb
    );
}

} // extern "C"

template <typename T_WK>
__global__ void __launch_bounds__(128, 4) adj_fold_gather_kernel(
    int m0_0based,
    int nc,
    int nt,
    int nt_neb,
    int neb_cols,
    int n_pts_max,
    int n_eff_max,
    int nc_max,
    int n_mod,
    const double* __restrict__ d_wsfh,
    const int* __restrict__ d_it_tesc,
    const double* __restrict__ d_fesc,
    const int* __restrict__ d_pt_off,
    const int* __restrict__ d_pt_ptr,
    const double* __restrict__ d_eff_val,
    const int* __restrict__ d_pe_ptr,
    const int* __restrict__ d_pe_k,
    const int* __restrict__ d_pe_e,
    const int* __restrict__ d_perm,
    const int* __restrict__ d_viz,
    const double* __restrict__ d_cz,
    const double* __restrict__ d_zsg,
    const T_WK* __restrict__ d_wk,
    const double* __restrict__ d_vb,
    double* __restrict__ d_wbar,
    double* __restrict__ d_wpb,
    double* __restrict__ d_fesc_bar,
    double* __restrict__ d_zbar
) {
    int j_idx = blockIdx.x;
    if (j_idx >= nc) return;

    int tx = threadIdx.x;

    int idx = m0_0based + j_idx;
    int m = d_perm[idx] - 1;
    int iz = d_viz[idx];
    double c = d_cz[idx];
    double zs = d_zsg[idx];
    int it_tesc_m = d_it_tesc[idx];
    double fesc_m = d_fesc[idx];

    double local_term_z = 0.0;
    double local_term_f = 0.0;

    size_t m_offset = (size_t)j_idx * 4 * n_eff_max;

    for (int t = tx + 1; t <= nt; t += blockDim.x) {
        int ptr_idx = (iz - 1) * (nt + 1) + (t - 1);
        int p_start = d_pt_ptr[ptr_idx];
        int p_end   = d_pt_ptr[ptr_idx + 1];

        int col_g  = (t <= it_tesc_m) ? 0 : 1;
        int col_ge = (t <= it_tesc_m) ? 2 : 3;
        double fac = (t <= nt_neb) ? fesc_m : 1.0;

        double wsfh_val = d_wsfh[(size_t)(t - 1) * n_mod + m];
        double c_wsfh = c * wsfh_val;

        double sx = 0.0;
        double se = 0.0;

        for (int p = p_start; p < p_end; ++p) {
            int gp = d_pt_off[iz - 1] + p;
            int q_start = d_pe_ptr[gp - 1] - 1;
            int q_end   = d_pe_ptr[gp] - 1;

            double g = 0.0;
            double ge = 0.0;

            for (int q = q_start; q < q_end; ++q) {
                int k_0 = d_pe_k[q] - 1;
                int e_0 = d_pe_e[q] - 1;
                double ev = d_eff_val[k_0];
                g += ev * d_vb[m_offset + col_g * n_eff_max + e_0];
                if (neb_cols > 0) {
                    ge += ev * d_vb[m_offset + col_ge * n_eff_max + e_0];
                }
            }

            double gx = g + fac * ge;
            int p_0 = p - 1;
            size_t wp_idx = (size_t)p_0 * nc_max + j_idx;
            d_wpb[wp_idx] += c_wsfh * gx;

            double wk_val = (double)d_wk[wp_idx];
            sx += wk_val * gx;
            se += wk_val * ge;
        }

        d_wbar[(size_t)m * nt + (t - 1)] += c * sx;
        local_term_z += wsfh_val * sx;
        if (t <= nt_neb) {
            local_term_f += wsfh_val * se;
        }
    }

    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_term_z += __shfl_down(local_term_z, offset, warpSize);
        local_term_f += __shfl_down(local_term_f, offset, warpSize);
    }

    __shared__ double s_sum_z[8];
    __shared__ double s_sum_f[8];
    int wave_id = tx / warpSize;
    int lane_id = tx % warpSize;

    if (lane_id == 0) {
        s_sum_z[wave_id] = local_term_z;
        s_sum_f[wave_id] = local_term_f;
    }
    __syncthreads();

    if (tx == 0) {
        int num_waves = blockDim.x / warpSize;
        double total_z = 0.0;
        double total_f = 0.0;
        for (int w = 0; w < num_waves; ++w) {
            total_z += s_sum_z[w];
            total_f += s_sum_f[w];
        }
        d_zbar[m] += zs * total_z;
        if (neb_cols > 0) {
            d_fesc_bar[m] += c * total_f;
        }
    }
}

extern "C" {

void opensps_hip_adj_fold_gather_sp(
    int m0_0based,
    int nc,
    int nt,
    int nt_neb,
    int neb_cols,
    int n_pts_max,
    int n_eff_max,
    int nc_max,
    int n_mod,
    const double* __restrict__ d_wsfh,
    const int* __restrict__ d_it_tesc,
    const double* __restrict__ d_fesc,
    const int* __restrict__ d_pt_off,
    const int* __restrict__ d_pt_ptr,
    const double* __restrict__ d_eff_val,
    const int* __restrict__ d_pe_ptr,
    const int* __restrict__ d_pe_k,
    const int* __restrict__ d_pe_e,
    const int* __restrict__ d_perm,
    const int* __restrict__ d_viz,
    const double* __restrict__ d_cz,
    const double* __restrict__ d_zsg,
    const float* __restrict__ d_wk,
    const double* __restrict__ d_vb,
    double* __restrict__ d_wbar,
    double* __restrict__ d_wpb,
    double* __restrict__ d_fesc_bar,
    double* __restrict__ d_zbar,
    void* stream
) {
    if (nc <= 0) return;
    dim3 block(128);
    dim3 grid(nc);
    hipStream_t s = (hipStream_t)stream;
    adj_fold_gather_kernel<float><<<grid, block, 0, s>>>(
        m0_0based, nc, nt, nt_neb, neb_cols, n_pts_max, n_eff_max, nc_max, n_mod,
        d_wsfh, d_it_tesc, d_fesc, d_pt_off, d_pt_ptr, d_eff_val, d_pe_ptr, d_pe_k, d_pe_e,
        d_perm, d_viz, d_cz, d_zsg, d_wk, d_vb, d_wbar, d_wpb, d_fesc_bar, d_zbar
    );
}

void opensps_hip_adj_fold_gather_dp(
    int m0_0based,
    int nc,
    int nt,
    int nt_neb,
    int neb_cols,
    int n_pts_max,
    int n_eff_max,
    int nc_max,
    int n_mod,
    const double* __restrict__ d_wsfh,
    const int* __restrict__ d_it_tesc,
    const double* __restrict__ d_fesc,
    const int* __restrict__ d_pt_off,
    const int* __restrict__ d_pt_ptr,
    const double* __restrict__ d_eff_val,
    const int* __restrict__ d_pe_ptr,
    const int* __restrict__ d_pe_k,
    const int* __restrict__ d_pe_e,
    const int* __restrict__ d_perm,
    const int* __restrict__ d_viz,
    const double* __restrict__ d_cz,
    const double* __restrict__ d_zsg,
    const double* __restrict__ d_wk,
    const double* __restrict__ d_vb,
    double* __restrict__ d_wbar,
    double* __restrict__ d_wpb,
    double* __restrict__ d_fesc_bar,
    double* __restrict__ d_zbar,
    void* stream
) {
    if (nc <= 0) return;
    dim3 block(128);
    dim3 grid(nc);
    hipStream_t s = (hipStream_t)stream;
    adj_fold_gather_kernel<double><<<grid, block, 0, s>>>(
        m0_0based, nc, nt, nt_neb, neb_cols, n_pts_max, n_eff_max, nc_max, n_mod,
        d_wsfh, d_it_tesc, d_fesc, d_pt_off, d_pt_ptr, d_eff_val, d_pe_ptr, d_pe_k, d_pe_e,
        d_perm, d_viz, d_cz, d_zsg, d_wk, d_vb, d_wbar, d_wpb, d_fesc_bar, d_zbar
    );
}

void opensps_hip_sync_stream(void* stream) {
    if (stream) {
        (void)hipStreamSynchronize((hipStream_t)stream);
    } else {
        (void)hipDeviceSynchronize();
    }
}

} // extern "C"

template <typename T>
__device__ inline T exp_kp(double x);

template <>
__device__ inline double exp_kp<double>(double x) {
    return exp(x);
}

template <>
__device__ inline float exp_kp<float>(double x) {
    const double LN2 = 0.693147180559945309417232121458;
    const double LOG2E = 1.442695040888963407359924681002;
    double xc = fmax(-1.0e4, fmin(1.0e4, x));
    int n = (int)round(xc * LOG2E);
    if (n < -125) return 0.0f;
    if (n >= 128) return 3.402823466e+38f;
    return ldexpf(expf((float)(xc - (double)n * LN2)), n);
}

template <typename T>
__global__ void __launch_bounds__(256, 4) nebular_continuum_tiled_kernel(
    int n_mod,
    int nspec,
    int ntab,
    int nt_neb,
    const double* __restrict__ d_tab,
    const int* __restrict__ d_ia,
    const double* __restrict__ d_wa,
    const double* __restrict__ d_dwa,
    const int* __restrict__ d_newr,
    const T* __restrict__ d_cy,
    const T* __restrict__ d_co,
    const double* __restrict__ d_lns,
    const int* __restrict__ d_iz,
    const double* __restrict__ d_wz,
    const int* __restrict__ d_iu,
    const double* __restrict__ d_wu,
    T* __restrict__ d_yo
) {
    // 2D grid: blockIdx.x covers pixels (e.g. 256 threads per block, 1 pixel per thread)
    //          blockIdx.y covers models (1 model per block)
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y;
    if (m >= n_mod || j >= nspec) return;

    double LN10 = 2.30258509299404568402;
    int jz = d_iz[m] - 1; // 0-based
    int ju = d_iu[m] - 1; // 0-based
    double wz_m = d_wz[m];
    double wu_m = d_wu[m];

    double c00 = (1.0 - wz_m) * (1.0 - wu_m);
    double c01 = (1.0 - wz_m) * wu_m;
    double c10 = wz_m * (1.0 - wu_m);
    double c11 = wz_m * wu_m;

    double lsm = d_lns[m];
    int a_prev = -999;
    double ra = 0.0, rb = 0.0, dl = 0.0;
    T f = 0.0;
    T r = 1.0;
    T y = 0.0, o = 0.0, sy = 0.0, so = 0.0;

    // Strides for d_tab(ntab, NEBNZ=11, NEBNAGE=10, NEBNIP=7): column-major in Fortran
    // index = j + ntab * (jz + NEBNZ * (a + NEBNAGE * ju))
    long long s_z = ntab;
    long long s_a = ntab * 11;
    long long s_u = ntab * 11 * 10;

    for (int t = 0; t < nt_neb; ++t) {
        int a = d_ia[t] - 1; // 0-based
        double wat = d_wa[t];
        T cyt = d_cy[t + (long long)m * nt_neb];
        T cot = d_co[t + (long long)m * nt_neb];

        if (a != a_prev) {
            y += sy;
            o += so;
            sy = 0.0;
            so = 0.0;
            long long idx00 = j + s_z * jz + s_a * a + s_u * ju;
            if (a == a_prev + 1) {
                ra = rb;
            } else {
                ra = c00 * d_tab[idx00] +
                     c01 * d_tab[idx00 + s_u] +
                     c10 * d_tab[idx00 + s_z] +
                     c11 * d_tab[idx00 + s_z + s_u];
            }
            long long idx00_b = j + s_z * jz + s_a * (a + 1) + s_u * ju;
            rb = c00 * d_tab[idx00_b] +
                 c01 * d_tab[idx00_b + s_u] +
                 c10 * d_tab[idx00_b + s_z] +
                 c11 * d_tab[idx00_b + s_z + s_u];

            dl = (rb - ra) * LN10;
            double arg = ra * LN10 + wat * dl + lsm;
            f = exp_kp<T>(arg);
            a_prev = a;
        } else {
            if (d_newr[t]) {
                r = exp_kp<T>(d_dwa[t] * dl);
            }
            f = f * r;
        }
        sy += cyt * f;
        so += cot * f;
    }
    y += sy;
    o += so;

    long long off_y = j + (long long)m * nspec;
    long long off_o = j + (long long)(n_mod + m) * nspec;
    d_yo[off_y] += y;
    d_yo[off_o] += o;
}

// ---------------------------------------------------------------------------
// Adjoint Nebular Continuum Kernel
// ---------------------------------------------------------------------------

template <typename T>
__global__ void __launch_bounds__(256, 4) adj_nebular_continuum_kernel(
    int n_mod,
    int nspec,
    int ntab,
    int ntn,
    int nrun,
    const double* __restrict__ d_tab,
    const T* __restrict__ d_gw,
    const int* __restrict__ d_it_tesc,
    const int* __restrict__ d_ia,
    const double* __restrict__ d_wa,
    const int* __restrict__ d_rs,
    const int* __restrict__ d_rl,
    const int* __restrict__ d_iz,
    const double* __restrict__ d_wz,
    const int* __restrict__ d_iu,
    const double* __restrict__ d_wu,
    const double* __restrict__ d_fabs,
    const double* __restrict__ d_qzt,
    double* __restrict__ d_lref,
    T* __restrict__ d_csum
) {
    int r = blockIdx.x; // 0 .. nrun - 1
    int m = blockIdx.y; // 0 .. n_mod - 1
    if (m >= n_mod || r >= nrun) return;

    int ts = d_rs[r] - 1; // 0-based
    int tl = d_rl[r];
    int a = d_ia[ts] - 1; // 0-based
    int z0 = d_iz[m] - 1;
    int u0 = d_iu[m] - 1;
    double wzm = d_wz[m];
    double wum = d_wu[m];
    double fabsm = fabs(d_fabs[m]);
    int it_tesc_m = d_it_tesc[m]; // 1-based, comparing with ts + step + 1

    const double SAFE_FLOOR = 1.0e-30;
    const double LN10 = 2.30258509299404568402;

    double lnq = 1.0e30;
    bool ok[4] = {false, false, false, false};
    for (int step = 0; step < tl && step < 4; ++step) {
        double q = d_qzt[ts + step + (size_t)m * ntn];
        if (q * fabsm > SAFE_FLOOR) {
            ok[step] = true;
            if (lnq >= 1.0e30) {
                lnq = log(q);
            }
        }
    }

    if (threadIdx.x == 0) {
        d_lref[r + (size_t)m * nrun] = lnq;
    }

    T a_sum[20];
    T k_comp[20];
    #pragma unroll
    for (int i = 0; i < 20; ++i) {
        a_sum[i] = 0.0;
        k_comp[i] = 0.0;
    }

    if (lnq < 1.0e30) {
        double c00 = (1.0 - wzm) * (1.0 - wum);
        double c01 = (1.0 - wzm) * wum;
        double c10 = wzm * (1.0 - wum);
        double c11 = wzm * wum;

        long long s_z = ntab;
        long long s_a = ntab * 11;
        long long s_u = ntab * 11 * 10;
        long long base_00 = s_z * z0 + s_a * a + s_u * u0;
        long long base_b  = s_z * z0 + s_a * (a + 1) + s_u * u0;

        double wa_ts = d_wa[ts];
        size_t off_gy = (size_t)m * nspec;
        size_t off_go = ((size_t)n_mod + m) * nspec;

        for (int j = threadIdx.x; j < nspec; j += blockDim.x) {
            T gy = d_gw[j + off_gy];
            T go = d_gw[j + off_go];
            if (gy == 0.0 && go == 0.0) continue;

            long long idx00 = j + base_00;
            double t00 = d_tab[idx00];
            double t01 = d_tab[idx00 + s_u];
            double t10 = d_tab[idx00 + s_z];
            double t11 = d_tab[idx00 + s_z + s_u];

            double ra = c00 * t00 + c01 * t01 + c10 * t10 + c11 * t11;
            double raz = (1.0 - wum) * (t10 - t00) + wum * (t11 - t01);
            double rau = (1.0 - wzm) * (t01 - t00) + wzm * (t11 - t10);

            long long idxb = j + base_b;
            double tb00 = d_tab[idxb];
            double tb01 = d_tab[idxb + s_u];
            double tb10 = d_tab[idxb + s_z];
            double tb11 = d_tab[idxb + s_z + s_u];

            double rb = c00 * tb00 + c01 * tb01 + c10 * tb10 + c11 * tb11;
            double rbz = (1.0 - wum) * (tb10 - tb00) + wum * (tb11 - tb01);
            double rbu = (1.0 - wzm) * (tb01 - tb00) + wzm * (tb11 - tb10);

            T rz_a = (T)raz;
            T rz_b = (T)rbz;
            T ru_a = (T)rau;
            T rbu_k = (T)rbu;

            double dl = (rb - ra) * LN10;
            T e = exp_kp<T>(ra * LN10 + wa_ts * dl + lnq);
            T rr = 1.0;
            double dwp = 0.0;

            for (int step = 0; step < tl && step < 4; ++step) {
                if (step > 0) {
                    double dw = d_wa[ts + step] - d_wa[ts + step - 1];
                    if (dw != dwp) {
                        rr = exp_kp<T>(dw * dl);
                        dwp = dw;
                    }
                    e = e * rr;
                }
                if (ok[step]) {
                    T g = (ts + step + 1 <= it_tesc_m) ? (gy * e) : (go * e);
                    int b = step * 5;
                    kahan_add(g,         a_sum[b + 0], k_comp[b + 0]);
                    kahan_add(g * rz_a,  a_sum[b + 1], k_comp[b + 1]);
                    kahan_add(g * rz_b,  a_sum[b + 2], k_comp[b + 2]);
                    kahan_add(g * ru_a,  a_sum[b + 3], k_comp[b + 3]);
                    kahan_add(g * rbu_k, a_sum[b + 4], k_comp[b + 4]);
                }
            }
        }
    }

    // Intra-block reduction across 256 threads (8 warps)
    __shared__ double s_red[20][8];
    int lane = threadIdx.x % warpSize;
    int wave = threadIdx.x / warpSize;

    #pragma unroll
    for (int i = 0; i < 20; ++i) {
        double val = (double)a_sum[i];
        #pragma unroll
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            val += __shfl_down(val, offset, warpSize);
        }
        if (lane == 0) {
            s_red[i][wave] = val;
        }
    }
    __syncthreads();

    if (threadIdx.x < 20) {
        int i = threadIdx.x;
        int step = i / 5;
        int q = i % 5;
        if (step < tl) {
            double total = 0.0;
            if (ok[step]) {
                int num_waves = blockDim.x / warpSize;
                for (int w = 0; w < num_waves; ++w) {
                    total += s_red[i][w];
                }
            }
            size_t out_idx = q + 5 * ((size_t)(ts + step) + (size_t)ntn * m);
            d_csum[out_idx] = (T)total;
        }
    }
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Adjoint IMF Analytic (Kroupa & Dave)
// ---------------------------------------------------------------------------

__device__ __constant__ double k_inv_n[25] = {
    0.0,
    1.0,
    1.0 / 2.0,
    1.0 / 3.0,
    1.0 / 4.0,
    1.0 / 5.0,
    1.0 / 6.0,
    1.0 / 7.0,
    1.0 / 8.0,
    1.0 / 9.0,
    1.0 / 10.0,
    1.0 / 11.0,
    1.0 / 12.0,
    1.0 / 13.0,
    1.0 / 14.0,
    1.0 / 15.0,
    1.0 / 16.0,
    1.0 / 17.0,
    1.0 / 18.0,
    1.0 / 19.0,
    1.0 / 20.0,
    1.0 / 21.0,
    1.0 / 22.0,
    1.0 / 23.0,
    1.0 / 24.0
};

__device__ inline void powerlaw_j_dj(double lnlo, double x, double e, double &jv, double &dj) {
    double plo = exp(e * lnlo);
    double el = e * x;
    double g, gp;
    if (fabs(el) < 0.5) {
        g = x;
        double term = x;
        double u = 0.5 * x * x;
        gp = u;
        for (int n = 2; n <= 24; ++n) {
            double inv = k_inv_n[n];
            term = term * el * inv;
            if (n >= 3) u = u * el * inv;
            if (g + term == g && gp + (double)(n - 1) * u == gp) break;
            g += term;
            if (n >= 3) gp += (double)(n - 1) * u;
        }
    } else {
        double ex = exp(el);
        g = (ex - 1.0) / e;
        gp = (x * ex - g) / e;
    }
    jv = plo * g;
    dj = plo * (lnlo * g + gp);
}

__device__ inline double powerlaw_bin(double plo, double lo, double hi, double e) {
    double x = (hi - lo) / lo;
    double u = 1.0 + x;
    double dl;
    if (u == 1.0) {
        dl = x;
    } else {
        dl = log(u) * x / (u - 1.0);
    }
    if (e == 0.0) return plo * dl;
    double el = e * dl;
    if (fabs(el) < 0.5) {
        double s = dl;
        double term = dl;
        for (int n = 2; n <= 24; ++n) {
            term = term * el * k_inv_n[n];
            if (s + term == s) break;
            s += term;
        }
        return plo * s;
    } else {
        return plo * (exp(el) - 1.0) / e;
    }
}

__device__ inline double integrate_powerlaw(double lo, double hi, double alpha, bool mass_weighted, double coeff) {
    if (hi <= lo) return 0.0;
    double denom = mass_weighted ? (2.0 - alpha) : (1.0 - alpha);
    return coeff * powerlaw_bin(pow(lo, denom), lo, hi, denom);
}

__device__ inline double powerlaw_dj_de(double lo, double hi, double e) {
    double x = (hi - lo) / lo;
    double term = 1.0 + x;
    if (term != 1.0) x = log(term) * x / (term - 1.0);
    double el = e * x;
    double ln_lo = log(lo);
    double g, gp;
    if (fabs(el) < 0.5) {
        g = x;
        term = x;
        gp = 0.5 * x * x;
        double ex = gp;
        for (int n = 2; n <= 24; ++n) {
            double inv = k_inv_n[n];
            term = term * el * inv;
            g += term;
            if (n >= 3) {
                ex = ex * el * inv;
                gp += (double)(n - 1) * ex;
            }
            if (g + term == g) break;
        }
    } else {
        double ex = exp(el);
        g = (ex - 1.0) / e;
        gp = (x * ex - g) / e;
    }
    return pow(lo, e) * (ln_lo * g + gp);
}

__device__ inline double imf_integral_dp(int imf_type, const double* p, double m1, double m2, bool mass_weighted, int row) {
    double lo = (m1 < m2) ? m1 : m2;
    double hi = (m1 > m2) ? m1 : m2;
    if (hi <= lo) return 0.0;
    double base = mass_weighted ? 2.0 : 1.0;
    double d = 0.0;
    int t = imf_type % 10;
    if (t == 2) {
        double a1 = p[1];
        double a2 = p[2];
        double a3 = p[3];
        double c2 = pow(0.5, -a1 + a2);
        double ln05 = -0.693147180559945309417; // log(0.5)
        double sl = (lo > 0.08) ? lo : 0.08;
        double sh = (hi < 0.5) ? hi : 0.5;
        if (sh > sl && row == 2) {
            d -= powerlaw_dj_de(sl, sh, base - a1);
        }
        sl = (lo > 0.5) ? lo : 0.5;
        sh = (hi < 1.0) ? hi : 1.0;
        if (sh > sl) {
            double j2 = integrate_powerlaw(sl, sh, a2, mass_weighted, 1.0);
            if (row == 2) d -= ln05 * c2 * j2;
            if (row == 3) d += ln05 * c2 * j2 - c2 * powerlaw_dj_de(sl, sh, base - a2);
        }
        sl = (lo > 1.0) ? lo : 1.0;
        sh = hi;
        if (sh > sl) {
            double j2 = integrate_powerlaw(sl, sh, a3, mass_weighted, 1.0);
            if (row == 2) d -= ln05 * c2 * j2;
            if (row == 3) d += ln05 * c2 * j2;
            if (row == 4) d -= c2 * powerlaw_dj_de(sl, sh, base - a3);
        }
    } else if (t == 4) {
        double a1 = p[1];
        double a2 = p[2];
        double md = p[5]; // IMF_P_MDAVE
        double c2 = pow(md, -a1 + a2);
        double lmd = log(md);
        double sl = (lo > 0.08) ? lo : 0.08;
        double sh = (hi < md) ? hi : md;
        if (sh > sl && row == 2) {
            d -= powerlaw_dj_de(sl, sh, base - a1);
        }
        sl = (lo > md) ? lo : md;
        sh = hi;
        if (sh > sl) {
            double j2 = integrate_powerlaw(sl, sh, a2, mass_weighted, 1.0);
            if (row == 2) d -= lmd * c2 * j2;
            if (row == 3) d += lmd * c2 * j2 - c2 * powerlaw_dj_de(sl, sh, base - a2);
            if (row == 6) d += c2 * (a2 - a1) / md * j2;
        }
    }
    return d;
}

template <typename T_WK>
__global__ void __launch_bounds__(256, 6) adj_imf_analytic_kernel(
    int m0_0based,
    int nc,
    int nt,
    int n_pts_max,
    int n_mod,
    int imf_type,
    int ldw,
    int ldk,
    const int*    __restrict__ d_perm,
    const int*    __restrict__ d_viz,
    const double* __restrict__ d_cz,
    const double* __restrict__ d_wsfh,
    const double* __restrict__ d_pm,
    const double* __restrict__ d_tot,
    const double* __restrict__ d_dtt,
    const double* __restrict__ d_msum_bar,
    const int*    __restrict__ d_pt_off,
    const int*    __restrict__ d_pt_ptr,
    const int*    __restrict__ d_pt_age,
    const double* __restrict__ d_pt_mini,
    const double* __restrict__ d_pt_m1,
    const double* __restrict__ d_pt_m2,
    const double* __restrict__ d_pt_llo,
    const double* __restrict__ d_pt_dl,
    const int*    __restrict__ d_pt_first,
    const double* __restrict__ d_pt_cm,
    const T_WK*   __restrict__ d_wk,
    const double* __restrict__ d_wpb,
    double*       __restrict__ d_imf_bar
) {
    int tj = threadIdx.x; // 0 .. 31
    int tp = threadIdx.y; // 0 .. 7
    int j = blockIdx.x * blockDim.x + tj;
    bool valid_j = (j < nc);

    int idx = m0_0based + (valid_j ? j : 0);
    int m = valid_j ? (d_perm[idx] - 1) : 0;
    int iz = valid_j ? d_viz[idx] : 1;
    int iz_0 = iz - 1;

    int n_pts = valid_j ? (d_pt_ptr[iz_0 * (nt + 1) + nt] - 1) : 0;

    int t_imf = imf_type % 10;
    const double* p_imf = d_pm + (size_t)m * 312;

    double a1 = valid_j ? p_imf[1] : 0.0;
    double a2 = valid_j ? p_imf[2] : 0.0;
    double a3 = valid_j ? p_imf[3] : 0.0;
    double md = valid_j ? p_imf[5] : 1.0;
    double c2 = valid_j ? ((t_imf == 2) ? pow(0.5, -a1 + a2) : pow(md, -a1 + a2)) : 0.0;
    double lc = valid_j ? ((t_imf == 2) ? -0.693147180559945309417 : log(md)) : 0.0;

    double tot_val = valid_j ? d_tot[j] : 0.0;
    double inv_tot = (tot_val > 0.0) ? (1.0 / tot_val) : 0.0;
    double msum_b = valid_j ? d_msum_bar[m] : 0.0;
    double cz_m = valid_j ? d_cz[idx] : 0.0;
    double lim_lo = valid_j ? p_imf[6] : 0.0;
    double lim_hi = valid_j ? p_imf[7] : 0.0;

    double c[5] = {0.0, 0.0, 0.0, 0.0, 0.0};

    int p_stride = gridDim.y * blockDim.y;
    for (int p_0 = blockIdx.y * blockDim.y + tp; p_0 < n_pts_max; p_0 += p_stride) {
        if (!valid_j || p_0 >= n_pts) continue;

        int gp_0 = d_pt_off[iz_0] + p_0;
        double mini = d_pt_mini[gp_0];

        if (mini < lim_lo || mini > lim_hi) continue;

        int is_first = d_pt_first[gp_0];
        double lo = (is_first == 1) ? p_imf[8] : d_pt_m1[gp_0];
        double hi = d_pt_m2[gp_0];

        if (hi <= lo) continue;

        double lnlo = (is_first == 1) ? log(lo) : d_pt_llo[gp_0];
        double lnr  = (is_first == 1) ? log(hi / lo) : d_pt_dl[gp_0];

        int seg = 0;
        if (t_imf == 2) {
            if (lo >= 0.08 && hi <= 0.5) seg = 1;
            else if (lo >= 0.5 && hi <= 1.0) seg = 2;
            else if (lo >= 1.0) seg = 3;
        } else if (t_imf == 4) {
            if (lo >= 0.08 && hi <= md) seg = 1;
            else if (lo >= md) seg = 2;
        }

        double dv[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
        if (seg == 1) {
            double jv, dj;
            powerlaw_j_dj(lnlo, lnr, 1.0 - a1, jv, dj);
            dv[0] = -dj;
        } else if (seg == 2 && t_imf == 2) {
            double jv, dj;
            powerlaw_j_dj(lnlo, lnr, 1.0 - a2, jv, dj);
            dv[0] = -lc * c2 * jv;
            dv[1] = lc * c2 * jv - c2 * dj;
        } else if (seg == 3) {
            double jv, dj;
            powerlaw_j_dj(lnlo, lnr, 1.0 - a3, jv, dj);
            dv[0] = -lc * c2 * jv;
            dv[1] = lc * c2 * jv;
            dv[2] = -c2 * dj;
        } else if (seg == 2 && t_imf == 4) {
            double jv, dj;
            powerlaw_j_dj(lnlo, lnr, 1.0 - a2, jv, dj);
            dv[0] = -lc * c2 * jv;
            dv[1] = lc * c2 * jv - c2 * dj;
            dv[4] = c2 * (a2 - a1) / md * jv;
        } else {
            dv[0] = imf_integral_dp(imf_type, p_imf, lo, hi, false, 2);
            dv[1] = imf_integral_dp(imf_type, p_imf, lo, hi, false, 3);
            if (t_imf == 2) {
                dv[2] = imf_integral_dp(imf_type, p_imf, lo, hi, false, 4);
            } else if (t_imf == 4) {
                dv[4] = imf_integral_dp(imf_type, p_imf, lo, hi, false, 6);
            }
        }

        int t_age = d_pt_age[gp_0];
        double wsfh_val = d_wsfh[(size_t)(t_age - 1) * n_mod + m];
        size_t wp_idx = (size_t)p_0 * ldw + j;
        size_t wk_idx = (size_t)p_0 * ldk + j;
        double cot_k = d_wpb[wp_idx] + msum_b * cz_m * wsfh_val * d_pt_cm[gp_0];
        double wk_val = (double)d_wk[wk_idx];

        if (t_imf == 2) {
            double dw0 = (dv[0] - wk_val * d_dtt[j * 5 + 0]) * inv_tot;
            double dw1 = (dv[1] - wk_val * d_dtt[j * 5 + 1]) * inv_tot;
            double dw2 = (dv[2] - wk_val * d_dtt[j * 5 + 2]) * inv_tot;
            c[0] += dw0 * cot_k;
            c[1] += dw1 * cot_k;
            c[2] += dw2 * cot_k;
        } else if (t_imf == 4) {
            double dw0 = (dv[0] - wk_val * d_dtt[j * 5 + 0]) * inv_tot;
            double dw1 = (dv[1] - wk_val * d_dtt[j * 5 + 1]) * inv_tot;
            double dw4 = (dv[4] - wk_val * d_dtt[j * 5 + 4]) * inv_tot;
            c[0] += dw0 * cot_k;
            c[1] += dw1 * cot_k;
            c[4] += dw4 * cot_k;
        }
    }

    __shared__ double s_red[5][8][32];
    #pragma unroll
    for (int ig = 0; ig < 5; ++ig) {
        s_red[ig][tp][tj] = c[ig];
    }
    __syncthreads();

    if (tp == 0 && valid_j) {
        #pragma unroll
        for (int ig = 0; ig < 5; ++ig) {
            double tot_c = s_red[ig][0][tj] + s_red[ig][1][tj] +
                           s_red[ig][2][tj] + s_red[ig][3][tj] +
                           s_red[ig][4][tj] + s_red[ig][5][tj] +
                           s_red[ig][6][tj] + s_red[ig][7][tj];
            if (tot_c != 0.0) {
                atomicAdd(&d_imf_bar[(size_t)m * 5 + ig], tot_c);
            }
        }
    }
}



// Nebular continuum with the models grouped by (Z, U) cell (FP32 kernels). One thread serves up to NG models that share
// a cell. Everything that depends only on (pixel, age interval, cell) is formed once per group, in FP64: the corner
// values T00 of the log grid, their differences to the other three corners (D01, D10, D11, rounded to FP32), the
// log-flux step between the interval's ends, and the range reduction x = n ln 2 + r of T00 ln 10. Per model there are only
// FP32 operations on small numbers: the interpolation offset c01 D01 + c10 D10 + c11 D11 (the weights c sum to 1, so
// T = T00 + sum_i c_i D_i), the model's own reduced scale ln(max_t C) = n_m ln 2 + l_m (once per model), and
// the exponential of a small argument times 2**(n + n_m). FP32 enters only through numbers of order 1 (rule 7 of
// doc/MIXED_PRECISION.md: the large exponent, ~100, is reduced in FP64): the error is a few 1e-8 of the offset (<~ 3 in
// log10) in the exponent, i.e. below ~5e-7 relative in the continuum. The FP64 operations per (pixel, model) drop from
// ~400 to ~400 / NG: on GPUs whose FP64 rate is 1/32-1/64 of FP32 (the R9700) they were the cost of this kernel.
// d_order: the models sorted by cell; d_gstart: group g holds d_order[d_gstart[g] .. d_gstart[g+1]).
template <int NG>
__global__ void __launch_bounds__(256, 2) nebular_continuum_grouped_kernel(
    int n_mod,
    int nspec,
    int ntab,
    int nt_neb,
    const double* __restrict__ d_tab,
    const int* __restrict__ d_ia,
    const double* __restrict__ d_wa,
    const double* __restrict__ d_dwa,
    const int* __restrict__ d_newr,
    const float* __restrict__ d_cy,
    const float* __restrict__ d_co,
    const double* __restrict__ d_lns,
    const int* __restrict__ d_iz,
    const double* __restrict__ d_wz,
    const int* __restrict__ d_iu,
    const double* __restrict__ d_wu,
    const int* __restrict__ d_order,
    const int* __restrict__ d_gstart,
    float* __restrict__ d_yo
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int g = blockIdx.y;
    if (j >= nspec) return;
    const int g0 = d_gstart[g];
    const int ng = d_gstart[g + 1] - g0;     // 1 .. NG, uniform over the block
    const double LN10 = 2.30258509299404568402;
    const double LN2 = 0.693147180559945309417232121458;
    const double LOG2E = 1.442695040888963407359924681002;
    const float LN10F = (float)LN10;

    int mi[NG], nm[NG];
    float c01[NG], c10[NG], c11[NG], lm[NG], dl[NG], f[NG], r[NG], y[NG], o[NG], sy[NG], so[NG];
    #pragma unroll
    for (int k = 0; k < NG; ++k) {
        mi[k] = d_order[g0 + (k < ng ? k : 0)];
        const double wz_m = d_wz[mi[k]], wu_m = d_wu[mi[k]];
        c01[k] = (float)((1.0 - wz_m) * wu_m);
        c10[k] = (float)(wz_m * (1.0 - wu_m));
        c11[k] = (float)(wz_m * wu_m);
        const double lsm = d_lns[mi[k]];
        const double xc = fmax(-1.0e4, fmin(1.0e4, lsm));
        nm[k] = (int)rint(xc * LOG2E);
        lm[k] = (float)(xc - (double)nm[k] * LN2);
        dl[k] = 0.f; f[k] = 0.f; r[k] = 1.f; y[k] = 0.f; o[k] = 0.f; sy[k] = 0.f; so[k] = 0.f;
    }
    const int jz = d_iz[mi[0]] - 1;          // the cell is common to the group
    const int ju = d_iu[mi[0]] - 1;
    const long long s_z = ntab, s_a = (long long)ntab * 11, s_u = (long long)ntab * 11 * 10;
    int a_prev = -999;
    // slab a_prev + 1 (the upper end of the last interval): its corner T00 and the differences to the other corners
    double Tb00 = 0.0; float Db01 = 0.f, Db10 = 0.f, Db11 = 0.f;

    for (int t = 0; t < nt_neb; ++t) {
        const int a = d_ia[t] - 1;
        if (a != a_prev) {
            double Ta00; float Da01, Da10, Da11;
            if (a == a_prev + 1) {
                Ta00 = Tb00; Da01 = Db01; Da10 = Db10; Da11 = Db11;
            } else {
                const long long i0 = j + s_z * jz + s_a * a + s_u * ju;
                const double A00 = d_tab[i0];
                Ta00 = A00;
                Da01 = (float)(d_tab[i0 + s_u] - A00);
                Da10 = (float)(d_tab[i0 + s_z] - A00);
                Da11 = (float)(d_tab[i0 + s_z + s_u] - A00);
            }
            const long long ib = j + s_z * jz + s_a * (a + 1) + s_u * ju;
            const double B00 = d_tab[ib];
            Db01 = (float)(d_tab[ib + s_u] - B00);
            Db10 = (float)(d_tab[ib + s_z] - B00);
            Db11 = (float)(d_tab[ib + s_z + s_u] - B00);
            Tb00 = B00;
            // shared FP64: the log step of the interval and the range reduction of the lower end
            const float dbase = (float)((B00 - Ta00) * LN10);
            const double X0 = Ta00 * LN10;
            const int na = (int)rint(X0 * LOG2E);
            const float ra_red = (float)(X0 - (double)na * LN2);
            const float wat = (float)d_wa[t];
            #pragma unroll
            for (int k = 0; k < NG; ++k) {
                if (k < ng) {
                    y[k] += sy[k]; o[k] += so[k]; sy[k] = 0.f; so[k] = 0.f;
                    const float offa = c01[k] * Da01 + c10[k] * Da10 + c11[k] * Da11;
                    const float offb = c01[k] * Db01 + c10[k] * Db10 + c11[k] * Db11;
                    dl[k] = dbase + (offb - offa) * LN10F;
                    const float arg = ra_red + lm[k] + offa * LN10F + wat * dl[k];
                    const int n = na + nm[k];
                    f[k] = (n < -125) ? 0.f : (n >= 128 ? 3.402823466e+38f : ldexpf(expf(arg), n));
                }
            }
            a_prev = a;
        } else {
            const bool nr = d_newr[t] != 0;
            const float dwa = (float)d_dwa[t];
            #pragma unroll
            for (int k = 0; k < NG; ++k) {
                if (k < ng) {
                    if (nr) r[k] = expf(dwa * dl[k]);
                    f[k] = f[k] * r[k];
                }
            }
        }
        #pragma unroll
        for (int k = 0; k < NG; ++k) {
            if (k < ng) {
                sy[k] += d_cy[t + (long long)mi[k] * nt_neb] * f[k];
                so[k] += d_co[t + (long long)mi[k] * nt_neb] * f[k];
            }
        }
    }
    #pragma unroll
    for (int k = 0; k < NG; ++k) {
        if (k < ng) {
            y[k] += sy[k]; o[k] += so[k];
            d_yo[j + (long long)mi[k] * nspec] += y[k];
            d_yo[j + (long long)(n_mod + mi[k]) * nspec] += o[k];
        }
    }
}

// The groups (host side: a counting sort of the models by cell), persistent device scratch
static int* g_d_order = nullptr; static int* g_d_gstart = nullptr; static int g_cap = 0;
static int build_cell_groups(int n_mod, const int* d_iz, const int* d_iu, int ng_max) {
    if (g_cap < n_mod) {
        if (g_d_order) { (void)hipFree(g_d_order); (void)hipFree(g_d_gstart); }
        (void)hipMalloc(&g_d_order, (size_t)n_mod * sizeof(int));
        (void)hipMalloc(&g_d_gstart, ((size_t)n_mod + 1) * sizeof(int));
        g_cap = n_mod;
    }
    int* hz = (int*)malloc((size_t)n_mod * 4 * sizeof(int) + 16);
    int* hu = hz + n_mod; int* ord = hu + n_mod; int* gs = ord + n_mod;
    (void)hipMemcpy(hz, d_iz, (size_t)n_mod * sizeof(int), hipMemcpyDeviceToHost);
    (void)hipMemcpy(hu, d_iu, (size_t)n_mod * sizeof(int), hipMemcpyDeviceToHost);
    int maxk = 0;
    for (int m = 0; m < n_mod; ++m) { int k = hz[m] * 64 + hu[m]; if (k > maxk) maxk = k; }
    int* cnt = (int*)calloc((size_t)maxk + 2, sizeof(int));
    for (int m = 0; m < n_mod; ++m) cnt[hz[m] * 64 + hu[m] + 1]++;
    for (int k = 0; k <= maxk; ++k) cnt[k + 1] += cnt[k];
    int* pos = (int*)malloc(((size_t)maxk + 2) * sizeof(int));
    for (int k = 0; k <= maxk + 1; ++k) pos[k] = cnt[k];
    for (int m = 0; m < n_mod; ++m) ord[pos[hz[m] * 64 + hu[m]]++] = m;
    int ngr = 0;
    for (int k = 0; k <= maxk; ++k) {
        for (int b = cnt[k]; b < cnt[k + 1]; b += ng_max) gs[ngr++] = b;
    }
    gs[ngr] = n_mod;
    (void)hipMemcpy(g_d_order, ord, (size_t)n_mod * sizeof(int), hipMemcpyHostToDevice);
    (void)hipMemcpy(g_d_gstart, gs, ((size_t)ngr + 1) * sizeof(int), hipMemcpyHostToDevice);
    free(hz); free(cnt); free(pos);
    return ngr;
}


__global__ void split_table_kernel(size_t n, const double* __restrict__ d, float2* __restrict__ out) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float h = (float)d[i];
    out[i] = make_float2(h, (float)(d[i] - (double)h));
}
static float2* g_neb2 = nullptr;
static size_t g_neb2_n = 0;


// ---------------------------------------------------------------------------
// Reverse pass of the nebular continuum, FP32 (adj_nebular_continuum_f32_kernel)
//
// The FP64 kernel above took ~40 FP64 operations per (pixel, model, run of ages) (the corner combinations, the
// derivatives, the exponent and its range reduction): 20 of the 55 ms of the CLOUDY config's reverse pass at 1000
// models (FP64 at 1/64 of the FP32 rate). Here the table is read as FP32 hi + lo pairs (split once per call,
// split_dustem_kernel), the corner values enter as differences (exact in FP32 where the neighbouring cells'
// values are within a factor of two, as log10 L of neighbouring cells are) and the exponent
//   E = ln10 T00 + ln q + [offset of the model's (Z, U) in the cell] ln10 + w_a dl
// is formed as a double-float sum (the product T00 ln10 by an fma, the sum by a two-sum) and range-reduced
// E = n ln2 + r in FP32 with the two-constant (Cody-Waite) ln2: the same ~1 ulp as the FP64 reduction without
// FP64 operations (rule 10 of doc/MIXED_PRECISION.md). The accumulations are those of the FP64 kernel
// (Kahan, the reduction over the block's 256 threads in FP64).
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(256, 4) adj_nebular_continuum_f32_kernel(
    int n_mod, int nspec, int ntab, int ntn, int nrun,
    const float2* __restrict__ d_tab2,
    const float* __restrict__ d_gw,
    const int* __restrict__ d_it_tesc,
    const int* __restrict__ d_ia,
    const double* __restrict__ d_wa,
    const int* __restrict__ d_rs,
    const int* __restrict__ d_rl,
    const int* __restrict__ d_iz,
    const double* __restrict__ d_wz,
    const int* __restrict__ d_iu,
    const double* __restrict__ d_wu,
    const double* __restrict__ d_fabs,
    const double* __restrict__ d_qzt,
    double* __restrict__ d_lref,
    float* __restrict__ d_csum
) {
#pragma clang fp contract(off)
    int r = blockIdx.x;
    int m = blockIdx.y;
    if (m >= n_mod || r >= nrun) return;

    int ts = d_rs[r] - 1;
    int tl = d_rl[r];
    int a = d_ia[ts] - 1;
    int z0 = d_iz[m] - 1;
    int u0 = d_iu[m] - 1;
    double wzm = d_wz[m];
    double wum = d_wu[m];
    double fabsm = fabs(d_fabs[m]);
    int it_tesc_m = d_it_tesc[m];

    const double SAFE_FLOOR = 1.0e-30;
    double lnq = 1.0e30;
    bool ok[4] = {false, false, false, false};
    for (int step = 0; step < tl && step < 4; ++step) {
        double q = d_qzt[ts + step + (size_t)m * ntn];
        if (q * fabsm > SAFE_FLOOR) {
            ok[step] = true;
            if (lnq >= 1.0e30) lnq = log(q);
        }
    }
    if (threadIdx.x == 0) d_lref[r + (size_t)m * nrun] = lnq;

    float a_sum[20], k_comp[20];
#pragma unroll
    for (int i = 0; i < 20; ++i) { a_sum[i] = 0.f; k_comp[i] = 0.f; }

    if (lnq < 1.0e30) {
        const float c01 = (float)((1.0 - wzm) * wum), c10 = (float)(wzm * (1.0 - wum)), c11 = (float)(wzm * wum);
        const float wzf = (float)wzm, wuf = (float)wum;
        const float wz1 = (float)(1.0 - wzm), wu1 = (float)(1.0 - wum);
        const float lq_hi = (float)lnq, lq_lo = (float)(lnq - (double)lq_hi);
        const float wat = (float)d_wa[ts];
        float dwf[4];
        dwf[0] = 0.f;
#pragma unroll
        for (int st = 1; st < 4; ++st) dwf[st] = (st < tl) ? (float)(d_wa[ts + st] - d_wa[ts + st - 1]) : 0.f;
        // the steps' dw are compared as in the FP64 kernel (a new factor only where dw changes)
        bool newdw[4];
        newdw[0] = false;
        double dwp_h = 0.0;
#pragma unroll
        for (int st = 1; st < 4; ++st) {
            double dwd = (st < tl) ? (d_wa[ts + st] - d_wa[ts + st - 1]) : 0.0;
            newdw[st] = (dwd != dwp_h);
            if (newdw[st]) dwp_h = dwd;
        }
        const float LN10H = 2.302585125f, LN10L = -3.2e-9f;      // ln10 = LN10H + LN10L (hi + lo)
        const float LOG2E = 1.44269504088896f, LN2H = 0.693145751953125f, LN2L = 1.42860677e-06f;

        const long long s_z = ntab, s_a = (long long)ntab * 11, s_u = (long long)ntab * 11 * 10;
        const long long base_00 = s_z * z0 + s_a * a + s_u * u0;
        const long long base_b = s_z * z0 + s_a * (a + 1) + s_u * u0;
        const size_t off_gy = (size_t)m * nspec;
        const size_t off_go = ((size_t)n_mod + m) * nspec;

        for (int j = threadIdx.x; j < nspec; j += blockDim.x) {
            const float gy = d_gw[j + off_gy];
            const float go = d_gw[j + off_go];
            if (gy == 0.f && go == 0.f) continue;

            const long long ia0 = j + base_00, ib0 = j + base_b;
            const float2 A00 = d_tab2[ia0], A01 = d_tab2[ia0 + s_u], A10 = d_tab2[ia0 + s_z], A11 = d_tab2[ia0 + s_z + s_u];
            const float2 B00 = d_tab2[ib0], B01 = d_tab2[ib0 + s_u], B10 = d_tab2[ib0 + s_z], B11 = d_tab2[ib0 + s_z + s_u];
            const float Da01 = (A01.x - A00.x) + (A01.y - A00.y), Da10 = (A10.x - A00.x) + (A10.y - A00.y),
                        Da11 = (A11.x - A00.x) + (A11.y - A00.y);
            const float Db01 = (B01.x - B00.x) + (B01.y - B00.y), Db10 = (B10.x - B00.x) + (B10.y - B00.y),
                        Db11 = (B11.x - B00.x) + (B11.y - B00.y);
            const float offa = c01 * Da01 + c10 * Da10 + c11 * Da11;
            const float offb = c01 * Db01 + c10 * Db10 + c11 * Db11;
            const float rz_a = wu1 * Da10 + wuf * (Da11 - Da01);
            const float ru_a = wz1 * Da01 + wzf * (Da11 - Da10);
            const float rz_b = wu1 * Db10 + wuf * (Db11 - Db01);
            const float rbu_k = wz1 * Db01 + wzf * (Db11 - Db10);
            const float dTab = (B00.x - A00.x) + (B00.y - A00.y);
            const float dl = (dTab + (offb - offa)) * LN10H;

            // E = ln10 T00 + ln q + offa ln10 + w_a dl, as a double-float
            const float p = __fmul_rn(A00.x, LN10H);
            const float pe = fmaf(A00.x, LN10H, -p) + (A00.x * LN10L + A00.y * LN10H);
            const float s1 = p + lq_hi;
            const float bb = s1 - p;
            const float err = (p - (s1 - bb)) + (lq_hi - bb);
            const float rest = ((err + pe) + lq_lo) + (offa * LN10H + wat * dl);
            const float nf = rintf(s1 * LOG2E);
            const int n = (int)nf;
            float e;
            if (n < -125) {
                e = 0.f;
            } else if (n >= 128) {
                e = 3.402823466e+38f;
            } else {
                float r1 = fmaf(-nf, LN2H, s1);
                r1 = fmaf(-nf, LN2L, r1);
                e = ldexpf(expf(r1 + rest), n);
            }
            float rr = 1.f;
#pragma unroll
            for (int step = 0; step < 4; ++step) {
                if (step < tl) {
                    if (step > 0) {
                        if (newdw[step]) rr = expf(dwf[step] * dl);
                        e = e * rr;
                    }
                    if (ok[step]) {
                        const float g = (ts + step + 1 <= it_tesc_m) ? (gy * e) : (go * e);
                        const int b = step * 5;
                        kahan_add(g, a_sum[b + 0], k_comp[b + 0]);
                        kahan_add(g * rz_a, a_sum[b + 1], k_comp[b + 1]);
                        kahan_add(g * rz_b, a_sum[b + 2], k_comp[b + 2]);
                        kahan_add(g * ru_a, a_sum[b + 3], k_comp[b + 3]);
                        kahan_add(g * rbu_k, a_sum[b + 4], k_comp[b + 4]);
                    }
                }
            }
        }
    }

    __shared__ double s_red[20][8];
    const int lane = threadIdx.x % warpSize;
    const int wave = threadIdx.x / warpSize;
#pragma unroll
    for (int i = 0; i < 20; ++i) {
        double val = (double)a_sum[i];
#pragma unroll
        for (int offset = warpSize / 2; offset > 0; offset /= 2) val += __shfl_down(val, offset, warpSize);
        if (lane == 0) s_red[i][wave] = val;
    }
    __syncthreads();
    if (threadIdx.x < 20) {
        const int i = threadIdx.x;
        const int step = i / 5;
        const int q = i % 5;
        if (step < tl) {
            double total = 0.0;
            if (ok[step]) {
                const int num_waves = blockDim.x / warpSize;
                for (int w = 0; w < num_waves; ++w) total += s_red[i][w];
            }
            d_csum[q + 5 * ((size_t)(ts + step) + (size_t)ntn * m)] = (float)total;
        }
    }
}

extern "C" {

void opensps_hip_nebular_continuum_sp(
    int n_mod,
    int nspec,
    int ntab,
    int nt_neb,
    const double* __restrict__ d_tab,
    const int* __restrict__ d_ia,
    const double* __restrict__ d_wa,
    const double* __restrict__ d_dwa,
    const int* __restrict__ d_newr,
    const float* __restrict__ d_cy,
    const float* __restrict__ d_co,
    const double* __restrict__ d_lns,
    const int* __restrict__ d_iz,
    const double* __restrict__ d_wz,
    const int* __restrict__ d_iu,
    const double* __restrict__ d_wu,
    float* __restrict__ d_yo,
    void* stream
) {
    if (n_mod <= 0 || nspec <= 0 || nt_neb <= 0) return;
    dim3 block(256);
    dim3 grid((nspec + 255) / 256, n_mod);
    hipStream_t s = (hipStream_t)stream;
    const char* grp_env = getenv("OPENSPS_NEB_GROUP");
    int gsz = grp_env ? atoi(grp_env) : 8;   // 0 or 1: the one-thread-per-model kernel (A/B timing)
    if (gsz >= 2) {
        const int G = gsz >= 8 ? 8 : (gsz >= 4 ? 4 : 2);
        const int ngr = build_cell_groups(n_mod, d_iz, d_iu, G);
        dim3 gg((nspec + 255) / 256, ngr);
#define NEB_GROUPED(GG) nebular_continuum_grouped_kernel<GG><<<gg, block, 0, s>>>( \
            n_mod, nspec, ntab, nt_neb, d_tab, d_ia, d_wa, d_dwa, d_newr, d_cy, d_co, d_lns, d_iz, d_wz, d_iu, d_wu, \
            g_d_order, g_d_gstart, d_yo)
        if (G == 8) { NEB_GROUPED(8); } else if (G == 4) { NEB_GROUPED(4); } else { NEB_GROUPED(2); }
        (void)hipDeviceSynchronize();
        return;
    }
    nebular_continuum_tiled_kernel<float><<<grid, block, 0, s>>>(
        n_mod, nspec, ntab, nt_neb, d_tab, d_ia, d_wa, d_dwa, d_newr,
        d_cy, d_co, d_lns, d_iz, d_wz, d_iu, d_wu, d_yo
    );
    (void)hipDeviceSynchronize();
}

void opensps_hip_nebular_continuum_dp(
    int n_mod,
    int nspec,
    int ntab,
    int nt_neb,
    const double* __restrict__ d_tab,
    const int* __restrict__ d_ia,
    const double* __restrict__ d_wa,
    const double* __restrict__ d_dwa,
    const int* __restrict__ d_newr,
    const double* __restrict__ d_cy,
    const double* __restrict__ d_co,
    const double* __restrict__ d_lns,
    const int* __restrict__ d_iz,
    const double* __restrict__ d_wz,
    const int* __restrict__ d_iu,
    const double* __restrict__ d_wu,
    double* __restrict__ d_yo,
    void* stream
) {
    if (n_mod <= 0 || nspec <= 0 || nt_neb <= 0) return;
    dim3 block(256);
    dim3 grid((nspec + 255) / 256, n_mod);
    hipStream_t s = (hipStream_t)stream;
    nebular_continuum_tiled_kernel<double><<<grid, block, 0, s>>>(
        n_mod, nspec, ntab, nt_neb, d_tab, d_ia, d_wa, d_dwa, d_newr,
        d_cy, d_co, d_lns, d_iz, d_wz, d_iu, d_wu, d_yo
    );
    (void)hipDeviceSynchronize();
}

void opensps_hip_adj_nebular_continuum_sp(
    int n_mod,
    int nspec,
    int ntab,
    size_t ntot,
    int ntn,
    int nrun,
    const double* __restrict__ d_tab,
    const float*  __restrict__ d_gw,
    const int*    __restrict__ d_it_tesc,
    const int*    __restrict__ d_ia,
    const double* __restrict__ d_wa,
    const int*    __restrict__ d_rs,
    const int*    __restrict__ d_rl,
    const int*    __restrict__ d_iz,
    const double* __restrict__ d_wz,
    const int*    __restrict__ d_iu,
    const double* __restrict__ d_wu,
    const double* __restrict__ d_fabs,
    const double* __restrict__ d_qzt,
    double*       __restrict__ d_lref,
    float*        __restrict__ d_csum,
    void* stream
) {
    if (n_mod <= 0 || nrun <= 0 || nspec <= 0 || ntn <= 0) return;
    dim3 block(256);
    dim3 grid(nrun, n_mod);
    hipStream_t s = (hipStream_t)stream;
    if (ntot > g_neb2_n) {
        if (g_neb2) (void)hipFree(g_neb2);
        if (hipMalloc((void**)&g_neb2, ntot * sizeof(float2)) != hipSuccess) { g_neb2 = nullptr; g_neb2_n = 0; }
        else g_neb2_n = ntot;
    }
    if (g_neb2) {
        // (the table as hi + lo pairs, formed per call: ~0.3 ms, no stale copy if the table is reloaded)
        split_table_kernel<<<(unsigned)((ntot + 255) / 256), 256, 0, s>>>(ntot, d_tab, g_neb2);
        adj_nebular_continuum_f32_kernel<<<grid, block, 0, s>>>(
            n_mod, nspec, ntab, ntn, nrun, g_neb2, d_gw, d_it_tesc, d_ia, d_wa,
            d_rs, d_rl, d_iz, d_wz, d_iu, d_wu, d_fabs, d_qzt, d_lref, d_csum
        );
    } else {
        adj_nebular_continuum_kernel<float><<<grid, block, 0, s>>>(
            n_mod, nspec, ntab, ntn, nrun, d_tab, d_gw, d_it_tesc, d_ia, d_wa,
            d_rs, d_rl, d_iz, d_wz, d_iu, d_wu, d_fabs, d_qzt, d_lref, d_csum
        );
    }
    (void)hipDeviceSynchronize();
}

void opensps_hip_adj_nebular_continuum_dp(
    int n_mod,
    int nspec,
    int ntab,
    int ntn,
    int nrun,
    const double* __restrict__ d_tab,
    const double* __restrict__ d_gw,
    const int*    __restrict__ d_it_tesc,
    const int*    __restrict__ d_ia,
    const double* __restrict__ d_wa,
    const int*    __restrict__ d_rs,
    const int*    __restrict__ d_rl,
    const int*    __restrict__ d_iz,
    const double* __restrict__ d_wz,
    const int*    __restrict__ d_iu,
    const double* __restrict__ d_wu,
    const double* __restrict__ d_fabs,
    const double* __restrict__ d_qzt,
    double*       __restrict__ d_lref,
    double*       __restrict__ d_csum,
    void* stream
) {
    if (n_mod <= 0 || nrun <= 0 || nspec <= 0 || ntn <= 0) return;
    dim3 block(256);
    dim3 grid(nrun, n_mod);
    hipStream_t s = (hipStream_t)stream;
    adj_nebular_continuum_kernel<double><<<grid, block, 0, s>>>(
        n_mod, nspec, ntab, ntn, nrun, d_tab, d_gw, d_it_tesc, d_ia, d_wa,
        d_rs, d_rl, d_iz, d_wz, d_iu, d_wu, d_fabs, d_qzt, d_lref, d_csum
    );
    (void)hipDeviceSynchronize();
}

void opensps_hip_adj_imf_analytic_sp(
    int m0_0based,
    int nc,
    int nt,
    int n_pts_max,
    int n_mod,
    int imf_type,
    int ldw,
    int ldk,
    const int*    d_perm,
    const int*    d_viz,
    const double* d_cz,
    const double* d_wsfh,
    const double* d_pm,
    const double* d_tot,
    const double* d_dtt,
    const double* d_msum_bar,
    const int*    d_pt_off,
    const int*    d_pt_ptr,
    const int*    d_pt_age,
    const double* d_pt_mini,
    const double* d_pt_m1,
    const double* d_pt_m2,
    const double* d_pt_llo,
    const double* d_pt_dl,
    const int*    d_pt_first,
    const double* d_pt_cm,
    const float*  d_wk,
    const double* d_wpb,
    double*       d_imf_bar,
    void* stream
) {
    if (nc <= 0 || n_pts_max <= 0) return;
    dim3 block(32, 8);
    int grid_j = (nc + 31) / 32;
    int grid_p = 64;
    dim3 grid(grid_j, grid_p);
    hipStream_t s = (hipStream_t)stream;
    adj_imf_analytic_kernel<float><<<grid, block, 0, s>>>(
        m0_0based, nc, nt, n_pts_max, n_mod, imf_type, ldw, ldk,
        d_perm, d_viz, d_cz, d_wsfh, d_pm, d_tot, d_dtt, d_msum_bar,
        d_pt_off, d_pt_ptr, d_pt_age, d_pt_mini, d_pt_m1, d_pt_m2,
        d_pt_llo, d_pt_dl, d_pt_first, d_pt_cm, d_wk, d_wpb, d_imf_bar
    );
    (void)hipStreamSynchronize(s);
}

void opensps_hip_adj_imf_analytic_dp(
    int m0_0based,
    int nc,
    int nt,
    int n_pts_max,
    int n_mod,
    int imf_type,
    int ldw,
    int ldk,
    const int*    d_perm,
    const int*    d_viz,
    const double* d_cz,
    const double* d_wsfh,
    const double* d_pm,
    const double* d_tot,
    const double* d_dtt,
    const double* d_msum_bar,
    const int*    d_pt_off,
    const int*    d_pt_ptr,
    const int*    d_pt_age,
    const double* d_pt_mini,
    const double* d_pt_m1,
    const double* d_pt_m2,
    const double* d_pt_llo,
    const double* d_pt_dl,
    const int*    d_pt_first,
    const double* d_pt_cm,
    const double* d_wk,
    const double* d_wpb,
    double*       d_imf_bar,
    void* stream
) {
    if (nc <= 0 || n_pts_max <= 0) return;
    dim3 block(32, 8);
    int grid_j = (nc + 31) / 32;
    int grid_p = 64;
    dim3 grid(grid_j, grid_p);
    hipStream_t s = (hipStream_t)stream;
    adj_imf_analytic_kernel<double><<<grid, block, 0, s>>>(
        m0_0based, nc, nt, n_pts_max, n_mod, imf_type, ldw, ldk,
        d_perm, d_viz, d_cz, d_wsfh, d_pm, d_tot, d_dtt, d_msum_bar,
        d_pt_off, d_pt_ptr, d_pt_age, d_pt_mini, d_pt_m1, d_pt_m2,
        d_pt_llo, d_pt_dl, d_pt_first, d_pt_cm, d_wk, d_wpb, d_imf_bar
    );
    (void)hipStreamSynchronize(s);
}

// ---------------------------------------------------------------------------
// Adjoint L_bol & Q_ion (adj_lq)
// ---------------------------------------------------------------------------

__global__ void adj_lq_kernel(
    int           m0_0based,
    int           nc,
    int           nt,
    int           nt_neb,
    int           ntq,
    int           n_pts_max,
    int           n_mod,
    int           ldw,
    const int*    __restrict__ d_perm,
    const int*    __restrict__ d_viz,
    const double* __restrict__ d_cz,
    const double* __restrict__ d_zsg,
    const int*    __restrict__ d_pt_off,
    const int*    __restrict__ d_pt_ptr,
    const int*    __restrict__ d_pt_age,
    const double* __restrict__ d_pt_cl,
    const double* __restrict__ d_pt_q,
    const double* __restrict__ d_wsfh,
    const double* __restrict__ d_slum,
    const double* __restrict__ d_lsum_bar,
    const double* __restrict__ d_qbar,
    const double* __restrict__ d_slz,
    const double* __restrict__ d_sqz,
    double*       __restrict__ d_wpb,
    double*       __restrict__ d_zbar
) {
    int tj = threadIdx.x; // 0 .. 31
    int tp = threadIdx.y; // 0 .. 7
    int j = blockIdx.x * blockDim.x + tj;
    bool valid_j = (j < nc);

    int idx = m0_0based + (valid_j ? j : 0);
    int m = valid_j ? (d_perm[idx] - 1) : 0;
    int iz = valid_j ? d_viz[idx] : 1;
    int iz_0 = iz - 1;
    int n_pts = valid_j ? (d_pt_ptr[iz_0 * (nt + 1) + nt] - 1) : 0;

    double cz_m = valid_j ? d_cz[idx] : 0.0;
    double lsum_m = valid_j ? d_lsum_bar[m] : 0.0;

    __shared__ double s_lf[128][32];
    __shared__ double s_qf[64][32];

    for (int t = tp; t < nt; t += blockDim.y) {
        double lf = 0.0;
        if (valid_j) {
            double sl = d_slz[(size_t)j * nt + t];
            if (sl > 2.2250738585072014e-308) {
                lf = lsum_m * d_wsfh[(size_t)t * n_mod + m] * d_slum[(size_t)m * nt + t] * cz_m / sl;
            }
        }
        s_lf[t][tj] = lf;
    }

    int max_q_ages = (nt_neb < 64) ? nt_neb : 64;
    for (int t = tp; t < max_q_ages; t += blockDim.y) {
        double qf = 0.0;
        if (valid_j && t < nt_neb && t < ntq) {
            qf = d_qbar[(size_t)m * ntq + t] * cz_m;
        }
        s_qf[t][tj] = qf;
    }

    __syncthreads();

    if (blockIdx.y == 0 && tp == 0 && valid_j) {
        double zs = 0.0;
        for (int t = 0; t < nt; ++t) {
            double sl = d_slz[(size_t)j * nt + t];
            if (sl > 2.2250738585072014e-308) {
                zs += lsum_m * d_wsfh[(size_t)t * n_mod + m] * d_slum[(size_t)m * nt + t] * 2.30258509299404568402 * log10(sl);
            }
            if (t < nt_neb && t < ntq) {
                zs += d_qbar[(size_t)m * ntq + t] * d_sqz[(size_t)j * nt + t];
            }
        }
        atomicAdd(&d_zbar[m], d_zsg[idx] * zs);
    }

    int p_stride = gridDim.y * blockDim.y;
    for (int p_0 = blockIdx.y * blockDim.y + tp; p_0 < n_pts_max; p_0 += p_stride) {
        if (!valid_j || p_0 >= n_pts) continue;

        int gp_0 = d_pt_off[iz_0] + p_0;
        int t = d_pt_age[gp_0] - 1;

        double lf = (t >= 0 && t < nt) ? s_lf[t][tj] : 0.0;
        double qf = (t >= 0 && t < max_q_ages) ? s_qf[t][tj] : 0.0;
        double cl = d_pt_cl[gp_0];
        double q  = d_pt_q[gp_0];

        size_t wp_idx = (size_t)p_0 * ldw + j;
        d_wpb[wp_idx] += lf * cl + qf * q;
    }
}

void opensps_hip_adj_lq(
    int           m0_0based,
    int           nc,
    int           nt,
    int           nt_neb,
    int           ntq,
    int           n_pts_max,
    int           n_mod,
    int           ldw,
    const int*    d_perm,
    const int*    d_viz,
    const double* d_cz,
    const double* d_zsg,
    const int*    d_pt_off,
    const int*    d_pt_ptr,
    const int*    d_pt_age,
    const double* d_pt_cl,
    const double* d_pt_q,
    const double* d_wsfh,
    const double* d_slum,
    const double* d_lsum_bar,
    const double* d_qbar,
    const double* d_slz,
    const double* d_sqz,
    double*       d_wpb,
    double*       d_zbar,
    void*         stream
) {
    if (nc <= 0 || n_pts_max <= 0) return;
    dim3 block(32, 8);
    int grid_j = (nc + 31) / 32;
    int grid_p = 64;
    dim3 grid(grid_j, grid_p);
    hipStream_t s = (hipStream_t)stream;
    adj_lq_kernel<<<grid, block, 0, s>>>(
        m0_0based, nc, nt, nt_neb, ntq, n_pts_max, n_mod, ldw,
        d_perm, d_viz, d_cz, d_zsg, d_pt_off, d_pt_ptr, d_pt_age,
        d_pt_cl, d_pt_q, d_wsfh, d_slum, d_lsum_bar, d_qbar,
        d_slz, d_sqz, d_wpb, d_zbar
    );
    (void)hipStreamSynchronize(s);
}

} // extern "C"

// ---------------------------------------------------------------------------
// Batched IMF pieces weights & partial sums (imf_pieces_multi Loop 2)
// ---------------------------------------------------------------------------

__device__ inline double integrate_imf_number(int t_imf, const double* p, double lo, double hi) {
    if (hi <= lo) return 0.0;
    if (t_imf == 0) {
        return integrate_powerlaw(lo, hi, p[0], false, 1.0);
    } else if (t_imf == 2) {
        double a1 = p[1], a2 = p[2], a3 = p[3];
        double c2 = pow(0.5, -a1 + a2);
        double s = 0.0;
        double s1_lo = fmax(lo, 0.08), s1_hi = fmin(hi, 0.5);
        if (s1_hi > s1_lo) s += integrate_powerlaw(s1_lo, s1_hi, a1, false, 1.0);
        double s2_lo = fmax(lo, 0.5), s2_hi = fmin(hi, 1.0);
        if (s2_hi > s2_lo) s += integrate_powerlaw(s2_lo, s2_hi, a2, false, c2);
        double s3_lo = fmax(lo, 1.0);
        if (hi > s3_lo) s += integrate_powerlaw(s3_lo, hi, a3, false, c2);
        return s;
    } else if (t_imf == 4) {
        double a1 = p[1], a2 = p[2];
        double md = p[5];
        double s = 0.0;
        double s1_lo = fmax(lo, 0.08), s1_hi = fmin(hi, md);
        if (s1_hi > s1_lo) s += integrate_powerlaw(s1_lo, s1_hi, a1, false, 1.0);
        double s2_lo = fmax(lo, md);
        if (hi > s2_lo) s += integrate_powerlaw(s2_lo, hi, a2, false, pow(md, -a1 + a2));
        return s;
    }
    return 0.0;
}

__device__ inline double expm1_series_hip(double y) {
    if (fabs(y) < 0.25) {
        double r = 1.0 + y * k_inv_n[13];
        r = 1.0 + y * k_inv_n[12] * r;
        r = 1.0 + y * k_inv_n[11] * r;
        r = 1.0 + y * k_inv_n[10] * r;
        r = 1.0 + y * k_inv_n[9] * r;
        r = 1.0 + y * k_inv_n[8] * r;
        r = 1.0 + y * k_inv_n[7] * r;
        r = 1.0 + y * k_inv_n[6] * r;
        r = 1.0 + y * k_inv_n[5] * r;
        r = 1.0 + y * k_inv_n[4] * r;
        r = 1.0 + y * k_inv_n[3] * r;
        r = 1.0 + y * k_inv_n[2] * r;
        return y * r;
    } else {
        return exp(y) - 1.0;
    }
}

template <typename T_WK>
__global__ void __launch_bounds__(128, 8) imf_pieces_weights_kernel(
    int m0_0based,
    int nc,
    int n_pc_max,
    int n_pts_max,
    int ldw,
    int imf_type,
    const int*    __restrict__ d_perm,
    const int*    __restrict__ d_viz,
    const double* __restrict__ d_pm,
    const double* __restrict__ d_tot,
    const int*    __restrict__ d_n_pc,
    const int*    __restrict__ d_pt_off,
    const int*    __restrict__ d_pc_off,
    const int*    __restrict__ d_pc_ptr,
    const double* __restrict__ d_pt_mini,
    const double* __restrict__ d_pt_m1,
    const double* __restrict__ d_pt_m2,
    const double* __restrict__ d_pt_llo,
    const double* __restrict__ d_pt_dl,
    const int*    __restrict__ d_pt_first,
    const double* __restrict__ d_pt_cm,
    const double* __restrict__ d_pt_cl,
    const double* __restrict__ d_pt_q,
    double*       __restrict__ d_w,
    T_WK*         __restrict__ d_wk,
    double*       __restrict__ d_pps
) {
    int tj = threadIdx.x; // 0..31 (model within the block)
    int tq = threadIdx.y; // 0..3 (piece within block)
    int j = blockIdx.x * blockDim.x + tj;   // model of the chunk (a chunk has up to 2048)
    int q = blockIdx.y * blockDim.y + tq; // 0..n_pc_max-1

    bool valid_j = (j < nc);
    if (!valid_j || q >= n_pc_max) return;

    int idx = m0_0based + j;
    int m = d_perm[idx] - 1;
    int iz = d_viz[idx];
    int iz_0 = iz - 1;

    if (q >= d_n_pc[iz_0]) return;

    int pc_off_val = d_pc_off[iz_0];
    int k0_0 = d_pc_ptr[pc_off_val + q] - 1;
    int k1_0 = d_pc_ptr[pc_off_val + q + 1] - 1;
    int n = k1_0 - k0_0;
    int o_0 = d_pt_off[iz_0];

    const double* p = d_pm + (size_t)m * 312;
    int t_imf = imf_type % 10;
    double lim_lo = p[6];
    double lim_hi = p[7];
    double p_bound = p[8];
    double tot_val = d_tot[j];
    double inv_total = (tot_val > 0.0) ? (1.0 / tot_val) : 1.0;

    double sm = 0.0, sl = 0.0, sq = 0.0;
    int cur = 0;
    double e = 0.0, c = 0.0, d = 1.0, ce = 0.0;
    double prev_hi = -1.0, prev_phi = 0.0;

    for (int k = 0; k < n; ++k) {
        int gp = k0_0 + k;
        int kl = gp - o_0;
        size_t out_idx = (size_t)kl * ldw + j;

        // d_w (the FP64 weights) is only the reverse pass's: a forward-only call passes a null pointer
        if (d_w) d_w[out_idx] = 0.0;
        d_wk[out_idx] = (T_WK)0.0;

        double mini = d_pt_mini[gp];
        if (mini < lim_lo || mini > lim_hi) continue;

        int first = d_pt_first[gp];
        double lo = (first == 1) ? p_bound : d_pt_m1[gp];
        double hi = d_pt_m2[gp];
        if (hi <= lo) continue;

        int seg = 0;
        if (t_imf == 0) {
            seg = 1;
        } else if (t_imf == 1) {
            if (lo >= 1.0) seg = 2;
        } else if (t_imf == 3) {
            if (lo >= 25.0 * p[4]) seg = 2;
        } else if (t_imf == 2) {
            if (lo >= 0.08 && hi <= 0.5) seg = 1;
            else if (lo >= 0.5 && hi <= 1.0) seg = 2;
            else if (lo >= 1.0) seg = 3;
        } else if (t_imf == 4) {
            double md = p[5];
            if (lo >= 0.08 && hi <= md) seg = 1;
            else if (lo >= md) seg = 2;
        }

        if (seg > 0 && seg != cur) {
            double alpha = 0.0, coeff = 1.0;
            if (t_imf == 0) {
                alpha = p[0];
            } else if (t_imf == 1) {
                alpha = 2.3;
                coeff = 0.28263124794373684;
            } else if (t_imf == 3) {
                alpha = 2.3;
                coeff = 0.0443;
            } else if (t_imf == 2) {
                alpha = p[seg];
                if (seg > 1) coeff = pow(0.5, -p[1] + p[2]);
            } else {
                alpha = p[seg];
                double md = p[5];
                if (seg > 1) coeff = pow(md, -p[1] + p[2]);
            }
            e = 1.0 - alpha;
            c = coeff;
            d = 1.0 - alpha;
            ce = (e != 0.0) ? (c / e) : 0.0;
            cur = seg;
            prev_hi = -1.0;
        }

        double val = 0.0;
        if (seg > 0 && cur == seg) {
            if (first == 1) {
                if (fabs(d) <= 1.0e-12) {
                    val = c * log(hi / lo);
                } else {
                    val = c * (pow(hi, e) - pow(lo, e)) / d;
                }
            } else if (e == 0.0) {
                val = c * d_pt_dl[gp];
            } else {
                double plo;
                if (lo == prev_hi) {
                    plo = prev_phi;
                } else {
                    plo = exp(e * d_pt_llo[gp]);
                }
                double em1 = expm1_series_hip(e * d_pt_dl[gp]);
                val = ce * plo * em1;
                prev_hi = hi;
                prev_phi = plo + plo * em1;
            }
        } else {
            val = integrate_imf_number(t_imf, p, lo, hi);
            prev_hi = -1.0;
            if (seg == 0) cur = 0;
        }
        double wv = val * inv_total;
        if (d_w) d_w[out_idx] = wv;
        d_wk[out_idx] = (T_WK)wv;
        sm += d_pt_cm[gp] * wv;
        sl += d_pt_cl[gp] * wv;
        sq += d_pt_q[gp] * wv;
    }

    size_t pps_base = ((size_t)q * 3) * ldw + j;
    d_pps[pps_base + 0 * ldw] = sm;
    d_pps[pps_base + 1 * ldw] = sl;
    d_pps[pps_base + 2 * ldw] = sq;
}


// ---------------------------------------------------------------------------
// FP32-arithmetic point weights (the forward-only call: the FP64 array w is not wanted)
//
// The same weights as imf_pieces_weights_kernel, with the per-point arithmetic in FP32, because the
// R9700's FP64 rate is 1/64 of its FP32 rate and the FP64 kernel is limited by it. Where FP32 would lose
// accuracy the kernel does not: every argument of an exponential is formed as an unevaluated sum of two
// FP32 numbers (hi + lo; the exact product error comes from an fma), so that
//   plo = lo^e  = exp(p) (1 + r),          p + r = e ln(m1)        (r ~ 1e-7 |p|, p the FP32 product)
//   em1 = expm1(y) + (1 + expm1(y)) yr,    y + yr = e ln(m2 / m1)
// are good to the rounding of expf / expm1f (1-2 ulp) instead of |e ln m| times the FP32 epsilon (4e-7
// for the largest arguments). The bin weights are independent (no chain of products, which would
// accumulate n roundings). The piece's sums of the mass, L_bol and Q coefficients are Neumaier sums of
// the exact products (fma error terms) in FP32, Q scaled by 2^-80 (photon rates exceed the FP32 range),
// and widened to FP64 at the end. The rare points where the closed form is not a power law of one
// segment (the first bin of an age, bins that straddle a break of the IMF) take the FP64 path of the
// original kernel. The per-point tests are done on the bit patterns of the (positive) FP64 values: an
// FP64 comparison also runs at the FP64 rate.
// ---------------------------------------------------------------------------

__device__ inline bool lt_pos(double a, double b) { return __double_as_longlong(a) < __double_as_longlong(b); }
__device__ inline bool le_pos(double a, double b) { return __double_as_longlong(a) <= __double_as_longlong(b); }

// (Products whose rounding error is taken with an fma are formed with __fmul_rn: a plain a * b + c is
// contracted into one fma by the compiler, which would count the rounding error twice.)
__device__ inline void neumaier_add(float& s, float& c, float x) {
    #pragma clang fp contract(off)
    float t = s + x;
    float bp = t - s;
    c += (s - (t - bp)) + (x - bp);
    s = t;
}

__global__ void __launch_bounds__(128, 8) imf_pieces_weights_f32_kernel(
    int m0_0based,
    int nc,
    int n_pc_max,
    int ldw,
    int imf_type,
    const int*    __restrict__ d_perm,
    const int*    __restrict__ d_viz,
    const double* __restrict__ d_pm,
    const double* __restrict__ d_tot,
    const int*    __restrict__ d_n_pc,
    const int*    __restrict__ d_pt_off,
    const int*    __restrict__ d_pc_off,
    const int*    __restrict__ d_pc_ptr,
    const double* __restrict__ d_pt_mini,
    const double* __restrict__ d_pt_m1,
    const double* __restrict__ d_pt_m2,
    const int*    __restrict__ d_pt_first,
    const float*  __restrict__ d_ptf,   // per point: ln m1 (hi, lo), ln(m2/m1) (hi, lo), cm, cl, q 2^-80, 0
    float*        __restrict__ d_wk,
    double*       __restrict__ d_pps
) {
    #pragma clang fp contract(off)
    int tj = threadIdx.x;
    int tq = threadIdx.y;
    int j = blockIdx.x * blockDim.x + tj;
    int q = blockIdx.y * blockDim.y + tq;
    if (j >= nc || q >= n_pc_max) return;

    int idx = m0_0based + j;
    int m = d_perm[idx] - 1;
    int iz_0 = d_viz[idx] - 1;
    if (q >= d_n_pc[iz_0]) return;

    int pc_off_val = d_pc_off[iz_0];
    int k0_0 = d_pc_ptr[pc_off_val + q] - 1;
    int k1_0 = d_pc_ptr[pc_off_val + q + 1] - 1;
    int n = k1_0 - k0_0;
    int o_0 = d_pt_off[iz_0];

    const double* p = d_pm + (size_t)m * 312;
    int t_imf = imf_type % 10;
    double lim_lo = p[6];
    double lim_hi = p[7];
    double p_bound = p[8];
    double md = (t_imf == 4) ? p[5] : 0.0;
    double tot_val = d_tot[j];
    double inv_total = (tot_val > 0.0) ? (1.0 / tot_val) : 1.0;

    float sm = 0.f, cm_ = 0.f, sl = 0.f, cl_ = 0.f, sq = 0.f, cq_ = 0.f;
    int cur = 0;
    double e = 0.0, c = 0.0, d = 1.0, ce = 0.0;
    float ehi = 0.f, elo = 0.f, scf = 0.f, cf = 0.f;

    for (int k = 0; k < n; ++k) {
        int gp = k0_0 + k;
        size_t out_idx = (size_t)(gp - o_0) * ldw + j;

        if (lt_pos(d_pt_mini[gp], lim_lo) || lt_pos(lim_hi, d_pt_mini[gp])) { d_wk[out_idx] = 0.f; continue; }

        int first = d_pt_first[gp];
        double lo = (first == 1) ? p_bound : d_pt_m1[gp];
        double hi = d_pt_m2[gp];
        if (!lt_pos(lo, hi)) { d_wk[out_idx] = 0.f; continue; }

        int seg = 0;
        if (t_imf == 0) {
            seg = 1;
        } else if (t_imf == 2) {
            if (!lt_pos(lo, 0.08) && le_pos(hi, 0.5)) seg = 1;
            else if (!lt_pos(lo, 0.5) && le_pos(hi, 1.0)) seg = 2;
            else if (!lt_pos(lo, 1.0)) seg = 3;
        } else {
            if (!lt_pos(lo, 0.08) && le_pos(hi, md)) seg = 1;
            else if (!lt_pos(lo, md)) seg = 2;
        }

        if (seg > 0 && seg != cur) {
            double alpha, coeff = 1.0;
            if (t_imf == 0) {
                alpha = p[0];
            } else if (t_imf == 2) {
                alpha = p[seg];
                if (seg > 1) coeff = pow(0.5, -p[1] + p[2]);
            } else {
                alpha = p[seg];
                if (seg > 1) coeff = pow(md, -p[1] + p[2]);
            }
            e = 1.0 - alpha;
            c = coeff;
            d = 1.0 - alpha;
            ce = (e != 0.0) ? (c / e) : 0.0;
            cur = seg;
            ehi = (float)e;
            elo = (float)(e - (double)ehi);
            scf = (float)(ce * inv_total);
            cf = (float)(c * inv_total);
        }

        float wv;
        if (seg > 0 && cur == seg && first != 1) {
            const float4 a = *reinterpret_cast<const float4*>(d_ptf + (size_t)gp * 8);
            if (e == 0.0) {
                wv = cf * (a.z + a.w);
            } else {
                float pp = __fmul_rn(ehi, a.x);
                float pe = fmaf(ehi, a.x, -pp) + (ehi * a.y + elo * a.x);
                float ex = expf(pp);
                float plo = fmaf(ex, pe, ex);
                float y = __fmul_rn(ehi, a.z);
                float ye = fmaf(ehi, a.z, -y) + (ehi * a.w + elo * a.z);
                float em1 = expm1f(y);
                em1 = fmaf(em1 + 1.0f, ye, em1);
                wv = scf * plo * em1;
            }
        } else {
            double val;
            if (seg > 0 && cur == seg) {   // first bin of an age
                if (fabs(d) <= 1.0e-12) val = c * log(hi / lo);
                else val = c * (pow(hi, e) - pow(lo, e)) / d;
            } else {
                val = integrate_imf_number(t_imf, p, lo, hi);
                if (seg == 0) cur = 0;
            }
            wv = (float)(val * inv_total);
        }
        d_wk[out_idx] = wv;
        const float4 b = *reinterpret_cast<const float4*>(d_ptf + (size_t)gp * 8 + 4);
        float x, ex_;
        x = __fmul_rn(b.x, wv); ex_ = fmaf(b.x, wv, -x); cm_ += ex_; neumaier_add(sm, cm_, x);
        x = __fmul_rn(b.y, wv); ex_ = fmaf(b.y, wv, -x); cl_ += ex_; neumaier_add(sl, cl_, x);
        x = __fmul_rn(b.z, wv); ex_ = fmaf(b.z, wv, -x); cq_ += ex_; neumaier_add(sq, cq_, x);
    }

    size_t pps_base = ((size_t)q * 3) * ldw + j;
    d_pps[pps_base + 0 * ldw] = (double)sm + (double)cm_;
    d_pps[pps_base + 1 * ldw] = (double)sl + (double)cl_;
    d_pps[pps_base + 2 * ldw] = ((double)sq + (double)cq_) * 1.2089258196146292e24;   // 2^80
}


// ---------------------------------------------------------------------------
// Forward fold of the per-point weights into template weights (fold_multi, first loop)
//
// vp(j, 1..4, s) = young / old / young-EUV / old-EUV sums of segment s of virtual model j:
// the sum over the segment's entries (template e, point p, value eff_val), grouped by the age t of
// the point, of eff_val w(j, p) times wsfh(m, t) cz(m), split at the escape age tesc.
// Per age the entries are summed in FP32 from exact products (eff_val as an FP32 hi + lo pair, fma
// error terms, Neumaier) and widened to FP64: the same value as the FP64 sum to ~1e-14, with no FP64
// work per entry (FP64 is 1/64 of the FP32 rate on the R9700). One block = 128 consecutive models of
// one segment; the grid runs the segments fastest, so the blocks of one model range are neighbours in
// time (the weights of 128 models, 54 MB, are reused by the ~7.6 segments sharing a point).
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(128, 8) fold_multi_kernel(
    int m0_0based, int nc, int nt_neb, int ldw, int n_mod_ld, int ldv, int nseg_max,
    const int*    __restrict__ d_perm,
    const int*    __restrict__ d_viz,
    const int*    __restrict__ d_n_seg,
    const int*    __restrict__ d_pt_off,
    const int*    __restrict__ d_seg_off,
    const int*    __restrict__ d_pt_age,
    const int*    __restrict__ d_eff_pt,
    const int*    __restrict__ d_seg_ptr,
    const float2* __restrict__ d_eff_f,
    const double* __restrict__ d_wsfh,
    const int*    __restrict__ d_it_tesc,
    const double* __restrict__ d_fesc,
    const double* __restrict__ d_cz,
    const float*  __restrict__ d_w,
    double*       __restrict__ d_vp
) {
    #pragma clang fp contract(off)
    int s = blockIdx.x;                         // 0-based segment
    int j = blockIdx.y * blockDim.x + threadIdx.x;
    if (j >= nc) return;
    int idx = m0_0based + j;
    int iz0 = d_viz[idx] - 1;
    if (s >= d_n_seg[iz0]) return;
    int m = d_perm[idx] - 1;
    int sg = d_seg_off[iz0] + s;                // 0-based index of the segment pointer
    int tesc = d_it_tesc[m];
    double f = d_fesc[m];
    double cz = d_cz[m];
    int k0 = d_seg_ptr[sg] - 1, k1 = d_seg_ptr[sg + 1] - 1;   // entries k0 .. k1 - 1 (0-based)
    int po = d_pt_off[iz0];
    double y = 0.0, o = 0.0, ye = 0.0, oe = 0.0;
    int tc = 0;
    float as = 0.f, ac = 0.f;
    for (int k = k0; k <= k1; ++k) {
        int t = -1, ep = 0;
        if (k < k1) { ep = d_eff_pt[k] - 1; t = d_pt_age[po + ep]; }
        if (t != tc) {
            if (tc > 0) {
                double a = (double)as + (double)ac;
                double x = a * d_wsfh[m + (size_t)n_mod_ld * (tc - 1)] * cz;
                double ae = (tc <= nt_neb) ? f * x : x;
                if (tc <= tesc) { y += x; ye += ae; } else { o += x; oe += ae; }
            }
            tc = t; as = 0.f; ac = 0.f;
        }
        if (t > 0) {
            float2 ev = d_eff_f[k];
            float w = d_w[(size_t)ep * ldw + j];
            float p1 = __fmul_rn(ev.x, w);
            float e1 = fmaf(ev.x, w, -p1);
            float tt = as + p1;
            float bp = tt - as;
            ac += (as - (tt - bp)) + (p1 - bp) + e1 + ev.y * w;
            as = tt;
        }
    }
    size_t base = (size_t)s * 4 * ldv + j;
    d_vp[base] = y;
    d_vp[base + ldv] = o;
    d_vp[base + 2 * (size_t)ldv] = ye;
    d_vp[base + 3 * (size_t)ldv] = oe;
}


// ---------------------------------------------------------------------------
// Template products yo(i, c0 + j) = sum_e etpl(i, e) v(e, v0 + j)   (template_columns, FP32 kernels)
//
// The arithmetic of the Fortran device kernel: the templates in blocks of KB, each block summed with
// plain multiply-adds (all terms are >= 0: a block loses at most ~KB eps of itself), the blocks added
// to the total with compensation (rule 1 of doc/MIXED_PRECISION.md). The tile of a block is
// 16 RM pixels x 16 RN columns (RM x RN per thread), the templates staged through shared memory KC at a time
// (each etpl element read once per column tile instead of once per 8 columns), the next stage's global
// loads issued before the current stage's multiply-adds.
// ---------------------------------------------------------------------------
template <int RM, int RN, int KC, int KB>
__global__ void __launch_bounds__(256, 2) template_columns_kernel(
    int np, int nc, int k, int lda, int ldb, int ldc,
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C
) {
    constexpr int BM = 16 * RM, BN = 16 * RN;
    __shared__ float As[KC][BM];
    __shared__ float Bs[KC][BN];
    const int tid = threadIdx.x;
    const int tx = tid & 15, ty = tid >> 4;
    const int i0 = blockIdx.x * BM, j0 = blockIdx.y * BN;

    float acc[RM][RN], cmp[RM][RN], blk[RM][RN];
#pragma unroll
    for (int r = 0; r < RM; ++r)
#pragma unroll
        for (int c = 0; c < RN; ++c) { acc[r][c] = 0.f; cmp[r][c] = 0.f; }

    float pa[KC * BM / 256], pb[KC * BN / 256];
    auto gload = [&](int e0) {
#pragma unroll
        for (int q = 0; q < KC * BM / 256; ++q) {
            int idx = tid + q * 256, i = idx % BM, e = idx / BM;
            pa[q] = (i0 + i < np && e0 + e < k) ? A[(size_t)(e0 + e) * lda + i0 + i] : 0.f;
        }
#pragma unroll
        for (int q = 0; q < KC * BN / 256; ++q) {
            int idx = tid + q * 256, e = idx % KC, j = idx / KC;
            pb[q] = (j0 + j < nc && e0 + e < k) ? B[(size_t)(j0 + j) * ldb + e0 + e] : 0.f;
        }
    };
    gload(0);
    for (int e0 = 0; e0 < k; e0 += KC) {
        __syncthreads();
#pragma unroll
        for (int q = 0; q < KC * BM / 256; ++q) {
            int idx = tid + q * 256; As[idx / BM][idx % BM] = pa[q];
        }
#pragma unroll
        for (int q = 0; q < KC * BN / 256; ++q) {
            int idx = tid + q * 256; Bs[idx % KC][idx / KC] = pb[q];
        }
        __syncthreads();
        if (e0 + KC < k) gload(e0 + KC);
#pragma unroll
        for (int g = 0; g < KC / KB; ++g) {
            if (e0 + g * KB >= k) break;
#pragma unroll
            for (int r = 0; r < RM; ++r)
#pragma unroll
                for (int c = 0; c < RN; ++c) blk[r][c] = 0.f;
#pragma unroll
            for (int u = 0; u < KB; ++u) {
                const int e = g * KB + u;
                float av[RM], bv[RN];
#pragma unroll
                for (int r = 0; r < RM; ++r) av[r] = As[e][tx + 16 * r];
#pragma unroll
                for (int c = 0; c < RN; ++c) bv[c] = Bs[e][ty + 16 * c];
#pragma unroll
                for (int r = 0; r < RM; ++r)
#pragma unroll
                    for (int c = 0; c < RN; ++c) blk[r][c] = fmaf(av[r], bv[c], blk[r][c]);
            }
#pragma unroll
            for (int r = 0; r < RM; ++r)
#pragma unroll
                for (int c = 0; c < RN; ++c) {
                    float term = blk[r][c] - cmp[r][c];
                    float tsum = acc[r][c] + term;
                    cmp[r][c] = (tsum - acc[r][c]) - term;
                    acc[r][c] = tsum;
                }
        }
    }
#pragma unroll
    for (int c = 0; c < RN; ++c) {
        int j = j0 + ty + 16 * c;
        if (j < nc) {
#pragma unroll
            for (int r = 0; r < RM; ++r) {
                int i = i0 + tx + 16 * r;
                if (i < np) C[(size_t)j * ldc + i] = acc[r][c];
            }
        }
    }
}


// ---------------------------------------------------------------------------
// Velocity smoothing of the spectra (smooth_kernel, FP32 kernels)
//
// out(i, m) = sum_{k = lo(i)}^{hi(i)} c(i, k) in(k, m), the banded operator of build_velocity_kernel
// (normalised trapezoid weights, all >= 0; spectra are >= 0); pixels with lo(i) = 0 pass through.
// The Fortran kernel summed in FP64 (1/64 of the FP32 rate on the R9700). Here: each coefficient is an
// FP32 hi + lo pair (exact to 1e-14), the taps are summed in blocks of SM_KB with plain multiply-adds
// (non-negative terms: a block loses at most ~2 SM_KB eps of itself), the blocks into the total with
// compensation (doc/MIXED_PRECISION.md, rule 1). A block of 256 threads = 64 pixels x 16 models (4 per
// thread); the spectra of the tile's window are staged through shared memory, 256 pixels at a time.
// ---------------------------------------------------------------------------
// adj = 0: out(i) = sum of its window (pixels with no window pass through); adj = 1: the transposed operator,
// out(k) = [forward window of k empty] in(k) + sum_i C(i, k) in(i), the windows (d_lo, d_hi) and coefficient
// starts (d_off, 1-based when cb1 = 1) being those of the columns of C (d_plo: the forward windows' starts)
template <int PB, int KCH>
__global__ void __launch_bounds__(256) smooth_kernel_f32(
    int nspec, int n_col, int kb, int adj, int cb1,
    const int* __restrict__ d_lo, const int* __restrict__ d_hi, const int* __restrict__ d_off, const int* __restrict__ d_plo,
    const float2* __restrict__ d_cf,
    const float* __restrict__ in, float* __restrict__ out
) {
    constexpr int MB = 4 * (256 / PB);   // models per block: 4 per thread, 256 / PB groups
    __shared__ float sp[MB][KCH];
    __shared__ int s_k0, s_k1;
    const int tid = threadIdx.x;
    const int ip = tid & (PB - 1), mq = tid / PB;
    const int i0 = blockIdx.x * PB, m0 = blockIdx.y * MB;
    const int i = i0 + ip;
    if (tid == 0) { s_k0 = nspec + 1; s_k1 = 0; }
    __syncthreads();
    int lo = 0, hi = -1, off = 0, plo = 1;
    if (i < nspec) { lo = d_lo[i]; hi = d_hi[i]; off = d_off[i] - cb1 - lo; plo = d_plo[i]; }
    const bool has = lo > 0 && hi >= lo;
    if (has) { atomicMin(&s_k0, lo); atomicMax(&s_k1, hi); }
    __syncthreads();
    const int k0 = s_k0, k1 = s_k1;
    float acc[4] = {0.f, 0.f, 0.f, 0.f}, cmp[4] = {0.f, 0.f, 0.f, 0.f}, blk[4] = {0.f, 0.f, 0.f, 0.f};
    if (adj && plo == 0 && i < nspec) {
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            int m = m0 + mq * 4 + q;
            if (m < n_col) acc[q] = in[(size_t)m * nspec + i];
        }
    }
    for (int c0 = k0; c0 <= k1; c0 += KCH) {
        __syncthreads();
        for (int idx = tid; idx < KCH * MB; idx += 256) {
            int row = idx % KCH, mc = idx / KCH;
            int k = c0 + row, m = m0 + mc;
            sp[mc][row] = (k <= k1 && m < n_col) ? in[(size_t)m * nspec + (k - 1)] : 0.f;
        }
        __syncthreads();
        if (has) {
            int ka = max(lo, c0), kend = min(hi, c0 + KCH - 1);   // (not kb: that is the plain block's length)
            for (int k = ka; k <= kend; ++k) {
                float2 cf = d_cf[off + k];
                int t = k - lo;
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    float x = sp[mq * 4 + q][k - c0];
                    blk[q] = fmaf(cf.x, x, blk[q]);
                    blk[q] = fmaf(cf.y, x, blk[q]);
                }
                if ((t + 1) % kb == 0 || k == hi) {
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        float term = blk[q] - cmp[q];
                        float tsum = acc[q] + term;
                        cmp[q] = (tsum - acc[q]) - term;
                        acc[q] = tsum;
                        blk[q] = 0.f;
                    }
                }
            }
        }
    }
    if (i < nspec) {
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            int m = m0 + mq * 4 + q;
            if (m < n_col) out[(size_t)m * nspec + i] = (adj || has) ? acc[q] - cmp[q] : in[(size_t)m * nspec + i];
        }
    }
}

// The tile shapes of the shared-memory products (template products, age sums, band fluxes): the registry's
// HIP_GEMM_RM / RN / KC and TEMPLATE_KB reach the kernels as template arguments; a combination that is not
// instantiated here is not supported and the caller takes its portable kernel (return value 1)
template <int RM_, int RN_, int KC_, int KB_> struct GemmCfg {
    static constexpr int RM = RM_, RN = RN_, KC = KC_, KB = KB_;
};

template <class F> static bool gemm_dispatch(int rm, int rn, int kc, int kb, F f) {
    if (kc != 16) return false;
#define GEMM_CASE(RM, RN, KB) if (rm == RM && rn == RN && kb == KB) { f(GemmCfg<RM, RN, 16, KB>()); return true; }
    GEMM_CASE(8, 4, 4) GEMM_CASE(8, 4, 8) GEMM_CASE(4, 4, 4) GEMM_CASE(4, 4, 8)
    GEMM_CASE(8, 8, 4) GEMM_CASE(8, 8, 8) GEMM_CASE(4, 8, 4) GEMM_CASE(4, 8, 8)
#undef GEMM_CASE
    return false;
}

extern "C" {

void opensps_hip_imf_pieces_weights_sp(
    int m0_0based,
    int nc,
    int n_pc_max,
    int n_pts_max,
    int ldw,
    int imf_type,
    int pieces,
    const int*    d_perm,
    const int*    d_viz,
    const double* d_pm,
    const double* d_tot,
    const int*    d_n_pc,
    const int*    d_pt_off,
    const int*    d_pc_off,
    const int*    d_pc_ptr,
    const double* d_pt_mini,
    const double* d_pt_m1,
    const double* d_pt_m2,
    const double* d_pt_llo,
    const double* d_pt_dl,
    const int*    d_pt_first,
    const double* d_pt_cm,
    const double* d_pt_cl,
    const double* d_pt_q,
    double*       d_w,
    float*        d_wk,
    double*       d_pps,
    const float*  d_ptf,
    void*         stream
) {
    if (nc <= 0 || n_pc_max <= 0) return;
    if (pieces < 1) pieces = 1;
    if (pieces > 4) pieces = 4;   // (the kernels' launch bounds: 32 models x 4 pieces)
    dim3 block(32, pieces);
    int grid_j = (nc + 31) / 32;
    int grid_q = (n_pc_max + pieces - 1) / pieces;
    dim3 grid(grid_j, grid_q);
    hipStream_t s = (hipStream_t)stream;
    if (d_w == nullptr) {   // forward only: FP32 arithmetic, no FP64 weights
        imf_pieces_weights_f32_kernel<<<grid, block, 0, s>>>(
            m0_0based, nc, n_pc_max, ldw, imf_type,
            d_perm, d_viz, d_pm, d_tot, d_n_pc, d_pt_off, d_pc_off, d_pc_ptr,
            d_pt_mini, d_pt_m1, d_pt_m2, d_pt_first, d_ptf, d_wk, d_pps
        );
        (void)hipStreamSynchronize(s);
        return;
    }
    imf_pieces_weights_kernel<float><<<grid, block, 0, s>>>(
        m0_0based, nc, n_pc_max, n_pts_max, ldw, imf_type,
        d_perm, d_viz, d_pm, d_tot, d_n_pc, d_pt_off, d_pc_off, d_pc_ptr,
        d_pt_mini, d_pt_m1, d_pt_m2, d_pt_llo, d_pt_dl, d_pt_first,
        d_pt_cm, d_pt_cl, d_pt_q, d_w, d_wk, d_pps
    );
    (void)hipStreamSynchronize(s);
}

void opensps_hip_imf_pieces_weights_dp(
    int m0_0based,
    int nc,
    int n_pc_max,
    int n_pts_max,
    int ldw,
    int imf_type,
    int pieces,
    const int*    d_perm,
    const int*    d_viz,
    const double* d_pm,
    const double* d_tot,
    const int*    d_n_pc,
    const int*    d_pt_off,
    const int*    d_pc_off,
    const int*    d_pc_ptr,
    const double* d_pt_mini,
    const double* d_pt_m1,
    const double* d_pt_m2,
    const double* d_pt_llo,
    const double* d_pt_dl,
    const int*    d_pt_first,
    const double* d_pt_cm,
    const double* d_pt_cl,
    const double* d_pt_q,
    double*       d_w,
    double*       d_wk,
    double*       d_pps,
    void*         stream
) {
    if (nc <= 0 || n_pc_max <= 0) return;
    if (pieces < 1) pieces = 1;
    if (pieces > 4) pieces = 4;   // (the kernels' launch bounds: 32 models x 4 pieces)
    dim3 block(32, pieces);
    int grid_j = (nc + 31) / 32;
    int grid_q = (n_pc_max + pieces - 1) / pieces;
    dim3 grid(grid_j, grid_q);
    hipStream_t s = (hipStream_t)stream;
    imf_pieces_weights_kernel<double><<<grid, block, 0, s>>>(
        m0_0based, nc, n_pc_max, n_pts_max, ldw, imf_type,
        d_perm, d_viz, d_pm, d_tot, d_n_pc, d_pt_off, d_pc_off, d_pc_ptr,
        d_pt_mini, d_pt_m1, d_pt_m2, d_pt_llo, d_pt_dl, d_pt_first,
        d_pt_cm, d_pt_cl, d_pt_q, d_w, d_wk, d_pps
    );
    (void)hipStreamSynchronize(s);
}

// Returns 0 if run, 1 if the tile shape is not supported (the caller then takes its own kernel)
int opensps_hip_template_columns(int np, int nc, int k, int rm, int rn, int kc, int kb, int lda, int ldb, int ldc,
                                 const float* d_A, const float* d_B, float* d_C, void* stream) {
    if (np <= 0 || nc <= 0 || k <= 0) return 0;
    hipStream_t s = (hipStream_t)stream;
    const bool ok = gemm_dispatch(rm, rn, kc, kb, [&](auto cfg) {
        constexpr int RM = decltype(cfg)::RM, RN = decltype(cfg)::RN, KC = decltype(cfg)::KC, KB = decltype(cfg)::KB;
        dim3 grid((np + 16 * RM - 1) / (16 * RM), (nc + 16 * RN - 1) / (16 * RN));
        template_columns_kernel<RM, RN, KC, KB><<<grid, 256, 0, s>>>(np, nc, k, lda, ldb, ldc, d_A, d_B, d_C);
    });
    if (!ok) return 1;
    (void)hipStreamSynchronize(s);
    return 0;
}

// pb pixels per block and kch window pixels per stage; kb taps per plain block. Returns 1 if not supported
int opensps_hip_smooth(int nspec, int n_col, int pb, int kch, int kb, int adj, int cb1, const int* d_lo, const int* d_hi,
                       const int* d_off, const int* d_plo, const float* d_cf, const float* d_in, float* d_out, void* stream) {
    if (nspec <= 0 || n_col <= 0) return 0;
    if (kb < 1) return 1;
    hipStream_t s = (hipStream_t)stream;
    const float2* cf = reinterpret_cast<const float2*>(d_cf);
#define SM_CASE(PB, KCH) if (pb == PB && kch == KCH) { \
        constexpr int MB = 4 * (256 / PB); \
        smooth_kernel_f32<PB, KCH><<<dim3((nspec + PB - 1) / PB, (n_col + MB - 1) / MB), 256, 0, s>>>( \
            nspec, n_col, kb, adj, cb1, d_lo, d_hi, d_off, d_plo, cf, d_in, d_out); \
        (void)hipStreamSynchronize(s); return 0; }
    SM_CASE(64, 256) SM_CASE(64, 128) SM_CASE(64, 512) SM_CASE(32, 256) SM_CASE(32, 128)
#undef SM_CASE
    return 1;
}

void opensps_hip_fold_multi(
    int m0_0based, int nc, int nt_neb, int ldw, int n_mod_ld, int ldv, int nseg_max, int jb,
    const int* d_perm, const int* d_viz, const int* d_n_seg, const int* d_pt_off, const int* d_seg_off,
    const int* d_pt_age, const int* d_eff_pt, const int* d_seg_ptr, const float* d_eff_f,
    const double* d_wsfh, const int* d_it_tesc, const double* d_fesc, const double* d_cz,
    const float* d_w, double* d_vp, void* stream
) {
    if (nc <= 0 || nseg_max <= 0) return;
    if (jb < 32) jb = 32;
    if (jb > 128) jb = 128;   // (the kernel's launch bounds)
    dim3 block(jb);
    dim3 grid(nseg_max, (nc + jb - 1) / jb);
    hipStream_t s = (hipStream_t)stream;
    fold_multi_kernel<<<grid, block, 0, s>>>(
        m0_0based, nc, nt_neb, ldw, n_mod_ld, ldv, nseg_max,
        d_perm, d_viz, d_n_seg, d_pt_off, d_seg_off, d_pt_age, d_eff_pt, d_seg_ptr,
        reinterpret_cast<const float2*>(d_eff_f), d_wsfh, d_it_tesc, d_fesc, d_cz, d_w, d_vp
    );
    (void)hipStreamSynchronize(s);
}

} // extern "C"

// ---------------------------------------------------------------------------
// Band Fluxes Native HIP Kernel
// ---------------------------------------------------------------------------

template <typename T_KP>
__global__ void __launch_bounds__(256, 4) band_fluxes_kernel(
    int n_mod,
    int nspec,
    int nbands,
    const T_KP* __restrict__ d_bands_t,
    const int*  __restrict__ d_j_lo,
    const int*  __restrict__ d_j_hi,
    const int*  __restrict__ d_do_band,
    const T_KP* __restrict__ d_wt,
    const T_KP* __restrict__ d_fobs_t,
    double*     __restrict__ d_fband
) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;

    if (b >= nbands || m >= n_mod) return;

    if (d_do_band[b] == 0) {
        d_fband[(size_t)m * nbands + b] = 0.0;
        return;
    }

    int j_start = d_j_lo[b] - 1; // 0-based
    int j_end   = d_j_hi[b] - 1; // 0-based

    T_KP s = 0;
    T_KP c = 0;

    size_t fobs_base = (size_t)m;
    size_t bands_base = (size_t)b;

    // Prefetch first pixel of the band
    T_KP b_curr = (j_start < nspec) ? d_bands_t[bands_base + (size_t)j_start * nbands] : (T_KP)0;
    T_KP f_curr = (j_start < nspec) ? d_fobs_t[fobs_base + (size_t)j_start * n_mod] : (T_KP)0;

    for (int j = j_start; j <= j_end; ++j) {
        T_KP w = d_wt[j];
        T_KP b_next = (j + 1 < nspec) ? d_bands_t[bands_base + (size_t)(j + 1) * nbands] : (T_KP)0;
        T_KP f_next = (j + 1 < nspec) ? d_fobs_t[fobs_base + (size_t)(j + 1) * n_mod] : (T_KP)0;

        T_KP term = w * (f_curr * b_curr + f_next * b_next);
        kahan_add(term, s, c);

        b_curr = b_next;
        f_curr = f_next;
    }

    d_fband[(size_t)m * nbands + b] = (double)s;
}


// Small batches: one workgroup per (model, band). The one-thread-per-(model, band) kernel above
// leaves the GPU almost idle when there are few models (N = 1: one working lane per band, each
// summing up to thousands of pixels serially). Here the pixels of a band are dealt out to the
// threads of a workgroup (thread t takes pixels j_lo + t, + BF_SMALL_THREADS, ...), each thread sums
// its share serially (compensated at FP32), and the partial sums are combined by a tree in a fixed
// order in LDS. The order of every addition depends only on the indices, never on the scheduling,
// so identical batches give identical results, and the tree sum is as accurate as the serial one.
#define BF_SMALL_THREADS 256
// (Batches of fewer models than HIP_BAND_SMALL_N use the workgroup kernel; R9700, FP32, us per model: N = 32
// 64 vs 69 for the other kernel, N = 48 49 vs 47, N = 100 37 vs 26)

template <typename T_KP>
__global__ void __launch_bounds__(BF_SMALL_THREADS) band_fluxes_small_kernel(
    int n_mod,
    int nspec,
    int nbands,
    const T_KP* __restrict__ d_bands_t,
    const int*  __restrict__ d_j_lo,
    const int*  __restrict__ d_j_hi,
    const int*  __restrict__ d_do_band,
    const T_KP* __restrict__ d_wt,
    const T_KP* __restrict__ d_fobs_t,
    double*     __restrict__ d_fband
) {
    const int m = blockIdx.x;
    const int b = blockIdx.y;
    const int t = threadIdx.x;
    __shared__ T_KP partial[BF_SMALL_THREADS];

    // (the whole workgroup takes this branch together)
    if (d_do_band[b] == 0) {
        if (t == 0) d_fband[(size_t)m * nbands + b] = 0.0;
        return;
    }

    const int j_start = d_j_lo[b] - 1;   // 0-based
    const int j_end   = d_j_hi[b] - 1;

    T_KP s = 0;
    T_KP c = 0;
    for (int j = j_start + t; j <= j_end; j += BF_SMALL_THREADS) {
        const T_KP w  = d_wt[j];
        const T_KP b0 = (j < nspec)     ? d_bands_t[(size_t)b + (size_t)j * nbands] : (T_KP)0;
        const T_KP f0 = (j < nspec)     ? d_fobs_t[(size_t)m + (size_t)j * n_mod]   : (T_KP)0;
        const T_KP b1 = (j + 1 < nspec) ? d_bands_t[(size_t)b + (size_t)(j + 1) * nbands] : (T_KP)0;
        const T_KP f1 = (j + 1 < nspec) ? d_fobs_t[(size_t)m + (size_t)(j + 1) * n_mod]   : (T_KP)0;
        kahan_add(w * (f0 * b0 + f1 * b1), s, c);
    }
    partial[t] = s - c;
    __syncthreads();
    for (int stride = BF_SMALL_THREADS / 2; stride > 0; stride >>= 1) {
        if (t < stride) partial[t] += partial[t + stride];
        __syncthreads();
    }
    if (t == 0) d_fband[(size_t)m * nbands + b] = (double)partial[0];
}


// ---------------------------------------------------------------------------
// Band fluxes of many models as a product (FP32): fband(m, b) = sum_p A(b, p) fobs_t(m, p)
//
// The trapezoid sum of band_fluxes_kernel, sum_{j = j_lo}^{j_hi} w_j (f_j B_j + f_{j+1} B_{j+1}), is linear in
// the pixels: pixel p carries A(b, p) = B_p (w_p [j_lo <= p <= j_hi] + w_{p-1} [j_lo <= p - 1 <= j_hi]).
// band_matrix_kernel forms A (nbands x nspec, bands fastest) once per call, band_gemm_kernel is the
// shared-memory tiled product (as template_columns_kernel: blocks of 4 pixels with plain multiply-adds,
// the blocks added with compensation; all terms are >= 0). Each observed pixel is read once per tile of
// 64 models instead of once per band by every thread of the band, and the whole call is a product of
// ~1e10 multiply-adds instead of the (bands x width) serial sums.
// ---------------------------------------------------------------------------
__global__ void band_matrix_kernel(int nspec, int nbands, const float* __restrict__ bands_t,
                                   const int* __restrict__ j_lo, const int* __restrict__ j_hi,
                                   const int* __restrict__ do_band, const float* __restrict__ wt,
                                   float* __restrict__ A) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;
    if (p >= nspec || b >= nbands) return;
    float v = 0.f;
    if (do_band[b] != 0) {
        int js = j_lo[b] - 1, je = j_hi[b] - 1;
        float w = 0.f;
        if (p >= js && p <= je) w += wt[p];
        if (p >= 1 && p - 1 >= js && p - 1 <= je) w += wt[p - 1];
        v = w * bands_t[(size_t)p * nbands + b];
    }
    A[(size_t)p * nbands + b] = v;
}

template <int RM, int RN, int KC, int KB>
__global__ void __launch_bounds__(256, 2) band_gemm_kernel(
    int nbands, int n_mod, int nspec,
    const float* __restrict__ A, const float* __restrict__ F, double* __restrict__ out
) {
    constexpr int BMT = 16 * RM, BN = 16 * RN;
    __shared__ float As[KC][BMT];
    __shared__ float Bs[KC][BN];
    const int tid = threadIdx.x;
    const int tx = tid & 15, ty = tid >> 4;
    const int i0 = blockIdx.y * BMT, m0 = blockIdx.x * BN;
    float acc[RM][RN], cmp[RM][RN], blk[RM][RN];
#pragma unroll
    for (int r = 0; r < RM; ++r)
#pragma unroll
        for (int c = 0; c < RN; ++c) { acc[r][c] = 0.f; cmp[r][c] = 0.f; }
    float pa[KC * BMT / 256], pb[KC * BN / 256];
    auto gload = [&](int e0) {
#pragma unroll
        for (int q = 0; q < KC * BMT / 256; ++q) {
            int idx = tid + q * 256, i = idx % BMT, e = idx / BMT;
            pa[q] = (i0 + i < nbands && e0 + e < nspec) ? A[(size_t)(e0 + e) * nbands + i0 + i] : 0.f;
        }
#pragma unroll
        for (int q = 0; q < KC * BN / 256; ++q) {
            int idx = tid + q * 256, i = idx % BN, e = idx / BN;
            pb[q] = (m0 + i < n_mod && e0 + e < nspec) ? F[(size_t)(e0 + e) * n_mod + m0 + i] : 0.f;
        }
    };
    gload(0);
    for (int e0 = 0; e0 < nspec; e0 += KC) {
        __syncthreads();
#pragma unroll
        for (int q = 0; q < KC * BMT / 256; ++q) {
            int idx = tid + q * 256; As[idx / BMT][idx % BMT] = pa[q];
        }
#pragma unroll
        for (int q = 0; q < KC * BN / 256; ++q) {
            int idx = tid + q * 256; Bs[idx / BN][idx % BN] = pb[q];
        }
        __syncthreads();
        if (e0 + KC < nspec) gload(e0 + KC);
#pragma unroll
        for (int g = 0; g < KC / KB; ++g) {
            if (e0 + g * KB >= nspec) break;
#pragma unroll
            for (int r = 0; r < RM; ++r)
#pragma unroll
                for (int c = 0; c < RN; ++c) blk[r][c] = 0.f;
#pragma unroll
            for (int u = 0; u < KB; ++u) {
                const int e = g * KB + u;
                float av[RM], bv[RN];
#pragma unroll
                for (int r = 0; r < RM; ++r) av[r] = As[e][tx + 16 * r];
#pragma unroll
                for (int c = 0; c < RN; ++c) bv[c] = Bs[e][ty + 16 * c];
#pragma unroll
                for (int r = 0; r < RM; ++r)
#pragma unroll
                    for (int c = 0; c < RN; ++c) blk[r][c] = fmaf(av[r], bv[c], blk[r][c]);
            }
#pragma unroll
            for (int r = 0; r < RM; ++r)
#pragma unroll
                for (int c = 0; c < RN; ++c) {
                    float term = blk[r][c] - cmp[r][c];
                    float tsum = acc[r][c] + term;
                    cmp[r][c] = (tsum - acc[r][c]) - term;
                    acc[r][c] = tsum;
                }
        }
    }
#pragma unroll
    for (int c = 0; c < RN; ++c) {
        int m = m0 + ty + 16 * c;
        if (m < n_mod) {
#pragma unroll
            for (int r = 0; r < RM; ++r) {
                int b = i0 + tx + 16 * r;
                if (b < nbands) out[(size_t)m * nbands + b] = (double)(acc[r][c] - cmp[r][c]);
            }
        }
    }
}

static float* g_band_A = nullptr;
static size_t g_band_A_n = 0;


// ---------------------------------------------------------------------------
// Dust attenuation of the batch (attenuate_batch, FP32 kernels, dust types with the tabulated curves)
//
// The Fortran kernel read ~20 FP64 parameters of the model and formed ~35 FP64 operations (the DL07
// interpolation) per (pixel, model): FP64 is 1/64 of the FP32 rate on the R9700 and this kernel was 13%
// of the fixed-IMF forward call. Here the per-model parameters are FP32 (mp, mi: prepared once per model),
// the arguments of the power laws, e ln(lambda / lambda_V), are formed as unevaluated sums of two FP32
// numbers (hi + lo, the fma gives the product's rounding error) so that exp(p) (1 + r) is good to the
// rounding of expf (rule 3 of doc/MIXED_PRECISION.md: no e L in FP32), and the DL07 template is the
// convex combination of the 4 x 2 table values with FP32 weights (the shape is renormalised afterwards).
// mp: 0 e_hi, 1 e_lo, 2 s1, 3 s2, 4 s3, 5 dust2, 6 dust1, 7 e1_hi, 8 e1_lo, 9 dust3, 10 frac_obrun,
//     11 frac_nodust, 12-15 DL07 corner weights, 16 gamma
// mi: 0 flags (1 has_ed, 2 birth, 4 old, 8 nodust, 16 emis, 32 agn), 1 iq (0-based), 2 iu (0-based)
// ---------------------------------------------------------------------------
__device__ inline float expdf(float ehi, float elo, float lhi, float llo) {
    #pragma clang fp contract(off)
    float p = __fmul_rn(ehi, lhi);
    float r = fmaf(ehi, lhi, -p) + (ehi * llo + elo * lhi);
    float ex = expf(p);
    return fmaf(ex, r, ex);
}

__device__ inline float one_minus_trans_f(float x, float t) {
    if (t == 1.0f) return x;
    if (fabsf(x) < 1.0f) return (1.0f - t) * x / (-logf(t));
    return 1.0f - t;
}

__global__ void split_ltab_kernel(int n, const double* __restrict__ l, float2* __restrict__ out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float h = (float)l[i];
    out[i] = make_float2(h, (float)(l[i] - (double)h));
}

__global__ void __launch_bounds__(256) attenuate_kernel(
    int nspec, int n_mod, int nq,
    const float* __restrict__ mp, const int* __restrict__ mi,
    const float* __restrict__ ttab, const float2* __restrict__ ltf, const float2* __restrict__ dustem,
    float* __restrict__ yo, float* __restrict__ total, float* __restrict__ absorbed, float* __restrict__ tdiff
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y;
    if (i >= nspec) return;
    const float* p = mp + (size_t)m * 24;
    const int fl = mi[m * 4];
    const float young = yo[(size_t)m * nspec + i];
    const float old = yo[(size_t)(n_mod + m) * nspec + i];
    const float2 L = ltf[i];
    const float frac_obrun = p[10], frac_nodust = p[11];

    float curve = ttab[i] + p[2] * ttab[(size_t)nspec + i] + p[3] * ttab[(size_t)2 * nspec + i] + p[4] * ttab[(size_t)3 * nspec + i];
    if (fl & 1) curve *= expdf(p[0], p[1], L.x, L.y);
    const float tau_d = p[5] * curve;
    const float trans_d = expf(-tau_d);
    float tau_b = 0.f, trans_birth = 1.f, tau_o = 0.f, trans_old = 1.f;
    if (fl & 2) { tau_b = p[6] * expdf(p[7], p[8], L.x, L.y); trans_birth = expf(-tau_b); }
    if (fl & 4) { tau_o = p[9] * curve; trans_old = expf(-tau_o); }
    if (tdiff != nullptr && (fl & 32)) tdiff[(size_t)m * nspec + i] = trans_d;

    const float spec_sum = (young * trans_birth * (1.0f - frac_obrun) + young * frac_obrun) + old * trans_old;
    total[(size_t)m * nspec + i] = (fl & 8) ? spec_sum * (trans_d * (1.0f - frac_nodust) + frac_nodust) : spec_sum;
    if (fl & 16) {
        const float one_m_tb = one_minus_trans_f(tau_b, trans_birth);
        const float one_m_to = one_minus_trans_f(tau_o, trans_old);
        const float one_m_td = (fl & 8) ? (1.0f - frac_nodust) * one_minus_trans_f(tau_d, trans_d) : 0.f;
        const float a_young = trans_birth * (1.0f - frac_obrun) + frac_obrun;
        absorbed[(size_t)m * nspec + i] = young * ((1.0f - frac_obrun) * one_m_tb + a_young * one_m_td) +
                                          old * (one_m_to + trans_old * one_m_td);
        yo[(size_t)m * nspec + i] = trans_d;
        const int iq = mi[m * 4 + 1], iu = mi[m * 4 + 2];
        const size_t plane = (size_t)nspec * nq;
        const float2* d0 = dustem + (size_t)i + (size_t)iq * nspec + (size_t)(2 * iu) * plane;   // (i, iq, 2 iu - 1), 0-based
        const float2* d1 = d0 + 2 * plane;                                                          // (i, iq, 2 (iu + 1) - 1)
        // 4 corners x (lo, hi) of the table, each a hi + lo pair: the convex sums in FP32 from exact-ish inputs
        float lo = 0.f, hi = 0.f, lo2 = 0.f, hi2 = 0.f;
        {
            const float2 a = d0[0], b = d0[nspec], c = d1[nspec], d = d1[0];
            lo = fmaf(p[12], a.x, fmaf(p[13], b.x, fmaf(p[14], c.x, p[15] * d.x)));
            lo2 = fmaf(p[12], a.y, fmaf(p[13], b.y, fmaf(p[14], c.y, p[15] * d.y)));
        }
        {
            const float2 a = d0[plane], b = d0[nspec + plane], c = d1[nspec + plane], d = d1[plane];
            hi = fmaf(p[12], a.x, fmaf(p[13], b.x, fmaf(p[14], c.x, p[15] * d.x)));
            hi2 = fmaf(p[12], a.y, fmaf(p[13], b.y, fmaf(p[14], c.y, p[15] * d.y)));
        }
        lo += lo2; hi += hi2;
        yo[(size_t)(n_mod + m) * nspec + i] = fmaxf((1.0f - p[16]) * lo + p[16] * hi, 1.17549435e-38f);
    } else {
        absorbed[(size_t)m * nspec + i] = 0.f;
    }
}

__global__ void split_dustem_kernel(size_t n, const double* __restrict__ d, float2* __restrict__ out) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float h = (float)d[i];
    out[i] = make_float2(h, (float)(d[i] - (double)h));
}

static float2* g_ltf = nullptr;
static size_t g_ltf_n = 0;
static float2* g_dust = nullptr;
static size_t g_dust_n = 0;


// ---------------------------------------------------------------------------
// Observed-frame spectra with a redshift per model (observed_frame_z, transposed layout, FP32 kernels)
//
// fobs(m, j) = max(f_rest(lambda_j / (1 + z_m)), 0) wl(j): the rest-frame spectrum linearly interpolated at
// the pixel's rest wavelength. The Fortran kernel did the wavelength arithmetic in FP64 (a division for the
// target, one for the interpolation fraction, FP64 comparisons) per (pixel, model). Here the fraction
// t = (lambda_j - lambda_il (1 + z)) / ((1 + z) (lambda_il+1 - lambda_il)) is formed from hi + lo FP32 pairs
// (the numerator is a difference of nearly equal wavelengths: an FMA gives the product's error term, so it is
// good to ~1e-7 of the pixel spacing, not 1e-4 as plain FP32 would be), and the interval is found from the
// sign of that numerator (consecutive pixels walk the interval index; each thread's first pixel bisects), as
// the hinted search of the Fortran kernel.
// ---------------------------------------------------------------------------
__device__ inline float obs_num(float2 lj, float2 li, float ahi, float alo) {
    #pragma clang fp contract(off)
    float P = __fmul_rn(li.x, ahi);   // (not contractible: Pe below is its rounding error)
    float Pe = fmaf(li.x, ahi, -P);
    float S = Pe + (li.x * alo + li.y * ahi);
    return (lj.x - P) + (lj.y - S);
}

__global__ void __launch_bounds__(128) observed_frame_kernel(
    int n_col, int nspec, int npad, int ch, int ldf, double zfloor,
    const double* __restrict__ zc, const float2* __restrict__ lam, const float* __restrict__ wl,
    const float* __restrict__ total, float* __restrict__ fobs
) {
    const int m = blockIdx.x * blockDim.x + threadIdx.x;
    const int jc = blockIdx.y;
    if (m >= n_col) return;
    const double z = zc[m];
    const bool zon = fabs(z) > zfloor;
    const double a = 1.0 + z;
    const float ahi = (float)a, alo = (float)(a - (double)ahi);
    const int j0 = jc * ch, j1 = min(j0 + ch, npad);
    const float* tm = total + (size_t)m * nspec;
    int il = -1;
    for (int j = j0; j < j1; ++j) {
        float f = 0.f;
        if (j < nspec) {
            float v;
            if (zon) {
                const float2 lj = lam[j];
                if (il < 0) {   // bisect: the last interval whose lower wavelength is <= the target
                    int lo = 0, hi = nspec - 2;
                    while (lo < hi) {
                        int mid = (lo + hi + 1) >> 1;
                        if (obs_num(lj, lam[mid], ahi, alo) >= 0.f) lo = mid; else hi = mid - 1;
                    }
                    il = lo;
                }
                // num = lambda_j - a lambda_il >= 0 and numu = lambda_j - a lambda_il+1 < 0 inside the interval; the
                // two weights t = num / (num - numu) and u = -numu / (num - numu) are each accurate to ~1e-7 relative
                // and sum to one (the denominator is a sum of two non-negative numbers, no cancellation): where
                // the spectrum drops by orders of magnitude between the nodes (Lyman break) f0 + t (f1 - f0) would
                // carry the error of t times f0 into a value ~f1, and a denominator a * dl of its own would scale the
                // whole interpolation by its rounding
                float num = obs_num(lj, lam[il], ahi, alo);
                float numu = obs_num(lj, lam[il + 1], ahi, alo);
                while (il > 0 && num < 0.f) {
                    --il;
                    numu = num;
                    num = obs_num(lj, lam[il], ahi, alo);
                }
                while (il < nspec - 2 && numu >= 0.f) {
                    ++il;
                    num = numu;
                    numu = obs_num(lj, lam[il + 1], ahi, alo);
                }
                if ((il == 0 && num < 0.f) || (il == nspec - 2 && numu >= 0.f)) {
                    // Outside the rest-frame grid the Fortran kernel extrapolates the end interval linearly: the
                    // fraction is far from [0, 1] there and f0 + t (f1 - f0) amplifies its error, so these pixels
                    // (a few per model, at the blue end) take the FP64 form of the original
                    const double l0 = (double)lam[il].x + (double)lam[il].y, l1 = (double)lam[il + 1].x + (double)lam[il + 1].y;
                    const double target = ((double)lj.x + (double)lj.y) / a;
                    const double td = (target - l0) / (l1 - l0);
                    v = (float)((double)tm[il] + td * ((double)tm[il + 1] - (double)tm[il]));
                } else {
                    // (the weights first: f * wavelength offset would underflow where the spectrum is ~1e-37)
                    const float den = num - numu;
                    v = tm[il] * (-numu / den) + tm[il + 1] * (num / den);
                }
            } else {
                v = tm[j];
            }
            if (v != v) v = 0.f;
            f = fmaxf(v, 0.f) * wl[j];
        }
        fobs[(size_t)j * ldf + m] = f;
    }
}



// ---------------------------------------------------------------------------
// Reverse pass of the observed-frame resampling (backward_observed_frame, FP32 kernels)
//
// The Fortran kernels located every (pixel, model)'s rest-frame interval with an FP64 bisection and formed the
// fraction in FP64 (~140 FP64 operations per element: 18 ms of a 55 ms reverse pass at 1000 models, FP64 at 1/64
// of the FP32 rate). Here the interval and the two weights come from the forward kernel's hi + lo numerators
// (obs_num), so the reverse pass uses the weights the forward pass applied:
//   K1  gw(j, m) = [v > 0] wl(j) sum_bands fb_bar(b, m) B(b, j)   (v: the resampled value, as the forward) and the
//       block partial sums of d L / d z = sum_j gw (S(il+1) - S(il)) dt/dz in FP64;
//   K2  the blocks of K1's partial sums, in order, per model;
//   K3  the rest-frame cotangent, a gather per (rest pixel, model): the observed pixels that reach it are one run
//       (the interval index does not decrease with j) found by one bisection, added in the order of j, as the
//       scatter of the portable code; models at z = 0 pass the cotangent through.
// ---------------------------------------------------------------------------
#define OBS_ZON(z, zfloor) (fabs(z) > (zfloor))

__device__ inline void obs_locate(const float2* __restrict__ lam, int nspec, float2 lj, float ahi, float alo, double a,
                                  int il_start, int& il, float& num, float& numu) {
    if (il_start < 0) {     // bisect: the last interval whose lower wavelength is <= the target
        int lo = 0, hi = nspec - 2;
        while (lo < hi) {
            int mid = (lo + hi + 1) >> 1;
            if (obs_num(lj, lam[mid], ahi, alo) >= 0.f) lo = mid; else hi = mid - 1;
        }
        il = lo;
    } else {
        il = il_start;
    }
    num = obs_num(lj, lam[il], ahi, alo);
    numu = obs_num(lj, lam[il + 1], ahi, alo);
    while (il > 0 && num < 0.f) {
        --il;
        numu = num;
        num = obs_num(lj, lam[il], ahi, alo);
    }
    while (il < nspec - 2 && numu >= 0.f) {
        ++il;
        num = numu;
        numu = obs_num(lj, lam[il + 1], ahi, alo);
    }
    // A pixel exactly on a node (the numerator is exactly zero: wavelengths that are in an exact ratio, as for round
    // redshifts) lies in two intervals; the forward value is the same in both, the derivative is not. The FP64 code
    // took the interval its rounded division fell into: do the same, so that the reverse pass agrees with it.
    if (num == 0.f && il > 0) {
        const double target = ((double)lj.x + (double)lj.y) / a;
        if (target < (double)lam[il].x + (double)lam[il].y) {
            numu = num;
            --il;
            num = obs_num(lj, lam[il], ahi, alo);
        }
    }
}

// fb_bar(band, model) as FP32, transposed to (model, band): the models of a warp read consecutive words
__global__ void obs_fb_to_float_kernel(int nbands, int n_mod, const double* __restrict__ in, float* __restrict__ out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nbands * n_mod) {
        int b = i % nbands, m = i / nbands;
        out[m + (size_t)b * n_mod] = (float)in[i];
    }
}

__global__ void __launch_bounds__(128) obs_bwd_gw_kernel(
    int n_mod, int nspec, int nbands, int ch, int dzon, double zfloor,
    const double* __restrict__ zc, const float2* __restrict__ lam, const float* __restrict__ wl,
    const int* __restrict__ bp, const int* __restrict__ bi, const float* __restrict__ bk,
    const float* __restrict__ fbf, const float* __restrict__ total, float* __restrict__ gw, double* __restrict__ dzp
) {
    // thread = (model, chunk of ch consecutive pixels): the interval index walks along the pixels, the first bisects
    const int m = blockIdx.x * blockDim.x + threadIdx.x;
    const int jc = blockIdx.y;
    if (m >= n_mod) return;
    const double z = zc[m];
    const bool zon = OBS_ZON(z, zfloor);
    const double a = 1.0 + z;
    const float ahi = (float)a, alo = (float)(a - (double)ahi);
    const float* tm = total + (size_t)m * nspec;
    double dsum = 0.0;
    int il = -1;
    const int j0 = jc * ch, j1 = min(j0 + ch, nspec);
    for (int j = j0; j < j1; ++j) {
        float gk = 0.f;
        for (int k = bp[j] - 1; k < bp[j + 1] - 1; ++k) gk += fbf[m + (size_t)(bi[k] - 1) * n_mod] * bk[k];
        float v;
        if (zon) {
            const float2 lj = lam[j];
            float num, numu;
            int ils = il;
            obs_locate(lam, nspec, lj, ahi, alo, a, ils, il, num, numu);
            if ((il == 0 && num < 0.f) || (il == nspec - 2 && numu >= 0.f)) {
                const double l0 = (double)lam[il].x + (double)lam[il].y, l1 = (double)lam[il + 1].x + (double)lam[il + 1].y;
                const double target = ((double)lj.x + (double)lj.y) / a;
                const double td = (target - l0) / (l1 - l0);
                v = (float)((double)tm[il] + td * ((double)tm[il + 1] - (double)tm[il]));
            } else {
                v = tm[il] * (-numu / (num - numu)) + tm[il + 1] * (num / (num - numu));
            }
            if (!(v > 0.f)) gk = 0.f;
            const float g = gk * wl[j];
            gw[j + (size_t)m * nspec] = g;
            if (dzon && g != 0.f) {
                const double l0 = (double)lam[il].x + (double)lam[il].y, l1 = (double)lam[il + 1].x + (double)lam[il + 1].y;
                dsum += (double)g * ((double)tm[il + 1] - (double)tm[il]) * (-((double)lj.x + (double)lj.y) / (a * a)) / (l1 - l0);
            }
        } else {
            v = tm[j];
            if (!(v > 0.f)) gk = 0.f;
            gw[j + (size_t)m * nspec] = gk * wl[j];
        }
    }
    if (dzon) dzp[jc + (size_t)gridDim.y * m] = dsum;
}

__global__ void obs_bwd_dz_kernel(int nblk, int n_mod, const double* __restrict__ dzp, double* __restrict__ dzo) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= n_mod) return;
    double s = 0.0;
    for (int b = 0; b < nblk; ++b) s += dzp[b + (size_t)nblk * m];
    dzo[m] = s;
}

__global__ void __launch_bounds__(128) obs_bwd_rest_kernel(
    int n_mod, int nspec, double zfloor, const double* __restrict__ zc, const float2* __restrict__ lam,
    int have_s, const float* __restrict__ sbar, int ld_sbar, float* __restrict__ gw
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;      // 0-based rest pixel
    const int m = blockIdx.y;
    if (i >= nspec) return;
    float vk = have_s ? sbar[i + (size_t)m * ld_sbar] : 0.f;
    const float* g = gw + (size_t)m * nspec;
    const double z = zc[m];
    if (!OBS_ZON(z, zfloor)) {
        gw[i + ((size_t)n_mod + m) * nspec] = vk + g[i];
        return;
    }
    const double a = 1.0 + z;
    const float ahi = (float)a, alo = (float)(a - (double)ahi);
    // the first observed pixel whose interval index is >= i - 1: its rest wavelength reaches lambda_(i-1)
    int jlo = 0;
    if (i >= 2) {
        int lo = 0, hi = nspec;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (obs_num(lam[mid], lam[i - 1], ahi, alo) >= 0.f) hi = mid; else lo = mid + 1;
        }
        jlo = lo;
    }
    int il = max(i - 1, 0);
    for (int j = jlo; j < nspec; ++j) {
        float num, numu;
        obs_locate(lam, nspec, lam[j], ahi, alo, a, il, il, num, numu);
        if (il > i) break;
        if (il == i - 1) {
            if (num != 0.f) vk += (num / (num - numu)) * g[j];
        } else if (il == i) {
            vk += (-numu / (num - numu)) * g[j];
        }
    }
    gw[i + ((size_t)n_mod + m) * nspec] = vk;
}


// ---------------------------------------------------------------------------
// Self-test of the error-free transforms (doc/MIXED_PRECISION.md, rule 9)
//
// The kernels' two-sums and exact products are only exact if the compiler does not fuse a rounded product into the
// sum that uses it (it would count the product's rounding error twice). hipcc/clang is told so by
// `#pragma clang fp contract(off)`, nvcc by __fmul_rn (a product it never merges into an fma; it ignores the pragma).
// This runs the observed-frame numerator and the fold's product-and-sum on random inputs chosen so that the answer
// is a small difference of large numbers, and compares them with FP64. Returns the number of failures (0: the guard
// holds), -1 if it could not run.
// ---------------------------------------------------------------------------
__device__ inline void eft_sum_product(float x, float w, float s, float& tt, float& c) {
#pragma clang fp contract(off)
    float p1 = __fmul_rn(x, w);
    float e1 = fmaf(x, w, -p1);
    tt = s + p1;
    float bp = tt - s;
    c = (s - (tt - bp)) + (p1 - bp) + e1;
}

__global__ void eft_selftest_kernel(int n, unsigned long long seed, int* fails) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned long long st = seed + 0x9E3779B97F4A7C15ull * (unsigned long long)(i + 1);
    auto rnd = [&]() {
        st ^= st << 13; st ^= st >> 7; st ^= st << 17;
        return (double)(st >> 11) * (1.0 / 9007199254740992.0);
    };
    int bad = 0;
    {   // obs_num: lambda_j - a lambda_i with lambda_j within a few Angstroms of a lambda_i
        const double li = 3000.0 + 6000.0 * rnd(), a = 1.05 + 2.5 * rnd();
        const double lj = a * li + (rnd() - 0.5) * 10.0;
        const float lih = (float)li, ljh = (float)lj;
        const float2 L = make_float2(lih, (float)(li - (double)lih)), J = make_float2(ljh, (float)(lj - (double)ljh));
        const float ahi = (float)a, alo = (float)(a - (double)ahi);
        const float num = obs_num(J, L, ahi, alo);
        const double ref = ((double)J.x + (double)J.y) - ((double)ahi + (double)alo) * ((double)L.x + (double)L.y);
        if (fabs((double)num - ref) > 4e-6 + 2e-7 * fabs(ref)) ++bad;
    }
    {   // the fold's and the weights' product added to a running sum
        const float x = (float)(0.01 + 3.0 * rnd()), w = (float)(2.0 * rnd()), s = (float)(5.0 * rnd());
        float tt, c;
        eft_sum_product(x, w, s, tt, c);
        const double exact = (double)s + (double)x * (double)w, got = (double)tt + (double)c;
        if (fabs(got - exact) > 1e-13 * fabs(exact)) ++bad;
    }
    if (bad) atomicAdd(fails, bad);
}

// ---------------------------------------------------------------------------
// Age sums with per-model metallicity (age_sums_z, FP32 kernels), as a product per metallicity pair
//
// Model m (metallicity interval zl, zl + 1, weight d) has young / old age sums
//   y(i) = sum_{k <= it} w_y(k) [(1 - d) B(i, k, zl) + d B(i, k, zl + 1)],   o(i) = same over k > it with w_o,
// B(i, k, z) the basis spectra. The two slabs of a pair are adjacent columns of the basis seen as an
// (nspec x nt nz) matrix, so for the models of one pair this is A times a coefficient matrix ((1 - d) w,
// d w per column) formed while the tile is staged. A block computes 16 RM pixels x 16 RN models of the young sums
// (blockIdx.z = 0: ages k <= the tile's largest it, in both slabs) or of the old sums (z = 1: ages above the
// tile's smallest it); the zero coefficients outside are not multiplied. 8 x 4 per thread, templates 16 at
// a time in shared memory; arithmetic as the template products: blocks of 4 terms with plain multiply-adds
// (all terms >= 0), the blocks added with compensation (the Fortran kernel summed ~100 ages plainly).
// tile t: models zord[first[t] .. first[t] + cnt[t] - 1] (1-based model numbers, cnt <= 16 RN), all of metallicity pair zl[t]
// ---------------------------------------------------------------------------
template <int RM, int RN, int KC, int KB>
__global__ void __launch_bounds__(256, 2) age_sums_z_kernel(
    int nspec, int nt, int n_mod,
    const int* __restrict__ zord, const int* __restrict__ tfirst, const int* __restrict__ tcnt, const int* __restrict__ tzl,
    const float* __restrict__ basis, const float* __restrict__ w2, const int* __restrict__ it_tesc,
    const double* __restrict__ dz, float* __restrict__ yo
) {
    constexpr int BM = 16 * RM, BN = 16 * RN;   // pixels, models per block
    __shared__ float As[KC][BM];
    __shared__ float Bs[KC][BN];
    __shared__ int   s_wb[BN];     // offset of the column's weights in w2
    __shared__ int   s_it[BN];
    __shared__ float s_d[BN];
    __shared__ int   s_out[BN];    // model column of yo (-1: no model)
    __shared__ int   s_klo, s_khi;
    const int tid = threadIdx.x;
    const int tx = tid & 15, ty = tid >> 4;
    const int i0 = blockIdx.x * BM;
    const int t = blockIdx.y;
    const int old = blockIdx.z;
    const int first = tfirst[t], cnt = tcnt[t], zl = tzl[t];
    if (tid == 0) { s_klo = nt; s_khi = 0; }
    __syncthreads();
    if (tid < BN) {
        if (tid < cnt) {
            const int m = zord[first + tid] - 1;
            s_wb[tid] = (old ? n_mod + m : m) * nt;
            s_it[tid] = it_tesc[m];
            s_d[tid] = (float)dz[m];
            s_out[tid] = old ? n_mod + m : m;
            if (old) atomicMin(&s_klo, it_tesc[m]); else atomicMax(&s_khi, it_tesc[m]);
        } else {
            s_wb[tid] = 0; s_it[tid] = 0; s_d[tid] = 0.f; s_out[tid] = -1;
        }
    }
    __syncthreads();
    // 0-based ages kk of the range: young kk < it_max, old kk >= it_min (it is 1-based: ages 1 .. it are young)
    const int klo = old ? s_klo : 0;
    const int khi = old ? nt : s_khi;
    const int len = max(khi - klo, 0);
    const int K = 2 * len;
    const float* Ab = basis + (size_t)(zl - 1) * nt * nspec;

    float acc[RM][RN], cmp[RM][RN], blk[RM][RN];
#pragma unroll
    for (int r = 0; r < RM; ++r)
#pragma unroll
        for (int c = 0; c < RN; ++c) { acc[r][c] = 0.f; cmp[r][c] = 0.f; }

    float pa[KC * BM / 256], pb[KC * BN / 256];
    auto gload = [&](int e0) {
#pragma unroll
        for (int q = 0; q < KC * BM / 256; ++q) {
            int idx = tid + q * 256, i = idx % BM, e = e0 + idx / BM;
            float v = 0.f;
            if (i0 + i < nspec && e < K) {
                int half = e >= len;
                int kk = klo + e - (half ? len : 0);
                v = Ab[(size_t)(half * nt + kk) * nspec + i0 + i];
            }
            pa[q] = v;
        }
#pragma unroll
        for (int q = 0; q < KC * BN / 256; ++q) {
            int idx = tid + q * 256, el = idx % KC, col = idx / KC;
            int e = e0 + el;
            float v = 0.f;
            if (e < K && s_out[col] >= 0) {
                int half = e >= len;
                int kk = klo + e - (half ? len : 0);
                int k1 = kk + 1;
                bool use = old ? (k1 > s_it[col]) : (k1 <= s_it[col]);
                if (use) v = w2[s_wb[col] + kk] * (half ? s_d[col] : 1.0f - s_d[col]);
            }
            pb[q] = v;
        }
    };
    gload(0);
    for (int e0 = 0; e0 < K; e0 += KC) {
        __syncthreads();
#pragma unroll
        for (int q = 0; q < KC * BM / 256; ++q) {
            int idx = tid + q * 256; As[idx / BM][idx % BM] = pa[q];
        }
#pragma unroll
        for (int q = 0; q < KC * BN / 256; ++q) {
            int idx = tid + q * 256; Bs[idx % KC][idx / KC] = pb[q];
        }
        __syncthreads();
        if (e0 + KC < K) gload(e0 + KC);
#pragma unroll
        for (int g = 0; g < KC / KB; ++g) {
            if (e0 + g * KB >= K) break;
#pragma unroll
            for (int r = 0; r < RM; ++r)
#pragma unroll
                for (int c = 0; c < RN; ++c) blk[r][c] = 0.f;
#pragma unroll
            for (int u = 0; u < KB; ++u) {
                const int e = g * KB + u;
                float av[RM], bv[RN];
#pragma unroll
                for (int r = 0; r < RM; ++r) av[r] = As[e][tx + 16 * r];
#pragma unroll
                for (int c = 0; c < RN; ++c) bv[c] = Bs[e][ty + 16 * c];
#pragma unroll
                for (int r = 0; r < RM; ++r)
#pragma unroll
                    for (int c = 0; c < RN; ++c) blk[r][c] = fmaf(av[r], bv[c], blk[r][c]);
            }
#pragma unroll
            for (int r = 0; r < RM; ++r)
#pragma unroll
                for (int c = 0; c < RN; ++c) {
                    float term = blk[r][c] - cmp[r][c];
                    float tsum = acc[r][c] + term;
                    cmp[r][c] = (tsum - acc[r][c]) - term;
                    acc[r][c] = tsum;
                }
        }
    }
#pragma unroll
    for (int c = 0; c < RN; ++c) {
        int col = ty + 16 * c;
        if (s_out[col] >= 0) {
#pragma unroll
            for (int r = 0; r < RM; ++r) {
                int i = i0 + tx + 16 * r;
                if (i < nspec) yo[(size_t)s_out[col] * nspec + i] = acc[r][c] - cmp[r][c];
            }
        }
    }
}

extern "C" {

int opensps_hip_selftest_eft(int n) {
    int* d = nullptr;
    if (n <= 0) return 0;
    if (hipMalloc((void**)&d, sizeof(int)) != hipSuccess) return -1;
    int zero = 0, bad = 0;
    if (hipMemcpy(d, &zero, sizeof(int), hipMemcpyHostToDevice) != hipSuccess) { (void)hipFree(d); return -1; }
    eft_selftest_kernel<<<(n + 255) / 256, 256>>>(n, 20261116ull, d);
    (void)hipDeviceSynchronize();
    if (hipMemcpy(&bad, d, sizeof(int), hipMemcpyDeviceToHost) != hipSuccess) bad = -1;
    (void)hipFree(d);
    return bad;
}

// Returns 1 if the tile shape is not supported (the caller then takes its portable kernel). Tiles hold up to
// 16 rn models (the caller builds them)
int opensps_hip_age_sums_z(int nspec, int nt, int n_mod, int ntile, int rm, int rn, int kc, int kb, const int* d_zord,
                           const int* d_tfirst, const int* d_tcnt, const int* d_tzl, const float* d_basis,
                           const float* d_w2, const int* d_it, const double* d_dz, float* d_yo, void* stream) {
    if (nspec <= 0 || ntile <= 0) return 0;
    hipStream_t s = (hipStream_t)stream;
    const bool ok = gemm_dispatch(rm, rn, kc, kb, [&](auto cfg) {
        constexpr int RM = decltype(cfg)::RM, RN = decltype(cfg)::RN, KC = decltype(cfg)::KC, KB = decltype(cfg)::KB;
        age_sums_z_kernel<RM, RN, KC, KB><<<dim3((nspec + 16 * RM - 1) / (16 * RM), ntile, 2), 256, 0, s>>>(
            nspec, nt, n_mod, d_zord, d_tfirst, d_tcnt, d_tzl, d_basis, d_w2, d_it, d_dz, d_yo);
    });
    if (!ok) return 1;
    (void)hipStreamSynchronize(s);
    return 0;
}

void opensps_hip_attenuate(int nspec, int n_mod, int nq, const float* d_mp, const int* d_mi, const float* d_ttab,
                           const double* d_ltab, const double* d_dustem, size_t n_dustem, int any_emis,
                           float* d_yo, float* d_total, float* d_absorbed, float* d_tdiff, int threads, void* stream) {
    if (nspec <= 0 || n_mod <= 0) return;
    if (threads < 32) threads = 32;
    if (threads > 256) threads = 256;   // (the kernel's launch bounds)
    hipStream_t s = (hipStream_t)stream;
    if ((size_t)nspec > g_ltf_n) {
        if (g_ltf) (void)hipFree(g_ltf);
        if (hipMalloc((void**)&g_ltf, (size_t)nspec * sizeof(float2)) != hipSuccess) { g_ltf = nullptr; g_ltf_n = 0; return; }
        g_ltf_n = nspec;
    }
    split_ltab_kernel<<<(nspec + threads - 1) / threads, threads, 0, s>>>(nspec, d_ltab, g_ltf);
    if (any_emis) {
        // (the table as hi + lo pairs; formed per call: 0.3 ms, and no stale copy if the table is reloaded)
        if (n_dustem > g_dust_n) {
            if (g_dust) (void)hipFree(g_dust);
            if (hipMalloc((void**)&g_dust, n_dustem * sizeof(float2)) != hipSuccess) { g_dust = nullptr; g_dust_n = 0; return; }
            g_dust_n = n_dustem;
        }
        split_dustem_kernel<<<(unsigned)((n_dustem + threads - 1) / threads), threads, 0, s>>>(n_dustem, d_dustem, g_dust);
    }
    attenuate_kernel<<<dim3((nspec + threads - 1) / threads, n_mod), threads, 0, s>>>(nspec, n_mod, nq, d_mp, d_mi, d_ttab, g_ltf, g_dust,
                                                                     d_yo, d_total, d_absorbed, d_tdiff);
    (void)hipStreamSynchronize(s);
}

int opensps_hip_observed_frame(int n_col, int nspec, int npad, int ch, int ldf, double zfloor, const double* d_zc,
                                const double* d_lambda, const float* d_wl, const float* d_total, float* d_fobs,
                                int threads, void* stream) {
    if (n_col <= 0 || nspec <= 1) return 0;
    if (threads < 32) threads = 32;
    if (threads > 128) threads = 128;   // (the kernel's launch bounds)
    hipStream_t s = (hipStream_t)stream;
    if ((size_t)nspec > g_ltf_n) {
        if (g_ltf) (void)hipFree(g_ltf);
        if (hipMalloc((void**)&g_ltf, (size_t)nspec * sizeof(float2)) != hipSuccess) { g_ltf = nullptr; g_ltf_n = 0; return 1; }
        g_ltf_n = nspec;
    }
    split_ltab_kernel<<<(nspec + 255) / 256, 256, 0, s>>>(nspec, d_lambda, g_ltf);
    observed_frame_kernel<<<dim3((n_col + threads - 1) / threads, (npad + ch - 1) / ch), threads, 0, s>>>(
        n_col, nspec, npad, ch, ldf, zfloor, d_zc, g_ltf, d_wl, d_total, d_fobs);
    (void)hipStreamSynchronize(s);
    return 0;
}

static float* g_fbf = nullptr;
static size_t g_fbf_n = 0;
static double* g_dzp = nullptr;
static size_t g_dzp_n = 0;

// Returns 1 if a scratch buffer could not be allocated (the caller then runs its portable kernels)
int opensps_hip_obs_backward(int n_mod, int nspec, int nbands, int threads, int ch, double zfloor, int dzon, int have_s,
                             const double* d_fb_bar, const double* d_zc, const double* d_lambda, const float* d_wl,
                             const int* d_bp, const int* d_bi, const float* d_bk, const float* d_total,
                             const float* d_sbar, int ld_sbar, float* d_gw, double* d_dzo, void* stream) {
    if (n_mod <= 0 || nspec <= 1) return 0;
    if (threads < 32) threads = 32;
    if (threads > 128) threads = 128;   // (the kernels' launch bounds)
    hipStream_t s = (hipStream_t)stream;
    if (ch < 1) ch = 1;
    const int nblk = (nspec + threads - 1) / threads;        // blocks of the rest-pixel gather
    const int nch = (nspec + ch - 1) / ch;                   // chunks of pixels of the cotangent kernel
    const size_t nfb = (size_t)nbands * n_mod, ndz = (size_t)nch * n_mod;
    if ((size_t)nspec > g_ltf_n) {
        if (g_ltf) (void)hipFree(g_ltf);
        if (hipMalloc((void**)&g_ltf, (size_t)nspec * sizeof(float2)) != hipSuccess) { g_ltf = nullptr; g_ltf_n = 0; return 1; }
        g_ltf_n = nspec;
    }
    if (nfb > g_fbf_n) {
        if (g_fbf) (void)hipFree(g_fbf);
        if (hipMalloc((void**)&g_fbf, nfb * sizeof(float)) != hipSuccess) { g_fbf = nullptr; g_fbf_n = 0; return 1; }
        g_fbf_n = nfb;
    }
    if (ndz > g_dzp_n) {
        if (g_dzp) (void)hipFree(g_dzp);
        if (hipMalloc((void**)&g_dzp, ndz * sizeof(double)) != hipSuccess) { g_dzp = nullptr; g_dzp_n = 0; return 1; }
        g_dzp_n = ndz;
    }
    split_ltab_kernel<<<(nspec + 255) / 256, 256, 0, s>>>(nspec, d_lambda, g_ltf);
    obs_fb_to_float_kernel<<<(unsigned)((nfb + 255) / 256), 256, 0, s>>>(nbands, n_mod, d_fb_bar, g_fbf);
    obs_bwd_gw_kernel<<<dim3((n_mod + threads - 1) / threads, nch), threads, 0, s>>>(n_mod, nspec, nbands, ch, dzon, zfloor,
                                                                                     d_zc, g_ltf, d_wl, d_bp, d_bi, d_bk, g_fbf,
                                                                                     d_total, d_gw, g_dzp);
    if (dzon) obs_bwd_dz_kernel<<<(n_mod + 127) / 128, 128, 0, s>>>(nch, n_mod, g_dzp, d_dzo);
    obs_bwd_rest_kernel<<<dim3(nblk, n_mod), threads, 0, s>>>(n_mod, nspec, zfloor, d_zc, g_ltf, have_s, d_sbar, ld_sbar, d_gw);
    (void)hipStreamSynchronize(s);
    return 0;
}

void opensps_hip_band_fluxes_sp(
    int n_mod,
    int nspec,
    int nbands,
    int small_n,
    int rn,
    int kc,
    int kb,
    const float*  d_bands_t,
    const int*    d_j_lo,
    const int*    d_j_hi,
    const int*    d_do_band,
    const float*  d_wt,
    const float*  d_fobs_t,
    double*       d_fband,
    void*         stream
) {
    if (n_mod <= 0 || nbands <= 0) return;
    hipStream_t s = (hipStream_t)stream;
    bool done = false;
    if (n_mod >= small_n && nspec > 0) {
        size_t need = (size_t)nspec * nbands;
        if (need > g_band_A_n) {
            if (g_band_A) (void)hipFree(g_band_A);
            if (hipMalloc((void**)&g_band_A, need * sizeof(float)) != hipSuccess) { g_band_A = nullptr; g_band_A_n = 0; }
            else g_band_A_n = need;
        }
        if (g_band_A) {
            // row tile of 16 / 32 / 64 bands by the number of bands (RMB), columns and stages from the registry
            const int rmb = nbands <= 16 ? 1 : (nbands <= 32 ? 2 : 4);
#define BAND_CASE(RMB, RN, KB) if (rmb == RMB && rn == RN && kb == KB && kc == 16) { \
                band_matrix_kernel<<<dim3((nspec + 255) / 256, nbands), 256, 0, s>>>( \
                    nspec, nbands, d_bands_t, d_j_lo, d_j_hi, d_do_band, d_wt, g_band_A); \
                band_gemm_kernel<RMB, RN, 16, KB><<<dim3((n_mod + 16 * RN - 1) / (16 * RN), (nbands + 16 * RMB - 1) / (16 * RMB)), 256, 0, s>>>( \
                    nbands, n_mod, nspec, g_band_A, d_fobs_t, d_fband); done = true; }
#define BAND_CASES(RMB) BAND_CASE(RMB, 4, 4) BAND_CASE(RMB, 4, 8) BAND_CASE(RMB, 8, 4) BAND_CASE(RMB, 8, 8)
            BAND_CASES(1) BAND_CASES(2) BAND_CASES(4)
#undef BAND_CASES
#undef BAND_CASE
        }
    }
    if (!done) {
        if (n_mod < small_n) {
            band_fluxes_small_kernel<float><<<dim3(n_mod, nbands), BF_SMALL_THREADS, 0, s>>>(
                n_mod, nspec, nbands, d_bands_t, d_j_lo, d_j_hi, d_do_band, d_wt, d_fobs_t, d_fband
            );
        } else {
            dim3 block(32, 1);
            dim3 grid((n_mod + 31) / 32, nbands);
            band_fluxes_kernel<float><<<grid, block, 0, s>>>(
                n_mod, nspec, nbands, d_bands_t, d_j_lo, d_j_hi, d_do_band, d_wt, d_fobs_t, d_fband
            );
        }
    }
    (void)hipStreamSynchronize(s);
}

void opensps_hip_band_fluxes_dp(
    int n_mod,
    int nspec,
    int nbands,
    int small_n,
    const double* d_bands_t,
    const int*    d_j_lo,
    const int*    d_j_hi,
    const int*    d_do_band,
    const double* d_wt,
    const double* d_fobs_t,
    double*       d_fband,
    void*         stream
) {
    if (n_mod <= 0 || nbands <= 0) return;
    hipStream_t s = (hipStream_t)stream;
    if (n_mod < small_n) {
        band_fluxes_small_kernel<double><<<dim3(n_mod, nbands), BF_SMALL_THREADS, 0, s>>>(
            n_mod, nspec, nbands, d_bands_t, d_j_lo, d_j_hi, d_do_band, d_wt, d_fobs_t, d_fband
        );
    } else {
        dim3 block(32, 1);
        dim3 grid((n_mod + 31) / 32, nbands);
        band_fluxes_kernel<double><<<grid, block, 0, s>>>(
            n_mod, nspec, nbands, d_bands_t, d_j_lo, d_j_hi, d_do_band, d_wt, d_fobs_t, d_fband
        );
    }
    (void)hipStreamSynchronize(s);
}

} // extern "C"


// ------------------------------------------------------------------------------------------------
// cue's hidden layers (opensps_cue): for the 17 networks g, Y(u, m, g) = act(b(u, g) + sum_k
// W(u, k, g) X(k, m, g)), act(a) = (beta + (1 - beta) / (1 + exp(-alpha a))) a. Fortran (column-major)
// arrays: W (256, K, 17), X (K, ld, 17), Y (256, ld, 17), b, alpha, beta (256, 17). TW: the weights'
// type (FP32 as trained, or FP64 in FP64 builds), TA: the activations' and the accumulation's. A
// 64 x 64 tile of (unit, model) per block of 256 threads, 4 x 4 outputs per thread, K in steps of 16
// through shared memory (the weights converted exactly to TA as they are staged).
// ------------------------------------------------------------------------------------------------
#define CUE_BU 64
#define CUE_BM 64
#define CUE_BK 16
#define CUE_NH 256

template <typename TW, typename TA>
__global__ void __launch_bounds__(256) cue_dense_act_kernel(
    int K, int nc, int ld,
    const TW* __restrict__ W, const TW* __restrict__ b, const TW* __restrict__ al, const TW* __restrict__ be,
    const TA* __restrict__ X, TA* __restrict__ Y, TA* __restrict__ P)
{
    __shared__ TA sw[CUE_BK][CUE_BU + 1];
    __shared__ TA sx[CUE_BK][CUE_BM + 1];
    const int g = blockIdx.z;
    const int u0 = blockIdx.x * CUE_BU;
    const int m0 = blockIdx.y * CUE_BM;
    const int tid = threadIdx.x;
    const int tu = tid % 16;     // 16 x 16 threads, each 4 x 4 outputs
    const int tm = tid / 16;
    const TW* Wg = W + (size_t)g * CUE_NH * K;
    const TA* Xg = X + (size_t)g * K * ld;
    TA acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0;

    for (int k0 = 0; k0 < K; k0 += CUE_BK) {
        // Stage W(u0:u0+63, k0:k0+15) and X(k0:k0+15, m0:m0+63): 1024 values each, 4 per thread
#pragma unroll
        for (int r = 0; r < 4; r++) {
            int e = tid + 256 * r;
            int uu = e % CUE_BU, kk = e / CUE_BU;              // W: u fastest (coalesced)
            int k = k0 + kk;
            sw[kk][uu] = (k < K) ? (TA)Wg[(size_t)k * CUE_NH + u0 + uu] : (TA)0;
            int kx = e % CUE_BK, mm = e / CUE_BK;              // X: k fastest (coalesced)
            int kk2 = k0 + kx, m = m0 + mm;
            sx[kx][mm] = (kk2 < K && m < nc) ? Xg[(size_t)m * K + kk2] : (TA)0;
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < CUE_BK; kk++) {
            TA a[4], x[4];
#pragma unroll
            for (int i = 0; i < 4; i++) a[i] = sw[kk][tu + 16 * i];
#pragma unroll
            for (int j = 0; j < 4; j++) x[j] = sx[kk][tm + 16 * j];
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) acc[i][j] += a[i] * x[j];
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
        int u = u0 + tu + 16 * i;
        TA bb = (TA)b[g * CUE_NH + u], aa = (TA)al[g * CUE_NH + u], ee = (TA)be[g * CUE_NH + u];
#pragma unroll
        for (int j = 0; j < 4; j++) {
            int m = m0 + tm + 16 * j;
            if (m < nc) {
                TA v = acc[i][j] + bb;
                // P: the pre-activation too (the reverse pass's recompute); Y: its activation (null: skipped)
                if (P) P[((size_t)g * ld + m) * CUE_NH + u] = v;
                if (Y) Y[((size_t)g * ld + m) * CUE_NH + u] = (ee + ((TA)1 - ee) / ((TA)1 + exp(-aa * v))) * v;
            }
        }
    }
}

template <typename TW, typename TA>
static void cue_dense_act_launch(int K, int nc, int ld, const void* W, const void* b, const void* al,
                                 const void* be, const void* X, void* Y, void* P, void* stream)
{
    if (nc <= 0) return;
    dim3 block(256);
    dim3 grid(CUE_NH / CUE_BU, (nc + CUE_BM - 1) / CUE_BM, 17);
    hipStream_t s = (hipStream_t)stream;
    cue_dense_act_kernel<TW, TA><<<grid, block, 0, s>>>(K, nc, ld, (const TW*)W, (const TW*)b, (const TW*)al,
                                                        (const TW*)be, (const TA*)X, (TA*)Y, (TA*)P);
    (void)hipDeviceSynchronize();
}

// din(k, m, g) = sum_u W(u, k, g) dout(u, m, g), k = 1..256 (the transpose of a 256 x 256 layer)
template <typename TW, typename TA>
__global__ void __launch_bounds__(256) cue_dense_back_kernel(
    int nc, int ld, const TW* __restrict__ W, const TA* __restrict__ D, TA* __restrict__ O)
{
    __shared__ TA sw[CUE_BK][CUE_BU + 1];
    __shared__ TA sd[CUE_BK][CUE_BM + 1];
    const int g = blockIdx.z;
    const int k0 = blockIdx.x * CUE_BU;
    const int m0 = blockIdx.y * CUE_BM;
    const int tid = threadIdx.x;
    const int tk = tid % 16;
    const int tm = tid / 16;
    const TW* Wg = W + (size_t)g * CUE_NH * CUE_NH;
    const TA* Dg = D + (size_t)g * ld * CUE_NH;
    TA acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0;

    for (int u0 = 0; u0 < CUE_NH; u0 += CUE_BK) {
        // W(u0:u0+15, k0:k0+63) and dout(u0:u0+15, m0:m0+63): u fastest in both (16 contiguous values)
#pragma unroll
        for (int r = 0; r < 4; r++) {
            int e = tid + 256 * r;
            int uu = e % CUE_BK, q = e / CUE_BK;
            sw[uu][q] = (TA)Wg[(size_t)(k0 + q) * CUE_NH + u0 + uu];
            int m = m0 + q;
            sd[uu][q] = (m < nc) ? Dg[(size_t)m * CUE_NH + u0 + uu] : (TA)0;
        }
        __syncthreads();
#pragma unroll
        for (int uu = 0; uu < CUE_BK; uu++) {
            TA a[4], x[4];
#pragma unroll
            for (int i = 0; i < 4; i++) a[i] = sw[uu][tk + 16 * i];
#pragma unroll
            for (int j = 0; j < 4; j++) x[j] = sd[uu][tm + 16 * j];
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) acc[i][j] += a[i] * x[j];
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) {
            int m = m0 + tm + 16 * j;
            if (m < nc) O[((size_t)g * ld + m) * CUE_NH + k0 + tk + 16 * i] = acc[i][j];
        }
}

template <typename TW, typename TA>
static void cue_dense_back_launch(int nc, int ld, const void* W, const void* D, void* O, void* stream)
{
    if (nc <= 0) return;
    dim3 block(256);
    dim3 grid(CUE_NH / CUE_BU, (nc + CUE_BM - 1) / CUE_BM, 17);
    cue_dense_back_kernel<TW, TA><<<grid, block, 0, (hipStream_t)stream>>>(nc, ld, (const TW*)W, (const TA*)D, (TA*)O);
    (void)hipDeviceSynchronize();
}

extern "C" {
// Weights FP32 (as trained): activations FP32 (_ss) or FP64 (_sd); FP64 builds: _dd
void opensps_hip_cue_dense_act_ss(int K, int nc, int ld, const void* W, const void* b, const void* al,
                                  const void* be, const void* X, void* Y, void* stream) {
    cue_dense_act_launch<float, float>(K, nc, ld, W, b, al, be, X, Y, nullptr, stream);
}
void opensps_hip_cue_dense_act_sd(int K, int nc, int ld, const void* W, const void* b, const void* al,
                                  const void* be, const void* X, void* Y, void* stream) {
    cue_dense_act_launch<float, double>(K, nc, ld, W, b, al, be, X, Y, nullptr, stream);
}
void opensps_hip_cue_dense_act_dd(int K, int nc, int ld, const void* W, const void* b, const void* al,
                                  const void* be, const void* X, void* Y, void* stream) {
    cue_dense_act_launch<double, double>(K, nc, ld, W, b, al, be, X, Y, nullptr, stream);
}
// The reverse pass: FP64 activations, with the pre-activation also written to P (Y may be null)
void opensps_hip_cue_dense_pre_sd(int K, int nc, int ld, const void* W, const void* b, const void* al,
                                  const void* be, const void* X, void* Y, void* P, void* stream) {
    cue_dense_act_launch<float, double>(K, nc, ld, W, b, al, be, X, Y, P, stream);
}
void opensps_hip_cue_dense_pre_dd(int K, int nc, int ld, const void* W, const void* b, const void* al,
                                  const void* be, const void* X, void* Y, void* P, void* stream) {
    cue_dense_act_launch<double, double>(K, nc, ld, W, b, al, be, X, Y, P, stream);
}
void opensps_hip_cue_dense_back_sd(int nc, int ld, const void* W, const void* D, void* O, void* stream) {
    cue_dense_back_launch<float, double>(nc, ld, W, D, O, stream);
}
void opensps_hip_cue_dense_back_dd(int nc, int ld, const void* W, const void* D, void* O, void* stream) {
    cue_dense_back_launch<double, double>(nc, ld, W, D, O, stream);
}
} // extern "C"
