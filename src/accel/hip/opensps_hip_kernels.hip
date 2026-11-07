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
    adj_nebular_continuum_kernel<float><<<grid, block, 0, s>>>(
        n_mod, nspec, ntab, ntn, nrun, d_tab, d_gw, d_it_tesc, d_ia, d_wa,
        d_rs, d_rl, d_iz, d_wz, d_iu, d_wu, d_fabs, d_qzt, d_lref, d_csum
    );
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
    int tj = threadIdx.x; // 0..31 (model j)
    int tq = threadIdx.y; // 0..3 (piece within block)
    int j = tj;
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

        d_w[out_idx] = 0.0;
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
        d_w[out_idx] = wv;
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

extern "C" {

void opensps_hip_imf_pieces_weights_sp(
    int m0_0based,
    int nc,
    int n_pc_max,
    int n_pts_max,
    int ldw,
    int imf_type,
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
    void*         stream
) {
    if (nc <= 0 || n_pc_max <= 0) return;
    dim3 block(32, 4);
    int grid_j = (nc + 31) / 32;
    int grid_q = (n_pc_max + 3) / 4;
    dim3 grid(grid_j, grid_q);
    hipStream_t s = (hipStream_t)stream;
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
    dim3 block(32, 4);
    int grid_j = (nc + 31) / 32;
    int grid_q = (n_pc_max + 3) / 4;
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

extern "C" {

void opensps_hip_band_fluxes_sp(
    int n_mod,
    int nspec,
    int nbands,
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
    dim3 block(32, 1);
    dim3 grid((n_mod + 31) / 32, nbands);
    hipStream_t s = (hipStream_t)stream;
    band_fluxes_kernel<float><<<grid, block, 0, s>>>(
        n_mod, nspec, nbands, d_bands_t, d_j_lo, d_j_hi, d_do_band, d_wt, d_fobs_t, d_fband
    );
    (void)hipStreamSynchronize(s);
}

void opensps_hip_band_fluxes_dp(
    int n_mod,
    int nspec,
    int nbands,
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
    dim3 block(32, 1);
    dim3 grid((n_mod + 31) / 32, nbands);
    hipStream_t s = (hipStream_t)stream;
    band_fluxes_kernel<double><<<grid, block, 0, s>>>(
        n_mod, nspec, nbands, d_bands_t, d_j_lo, d_j_hi, d_do_band, d_wt, d_fobs_t, d_fband
    );
    (void)hipStreamSynchronize(s);
}

} // extern "C"
