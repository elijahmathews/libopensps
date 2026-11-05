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

    for (int k0 = 0; k0 < nspec; k0 += TILE_K) {
        int k = k0 + tx;
        if (j_local < n_run_models && k < nspec) {
            s_gwy[tx][ty] = d_gw[(size_t)model_idx * nspec + k];
            s_gwo[tx][ty] = d_gw[(size_t)(n_mod + model_idx) * nspec + k];
        } else {
            s_gwy[tx][ty] = 0;
            s_gwo[tx][ty] = 0;
        }
        __syncthreads();

        int k_end = (k0 + TILE_K < nspec) ? (k0 + TILE_K) : nspec;
        if (e < n_eff && j_local < n_run_models) {
            for (int curr_k = k0; curr_k < k_end; ++curr_k) {
                int k_sub = curr_k - k0;
                T ev = d_etpl_t[(size_t)curr_k * n_eff + e];
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
    float b = (n < 0) ? 0.5f : 2.0f;
    int k = abs(n);
    float p2 = 1.0f;
    while (k > 0) {
        if (k & 1) p2 *= b;
        b *= b;
        k >>= 1;
    }
    return expf((float)(xc - (double)n * LN2)) * p2;
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
    hipDeviceSynchronize();
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
    hipDeviceSynchronize();
}

}

template <typename T_KP>
__global__ void fold_multi_kernel(
    int m0_0based,
    int nc,
    int n_eff_max,
    int nt_neb,
    int neb_cols,
    int ldw,
    int n_mod,
    const double* __restrict__ d_wsfh,
    const int*    __restrict__ d_it_tesc,
    const double* __restrict__ d_fesc,
    const int*    __restrict__ d_n_eff,
    const int*    __restrict__ d_pt_off,
    const int*    __restrict__ d_seg_off,
    const int*    __restrict__ d_eff_off,
    const int*    __restrict__ d_pt_age,
    const int*    __restrict__ d_eff_pt,
    const double* __restrict__ d_eff_val,
    const int*    __restrict__ d_seg_ptr,
    const int*    __restrict__ d_seg_first,
    const int*    __restrict__ d_perm,
    const int*    __restrict__ d_viz,
    const double* __restrict__ d_cz,
    const T_KP*   __restrict__ d_w,
    T_KP*         __restrict__ d_v,
    T_KP*         __restrict__ d_veuv
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (e >= n_eff_max || j >= nc) return;

    int idx = m0_0based + j;
    int iz = d_viz[idx];
    int iz_0 = iz - 1;

    if (e >= d_n_eff[iz_0]) return;

    int m = d_perm[idx] - 1;
    int tesc = d_it_tesc[m];
    double f = (d_fesc != nullptr) ? d_fesc[m] : 1.0;
    double cz_m = d_cz[m];

    int eg_0 = d_eff_off[iz_0] + e;
    int s_start = d_seg_first[eg_0];
    int s_end   = d_seg_first[eg_0 + 1];

    double y  = 0.0;
    double o  = 0.0;
    double ye = 0.0;
    double oe = 0.0;

    for (int s = s_start; s < s_end; ++s) {
        int sg_0 = d_seg_off[iz_0] + (s - 1);
        int k_start = d_seg_ptr[sg_0];
        int k_end   = d_seg_ptr[sg_0 + 1];

        int tc = 0;
        double a = 0.0;

        for (int k_1 = k_start; k_1 <= k_end; ++k_1) {
            int t;
            int k_0 = k_1 - 1;
            if (k_1 < k_end) {
                int p_1 = d_eff_pt[k_0];
                t = d_pt_age[d_pt_off[iz_0] + (p_1 - 1)];
            } else {
                t = -1;
            }

            if (t != tc) {
                if (tc > 0) {
                    double wsfh_val = d_wsfh[(size_t)(tc - 1) * n_mod + m];
                    double x = a * wsfh_val * cz_m;
                    double ae = x;
                    if (tc <= nt_neb) ae = f * x;
                    if (tc <= tesc) {
                        y  += x;
                        ye += ae;
                    } else {
                        o  += x;
                        oe += ae;
                    }
                }
                tc = t;
                a = 0.0;
            }

            if (t > 0) {
                int p_0 = d_eff_pt[k_0] - 1;
                double w_val = (double)d_w[j + (size_t)p_0 * ldw];
                a += d_eff_val[k_0] * w_val;
            }
        }
    }

    size_t ldv = (size_t)n_eff_max;
    d_v[e + (size_t)j * ldv] = (T_KP)y;
    d_v[e + (size_t)(nc + j) * ldv] = (T_KP)o;
    if (neb_cols > 0 && d_veuv != nullptr) {
        d_veuv[e + (size_t)j * ldv] = (T_KP)ye;
        d_veuv[e + (size_t)(nc + j) * ldv] = (T_KP)oe;
    }
}

extern "C" {

void opensps_hip_fold_multi_sp(
    int m0_0based,
    int nc,
    int n_eff_max,
    int nt_neb,
    int neb_cols,
    int ldw,
    int n_mod,
    const double* __restrict__ d_wsfh,
    const int*    __restrict__ d_it_tesc,
    const double* __restrict__ d_fesc,
    const int*    __restrict__ d_n_eff,
    const int*    __restrict__ d_pt_off,
    const int*    __restrict__ d_seg_off,
    const int*    __restrict__ d_eff_off,
    const int*    __restrict__ d_pt_age,
    const int*    __restrict__ d_eff_pt,
    const double* __restrict__ d_eff_val,
    const int*    __restrict__ d_seg_ptr,
    const int*    __restrict__ d_seg_first,
    const int*    __restrict__ d_perm,
    const int*    __restrict__ d_viz,
    const double* __restrict__ d_cz,
    const float*  __restrict__ d_w,
    float*        __restrict__ d_v,
    float*        __restrict__ d_veuv,
    void* stream
) {
    if (nc <= 0 || n_eff_max <= 0) return;
    dim3 block(16, 16);
    dim3 grid((n_eff_max + 15) / 16, (nc + 15) / 16);
    hipStream_t s = (hipStream_t)stream;
    fold_multi_kernel<float><<<grid, block, 0, s>>>(
        m0_0based, nc, n_eff_max, nt_neb, neb_cols, ldw, n_mod,
        d_wsfh, d_it_tesc, d_fesc, d_n_eff, d_pt_off, d_seg_off, d_eff_off,
        d_pt_age, d_eff_pt, d_eff_val, d_seg_ptr, d_seg_first,
        d_perm, d_viz, d_cz, d_w, d_v, d_veuv
    );
}

void opensps_hip_fold_multi_dp(
    int m0_0based,
    int nc,
    int n_eff_max,
    int nt_neb,
    int neb_cols,
    int ldw,
    int n_mod,
    const double* __restrict__ d_wsfh,
    const int*    __restrict__ d_it_tesc,
    const double* __restrict__ d_fesc,
    const int*    __restrict__ d_n_eff,
    const int*    __restrict__ d_pt_off,
    const int*    __restrict__ d_seg_off,
    const int*    __restrict__ d_eff_off,
    const int*    __restrict__ d_pt_age,
    const int*    __restrict__ d_eff_pt,
    const double* __restrict__ d_eff_val,
    const int*    __restrict__ d_seg_ptr,
    const int*    __restrict__ d_seg_first,
    const int*    __restrict__ d_perm,
    const int*    __restrict__ d_viz,
    const double* __restrict__ d_cz,
    const double* __restrict__ d_w,
    double*       __restrict__ d_v,
    double*       __restrict__ d_veuv,
    void* stream
) {
    if (nc <= 0 || n_eff_max <= 0) return;
    dim3 block(16, 16);
    dim3 grid((n_eff_max + 15) / 16, (nc + 15) / 16);
    hipStream_t s = (hipStream_t)stream;
    fold_multi_kernel<double><<<grid, block, 0, s>>>(
        m0_0based, nc, n_eff_max, nt_neb, neb_cols, ldw, n_mod,
        d_wsfh, d_it_tesc, d_fesc, d_n_eff, d_pt_off, d_seg_off, d_eff_off,
        d_pt_age, d_eff_pt, d_eff_val, d_seg_ptr, d_seg_first,
        d_perm, d_viz, d_cz, d_w, d_v, d_veuv
    );
}

}
