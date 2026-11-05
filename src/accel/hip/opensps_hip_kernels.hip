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

}
