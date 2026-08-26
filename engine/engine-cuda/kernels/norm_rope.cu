// Port of llama.cpp ggml_compute_forward_rms_norm + rope (ggml/src/ggml.c, ggml/src/ggml-cuda/rope.cu @ cb1adf8)

extern "C" __global__ void norm_rope_swiglu_kernel(
    const float* __restrict__ x,
    const float* __restrict__ residual,
    const float* __restrict__ norm_w,
    int n,
    int n_dims,
    float freq_base,
    unsigned int pos,
    const float* __restrict__ up,
    float* __restrict__ out,
    float eps,
    int mode,
    const unsigned int* __restrict__ pos_ptr)
{
    if (pos_ptr != nullptr) {
        pos = *pos_ptr;
    }

    const int row = blockIdx.x;
    const float* row_x = x + (size_t)row * (size_t)n;
    const float* row_residual = residual + (size_t)row * (size_t)n;
    const float* row_norm_w = norm_w;
    const float* row_up = up + (size_t)row * (size_t)n;
    float* row_out = out + (size_t)row * (size_t)n;

    __shared__ double s_sum[256];
    __shared__ float s_scale;

    // Phase 1: RMSNorm + Residual add (bit0 of mode)
    if (mode & 1) {
        double psum = 0.0;
        for (int j = threadIdx.x; j < n; j += blockDim.x) {
            float tmp = row_x[j] + row_residual[j];
            psum += (double)(tmp * tmp);
        }
        s_sum[threadIdx.x] = psum;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            double sum = s_sum[0];
            float mean = (float)(sum / (double)n);
            s_scale = rsqrtf(mean + eps);
        }
        __syncthreads();

        const float scale = s_scale;
        for (int j = threadIdx.x; j < n; j += blockDim.x) {
            float tmp = row_x[j] + row_residual[j];
            row_out[j] = tmp * scale * row_norm_w[j];
        }
    } else {
        for (int j = threadIdx.x; j < n; j += blockDim.x) {
            row_out[j] = row_x[j] + row_residual[j];
        }
    }

    __syncthreads();

    // Phase 2: Rotary Position Embedding (bit1 of mode)
    if (mode & 2) {
        const int half = n_dims / 2;
        for (int k = threadIdx.x; k < half; k += blockDim.x) {
            float theta = (float)pos * powf(freq_base, -2.0f * (float)k / (float)n_dims);
            float c = cosf(theta);
            float s = sinf(theta);
            float x0 = row_out[k];
            float x1 = row_out[k + half];
            row_out[k] = x0 * c - x1 * s;
            row_out[k + half] = x0 * s + x1 * c;
        }
    }

    __syncthreads();

    // Phase 3: SwiGLU (bit2 of mode)
    if (mode & 4) {
        for (int j = threadIdx.x; j < n; j += blockDim.x) {
            float y = row_out[j];
            float silu = y / (1.0f + expf(-y));
            row_out[j] = silu * row_up[j];
        }
    }
}
