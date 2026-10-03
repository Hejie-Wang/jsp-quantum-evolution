// Tiled coalesced transpose of a rows x cols int32 matrix. Used to convert a
// natively laid-out (B, M*J) candidate batch into the (M*J, B) layout the
// evaluate_batch kernel reads; CuPy's generic copy kernel is uncoalesced for
// this permutation and ends up as slow as a host-side transpose.
// Host-emulated logic testing skips this file: correctness of the tiled
// transpose is covered by the device parity test (chunked base/stride).
extern "C" __global__ void transpose_bm_mj(
    const int* src, int* dst, int rows, int cols) {
    __shared__ int tile[32][33];
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    if (x < cols && y < rows) tile[threadIdx.y][threadIdx.x] = src[y*cols+x];
    __syncthreads();
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;
    if (x < rows && y < cols) dst[y*rows+x] = tile[threadIdx.x][threadIdx.y];
}
