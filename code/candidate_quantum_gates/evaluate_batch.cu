// One logical DAG per CUDA thread; structure-of-arrays workspace across batch.
// Positive durations and int32 total-duration bound are checked by Python.
// Cycles return the same sentinel as the independent CPU evaluator.
extern "C" __global__ void evaluate_batch(
    const int* orders, const int* p, int* work, int* out,
    int B, int J, int M) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    int N = J * M;
    int* successor = work;
    int* degree = work + N * B;
    int* head = work + 2 * N * B;
    int* queue = work + 3 * N * B;
    for (int v = 0; v < N; ++v) {
        successor[v*B+b] = -1;
        degree[v*B+b] = (v % M != 0);
        head[v*B+b] = 0;
    }
    for (int m = 0; m < M; ++m) for (int k = 1; k < J; ++k) {
        int u = orders[(m*J+k-1)*B+b], v = orders[(m*J+k)*B+b];
        successor[u*B+b] = v;
        ++degree[v*B+b];
    }
    int size = 0;
    for (int v = 0; v < N; ++v)
        if (!degree[v*B+b]) queue[(size++)*B+b] = v;
    int makespan = 0;
    for (int i = 0; i < size; ++i) {
        int u = queue[i*B+b], finish = head[u*B+b] + p[u];
        if (finish > makespan) makespan = finish;
        for (int edge = 0; edge < 2; ++edge) {
            int v = edge ? successor[u*B+b] : ((u%M < M-1) ? u+1 : -1);
            if (v >= 0) {
                if (finish > head[v*B+b]) head[v*B+b] = finish;
                if (--degree[v*B+b] == 0) queue[(size++)*B+b] = v;
            }
        }
    }
    out[b] = size == N ? makespan : (1 << 30);
}
