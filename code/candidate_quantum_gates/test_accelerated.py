import unittest
import ctypes
import shutil
import subprocess
import tempfile
from pathlib import Path
import numpy as np
from circuits import CandidatePool, JointTransition, WitnessSpec
from compact_simulator import CompactSimulator
from batch_evaluator import BatchEvaluator, INVALID, evaluate_one
from adaptive_search import solve, gt_initial, critical_moves, apply_moves
from search_loop import (demo_instance, demo_candidate_pool, covering_demo_witnesses,
                         build_fixed_t_ansatz, mq, train_layer_params)


class AcceleratedTests(unittest.TestCase):
    def test_batch_matches_independent_graph(self):
        inst = demo_instance()
        rng = np.random.default_rng(17)
        orders = np.array([[rng.permutation(g) for g in inst.groups] for _ in range(80)])
        values = BatchEvaluator(inst).evaluate(orders)
        expected = []
        for order in orders:
            d = mq.decode(inst, order)
            expected.append(d['makespan'] if d['feasible'] else INVALID)
        np.testing.assert_array_equal(values, expected)
        self.assertIn(INVALID, expected)
        self.assertTrue(any(v != INVALID for v in expected))

    def test_input_validation(self):
        inst = demo_instance()
        evaluator = BatchEvaluator(inst)
        with self.assertRaises(ValueError):
            evaluator.evaluate(np.zeros((2, 3, 3), dtype=int))
        with self.assertRaises(ValueError):
            evaluator.evaluate(np.zeros((2, 3, 3)))
        self.assertEqual(len(evaluator.evaluate(np.empty((0, 3, 3), dtype=int))), 0)

    def test_head_tail_and_gt(self):
        inst = demo_instance()
        p, routes = np.array(inst.processing, np.int32), np.array(inst.machine_of, np.int32)
        for seed in range(8):
            order = gt_initial(p, routes, inst.jobs, inst.machines, seed)
            c, head, tail = evaluate_one(order, p, inst.machines)
            self.assertEqual(mq.validate_schedule(inst, head), c)
            self.assertTrue(np.all(head + p + tail <= c))
            batch = apply_moves(order, critical_moves(order, p, head, tail, c))
            BatchEvaluator(inst).evaluate(batch)

    def test_compact_amplitudes_match_gates(self):
        from qiskit.quantum_info import Statevector
        pool = demo_candidate_pool()
        _, specs = covering_demo_witnesses(pool)
        joint = (JointTransition((0, 2), (0, 1), (1, 0), weight=0.7),)
        params = [(0.13, 0.27, 0.44), (0.21, -0.14, 0.17)]
        sim = CompactSimulator(pool, specs, joint, 14, penalty=3.2)
        for mode in ('xy', 'xy_joint', 'uniform'):
            circuit = build_fixed_t_ansatz(pool, specs, joint, params, 14, 3.2, mode=mode)
            state = np.asarray(Statevector(circuit).data)
            indices = [sum(1 << pool.offset(m, int(a)) for m, a in enumerate(ch))
                       for ch in sim.labels]
            np.testing.assert_allclose(sim.state(params, mode), state[indices], atol=1e-10)
            self.assertAlmostEqual(float(np.sum(np.abs(state[indices])**2)), 1.0)

    def test_unequal_pool_sizes_and_constant_witness(self):
        from qiskit.quantum_info import Statevector
        pool = CandidatePool((((0, 1, 2), (1, 0, 2), (2, 1, 0)), ((3, 4), (4, 3))))
        specs = (WitnessSpec('cycle', ((0, 1, 2), (0, 1)), pool_hash=pool.content_hash),
                 WitnessSpec('path', ((1, 2), (1,)), length=9, pool_hash=pool.content_hash))
        sim = CompactSimulator(pool, specs, (), 8, 2)
        params = [(0.12, 0.36, 0.0)]
        state = np.asarray(Statevector(build_fixed_t_ansatz(pool, specs, (), params, 8, 2)).data)
        indices = [sum(1 << pool.offset(m, int(a)) for m, a in enumerate(ch)) for ch in sim.labels]
        np.testing.assert_allclose(sim.state(params), state[indices], atol=1e-10)

    def test_compact_budget_and_pool_binding(self):
        pool = demo_candidate_pool()
        with self.assertRaises(ValueError):
            CompactSimulator(pool, (), (), 10, max_states=7)
        with self.assertRaises(ValueError):
            CompactSimulator(pool, (WitnessSpec('cycle', ((0,),)*3, pool_hash='stale'),), (), 10)

    def test_training_matches_reference(self):
        pool = demo_candidate_pool()
        _, specs = covering_demo_witnesses(pool)
        kw = dict(layer_counts=(1,), max_evaluations=4, mode='xy')
        a = train_layer_params(pool, specs, (), 13, 1, simulation_backend='compact', **kw)
        b = train_layer_params(pool, specs, (), 13, 1, simulation_backend='qiskit', **kw)
        self.assertAlmostEqual(a['expected_energy'], b['expected_energy'], places=10)
        np.testing.assert_allclose(a['params'], b['params'])

    def test_adaptive_search_and_proposal_accounting(self):
        inst, pool = demo_instance(), demo_candidate_pool()
        for mode in ('classical', 'uniform', 'quantum'):
            report = solve(inst, mode=mode, seconds=20, max_iterations=12,
                           proposal_every=2, train_budget=3, shots=8,
                           initial_orders=[p[0] for p in pool.candidates])
            self.assertTrue(report['independent_schedule_verification'])
            self.assertLessEqual(report['best_makespan'], report['initial_makespan'])
            self.assertEqual(report['hardware_jobs_submitted'], 0)
            self.assertEqual(report['iterations'], 12)
            if mode != 'classical':
                self.assertEqual(report['proposal_calls'], 6)

    def test_cuda_parity_if_available(self):
        inst = demo_instance()
        try:
            import cupy as cp
        except ImportError as exc:
            self.skipTest(str(exc))
        try:
            devices = cp.cuda.runtime.getDeviceCount()
        except cp.cuda.runtime.CUDARuntimeError as exc:
            self.skipTest(str(exc))
        if not devices:
            self.skipTest('no CUDA device')
        # Device present: compilation or execution errors must fail, not skip.
        evaluator = BatchEvaluator(inst, 'cuda', chunk_size=7)
        rng = np.random.default_rng(71)
        batch = np.array([[rng.permutation(g) for g in inst.groups] for _ in range(41)])
        np.testing.assert_array_equal(evaluator.evaluate(batch), BatchEvaluator(inst).evaluate(batch))

    def test_cuda_kernel_body_on_host(self):
        """Check the actual .cu indexing/logic without claiming device execution."""
        compiler = shutil.which('g++')
        if not compiler:
            self.skipTest('C++ compiler unavailable for kernel-body check')
        kernel = Path(__file__).with_name('evaluate_batch.cu').read_text()
        shim = '#define __global__\nstruct Dim { int x; };\nDim blockIdx,blockDim,threadIdx;\n'
        shim += kernel
        shim += '''
extern "C" void run_host(const int* o, const int* p, int* w, int* out, int B,int J,int M){
  blockDim.x=128;
  for(int b=0;b<B+7;++b){blockIdx.x=b/128;threadIdx.x=b%128;evaluate_batch(o,p,w,out,B,J,M,0,B);}
}
'''
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            source, library = Path(directory)/'kernel.cpp', Path(directory)/'kernel.so'
            source.write_text(shim)
            subprocess.run([compiler, '-shared', '-fPIC', '-O2', str(source), '-o', str(library)], check=True)
            run = ctypes.CDLL(str(library)).run_host
            ptr = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
            run.argtypes = [ptr, ptr, ptr, ptr, ctypes.c_int, ctypes.c_int, ctypes.c_int]
            run.restype = None
            inst, rng = demo_instance(), np.random.default_rng(63)
            batch = np.array([[rng.permutation(g) for g in inst.groups] for _ in range(137)], np.int32)
            orders = np.ascontiguousarray(batch.transpose(1,2,0))
            work = np.empty((4,inst.operations,len(batch)), np.int32)
            out = np.empty(len(batch), np.int32)
            run(orders, np.array(inst.processing,np.int32),work,out,len(batch),inst.jobs,inst.machines)
            np.testing.assert_array_equal(out, BatchEvaluator(inst).evaluate(batch))


if __name__ == '__main__':
    unittest.main()
