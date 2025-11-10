# two_phase_simplex.py
import sys
import numpy as np
from typing import List, Tuple

class LPSolverTwoPhase:
    def __init__(self):
        self.verbose = False

    @staticmethod
    def parse_lp_text(text: str):
        lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip() and not ln.strip().startswith('#')]
        sense = lines[0].lower()
        if sense not in ('min', 'max'):
            raise ValueError("First line must be 'min' or 'max'")
        c = np.array([float(x) for x in lines[1].split()], dtype=float)
        mtx, rhs, signs = [], [], []
        for ln in lines[2:]:
            if ln.lower() == 'end':
                break
            if '<=' in ln:
                left, b = ln.split('<='); signs.append('<=')
            elif '>=' in ln:
                left, b = ln.split('>='); signs.append('>=')
            elif '=' in ln:
                left, b = ln.split('='); signs.append('=')
            else:
                raise ValueError("Constraint line must contain <=, >=, or =")
            a = [float(x) for x in left.split()]
            if len(a) != len(c):
                raise ValueError("Constraint has wrong number of coefficients")
            mtx.append(a); rhs.append(float(b))
        A = np.array(mtx, dtype=float); b = np.array(rhs, dtype=float)
        return sense, c, A, signs, b

    @staticmethod
    def _pivot(T: np.ndarray, row: int, col: int):
        piv = T[row, col]
        T[row, :] /= piv
        for r in range(T.shape[0]):
            if r != row:
                T[r, :] -= T[r, col] * T[row, :]

    @staticmethod
    def _enter(z_row: np.ndarray) -> int:
        # Bland: самый левый отрицательный редуц.стоимость
        idxs = [j for j, v in enumerate(z_row[:-1]) if v < -1e-10]
        return min(idxs) if idxs else -1

    @staticmethod
    def _leave(T: np.ndarray, col: int) -> int:
        rows = T.shape[0] - 1
        rhs = T.shape[1] - 1
        cand = []
        for i in range(rows):
            a = T[i, col]
            if a > 1e-10:
                cand.append((T[i, rhs] / a, i))
        if not cand:
            return -1
        cand.sort(key=lambda x: (x[0], x[1]))
        return cand[0][1]

    def _phase1_tableau(self, A: np.ndarray, signs: List[str], b: np.ndarray):
        m, n = A.shape
        A2, b2, signs2 = A.copy().astype(float), b.copy().astype(float), signs[:]
        # Сделаем b >= 0
        for i in range(m):
            if b2[i] < 0:
                A2[i, :] *= -1; b2[i] *= -1
                if signs2[i] == '<=': signs2[i] = '>='
                elif signs2[i] == '>=': signs2[i] = '<='

        num_s = sum(1 for s in signs2 if s == '<=')
        num_t = sum(1 for s in signs2 if s == '>=')
        num_a = sum(1 for s in signs2 if s in ('=', '>='))

        total = n + num_s + num_t + num_a
        T = np.zeros((m + 1, total + 1))
        xcols = list(range(n))
        sbase, tbase, abase = n, n + num_s, n + num_s + num_t
        si = ti = ai = 0
        basis = []

        for i in range(m):
            T[i, xcols] = A2[i, :]
            if signs2[i] == '<=':
                T[i, sbase + si] = 1.0; basis.append(sbase + si); si += 1
            elif signs2[i] == '=':
                T[i, abase + ai] = 1.0; basis.append(abase + ai); ai += 1
            elif signs2[i] == '>=':
                T[i, tbase + ti] = -1.0
                T[i, abase + ai] = 1.0
                basis.append(abase + ai); ti += 1; ai += 1
            T[i, -1] = b2[i]

        # Фаза I: max -sum(a)
        z = np.zeros(total + 1)
        for r, bc in enumerate(basis):
            if bc >= abase:
                z -= T[r, :]
        for j in range(abase, abase + num_a):
            if j not in basis:
                z[j] = -1.0
        T[-1, :] = z
        return T, basis, (n, num_s, num_t, num_a)

    def _simplex(self, T: np.ndarray, basis: List[int]):
        for _ in range(1000):
            j = self._enter(T[-1, :])
            if j == -1:
                return 'optimal', T, basis
            i = self._leave(T, j)
            if i == -1:
                return 'unbounded', T, basis
            self._pivot(T, i, j); basis[i] = j
        return 'iteration_limit', T, basis

    def solve(self, sense: str, c: np.ndarray, A: np.ndarray, signs: List[str], b: np.ndarray):
        # Фаза I
        T, basis, counts = self._phase1_tableau(A, signs, b)
        s1, T1, basis1 = self._simplex(T, basis)
        if s1 != 'optimal': return {'status': f'Phase I failed: {s1}'}
        if T1[-1, -1] < -1e-8: return {'status': 'infeasible'}

        n, ns, nt, na = counts
        total = T1.shape[1] - 1
        astart = n + ns + nt

        # Уберём искусственные столбцы
        mask = np.ones(total, dtype=bool)
        mask[astart:astart + na] = False
        for r, bc in enumerate(basis1):
            if astart <= bc < astart + na:
                for j in range(total):
                    if mask[j] and abs(T1[r, j]) > 1e-10:
                        self._pivot(T1, r, j); basis1[r] = j; break
        non_rhs = T1[:, :total]
        T2 = np.hstack([non_rhs[:, mask], T1[:, -1:]])

        # Фаза II: восстановим целевую
        cvec = c.copy().astype(float)
        if sense == 'min': cvec = -cvec  # максимизируем -c^T x
        tot2 = T2.shape[1] - 1
        z = np.zeros(tot2 + 1); z[:n] = -cvec; T2[-1, :] = z
        for r, bc in enumerate(basis1):
            cj = cvec[bc] if bc < n else 0.0
            if abs(cj) > 1e-12: T2[-1, :] += cj * T2[r, :]

        s2, Top, bopt = self._simplex(T2, basis1)
        if s2 != 'optimal': return {'status': f'Phase II failed: {s2}'}

        # Извлечём решение
        xall = np.zeros(tot2)
        for r, bc in enumerate(bopt): xall[bc] = Top[r, -1]
        x_orig = xall[:n]
        obj = Top[-1, -1]
        if sense == 'min': obj = -obj

        # --- ENFORCE x >= 0 (simple check & cleanup) ---
        tol = 1e-9
        if np.any(x_orig < -tol):
            # если уходит ниже 0 значительно — сообщаем
            return {'status': 'violates_nonnegativity', 'x': x_orig}
        # отрезаем микронегативы из-за численной погрешности
        x_orig = np.where(x_orig < 0, 0.0, x_orig)

        return {'status': 'optimal', 'x': x_orig, 'objective': obj}

def main():
    if len(sys.argv) < 2:
        print("Usage: python two_phase_simplex.py <lp_file.txt>")
        return
    with open(sys.argv[1], 'r', encoding='utf-8') as f:
        txt = f.read()
    solver = LPSolverTwoPhase()
    sense, c, A, signs, b = solver.parse_lp_text(txt)
    res = solver.solve(sense, c, A, signs, b)
    if res.get('status') == 'optimal':
        x = res['x']; obj = res['objective']
        print("status: optimal")
        print("x*:", " ".join(f"{v:.6g}" for v in x))
        print("objective:", f"{obj:.6g}")
        print("nonnegativity check: passed (x >= 0)")
    else:
        print(res)

if __name__ == "__main__":
    main()
