import sqlite3
from pathlib import Path
from statistics import median
from collections import Counter

root = Path(__file__).parent
for qubits in (16, 24, 28):
    path = (root / f"qft_{qubits}q.sqlite").as_uri() + "?mode=ro"
    with sqlite3.connect(path, uri=True) as conn:
        rows = conn.execute("""
            SELECT n.value, k.start, k.end, k.correlationId,
                   a.globalTid, api.value, a.start
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds n ON k.shortName = n.id
            LEFT JOIN CUPTI_ACTIVITY_KIND_RUNTIME a ON a.correlationId = k.correlationId
            LEFT JOIN StringIds api ON a.nameId = api.id
            WHERE n.value LIKE 'kernel_statevec_anyCtrl%'
            ORDER BY k.start
        """).fetchall()
        count = qubits * (qubits + 1) // 2 + qubits // 2
        assert len(rows) == 6 * count
        rep = rows[count:2 * count]
        print(qubits, "timed rep 0 GPU interval (ms):", rep[0][1] / 1e6, rep[-1][2] / 1e6)
        print("Gate median (us):", median((r[2] - r[1]) / 1000 for r in rep))
        print("CPU thread and launch API:", Counter((r[4] % (1 << 24), r[5]) for r in rep))
        output = conn.execute("""
            SELECT s.value FROM ANALYSIS_FILE a JOIN StringIds s ON s.id = a.contentId
            WHERE a.filename LIKE '%/stdout_%'
        """).fetchone()
        if output:
            print(output[0])
        for keyword in ("Dense", "Diag", "Swap"):
            first = next(r for r in rep if keyword in r[0])
            print("First", keyword, "event (kernel, correlation, API ms, GPU ms):",
                  first[0], first[3], first[6] / 1e6, first[1] / 1e6)
        if qubits == 28:
            print("Memory types:", conn.execute("SELECT * FROM ENUM_CUDA_MEM_KIND").fetchall())
            print("Operations:", conn.execute("SELECT * FROM ENUM_CUDA_DEV_MEM_EVENT_OPER").fetchall())
            rows = conn.execute("""
                SELECT bytes, memoryOperationType FROM CUDA_GPU_MEMORY_USAGE_EVENTS
                WHERE memKind = 2 ORDER BY start
            """)
            current = peak = 0
            for size, op in rows:
                current += size if op == 0 else -size
                peak = max(current, peak)
            print("Peak CUDA device allocations:", peak, "bytes")
