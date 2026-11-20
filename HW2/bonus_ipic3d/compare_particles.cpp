// Compares the particle dumps of two runs (see dump_particles() in src/sputniPIC.cpp).
// Usage: ./compare_particles prefixA prefixB ns
// For every species and every array (x y z u v w) it prints the number of differing values,
// the maximum absolute difference and the maximum relative difference.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static bool readDump(const char* name, long& nop, std::vector<float> arr[6]) {
    FILE* f = fopen(name, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", name); return false; }
    if (fread(&nop, sizeof(long), 1, f) != 1) { fclose(f); return false; }
    for (int a = 0; a < 6; a++) {
        arr[a].resize(nop);
        if (fread(arr[a].data(), sizeof(float), nop, f) != (size_t) nop) { fclose(f); return false; }
    }
    fclose(f);
    return true;
}

int main(int argc, char** argv) {
    if (argc != 4) { fprintf(stderr, "Usage: %s prefixA prefixB ns\n", argv[0]); return 2; }
    const int ns = atoi(argv[3]);
    const char* labels[6] = {"x", "y", "z", "u", "v", "w"};
    long totalDiff = 0;
    for (int is = 0; is < ns; is++) {
        char na[1024], nb[1024];
        snprintf(na, sizeof(na), "%s_s%d.bin", argv[1], is);
        snprintf(nb, sizeof(nb), "%s_s%d.bin", argv[2], is);
        long nopA, nopB;
        std::vector<float> A[6], B[6];
        if (!readDump(na, nopA, A) || !readDump(nb, nopB, B)) return 2;
        if (nopA != nopB) { printf("species %d: particle counts differ (%ld vs %ld)\n", is, nopA, nopB); return 1; }
        for (int a = 0; a < 6; a++) {
            long diff = 0;
            double maxAbs = 0, maxRel = 0;
            for (long i = 0; i < nopA; i++) {
                const double d = std::fabs((double) A[a][i] - (double) B[a][i]);
                if (d > 0) {
                    diff++;
                    if (d > maxAbs) maxAbs = d;
                    const double rel = d / (std::fabs((double) A[a][i]) + 1e-30);
                    if (rel > maxRel) maxRel = rel;
                }
            }
            printf("species %d  %s: %ld of %ld values differ, max abs diff %.3e, max rel diff %.3e\n",
                   is, labels[a], diff, nopA, maxAbs, maxRel);
            totalDiff += diff;
        }
    }
    printf(totalDiff == 0 ? "RESULT: particle data is IDENTICAL\n"
                          : "RESULT: particle data differs in %ld values\n", totalDiff);
    return totalDiff == 0 ? 0 : 1;
}
