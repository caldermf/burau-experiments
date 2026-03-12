/* Search with variable level_delta (level change at hump 0).
 * Original code uses HUMPS+1 = 4 for n=4.
 * We try different values to find mod-m solutions.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HUMPS 3
#define LEVELS 30000
#define STARTLEVEL 15000

int gcd(int i, int j) {
    if (i < 0) i = -i;
    if (j < 0) j = -j;
    while (j) { int t = j; j = i % j; i = t; }
    return i;
}

char example_ld(int *nextcut, int *flcut, int leftend, int mod,
                int level_delta,
                int *out_poly, int *out_lo, int *out_hi) {
    int cut, newcut;
    int level = STARTLEVEL;
    int i;
    int lo = STARTLEVEL;
    int hi = STARTLEVEL - 1;
    int parity = 0;
    int norm_z = 0, norm_m = 0;
    int togo, rightend;
    int total = nextcut[HUMPS - 1];
    int poly[LEVELS];

    togo = total / 2;
    rightend = leftend + total / 2;
    cut = nextcut[0] / 2;

    while (togo >= norm_m) {
        if (level < 0 || level >= LEVELS) return 'o';
        while (lo > level) poly[--lo] = 0;
        while (hi < level) poly[++hi] = 0;

        int old_val = poly[level];
        int old_mod = ((old_val % mod) + mod) % mod;

        if (parity) {
            poly[level]++;
            if (old_val >= 0) norm_z++; else norm_z--;
        } else {
            poly[level]--;
            if (old_val <= 0) norm_z++; else norm_z--;
        }

        int new_mod = ((poly[level] % mod) + mod) % mod;
        if (old_mod != 0 && new_mod == 0) norm_m--;
        else if (old_mod == 0 && new_mod != 0) norm_m++;

        togo--;
        cut = 2 * leftend - 1 - cut;
        if (cut < 0) cut += total;
        else if (cut >= total) cut -= total;

        for (i = 0; nextcut[i] <= cut; i++);

        newcut = flcut[i] - cut;
        if (newcut < cut) {
            if (i) level--;
            else { parity = !parity; level += level_delta; }
            if (newcut < leftend && cut >= leftend) level--;
            if (newcut < rightend && cut >= rightend) level--;
        } else if (newcut > cut) {
            if (i) level++;
            else { parity = !parity; level -= level_delta; }
            if (newcut >= leftend && cut < leftend) level++;
            if (newcut >= rightend && cut < rightend) level++;
        } else {
            if (norm_m) return 0;
            if (norm_z == 0) return 0;
            if (out_poly) {
                *out_lo = lo; *out_hi = hi;
                for (i = lo; i <= hi; i++) out_poly[i] = poly[i];
            }
            if (togo) return 'v';
            return 'V';
        }
        cut = newcut;
    }
    return 0;
}

void print_poly(int *poly, int lo, int hi) {
    int first = 1;
    for (int i = lo; i <= hi; i++) {
        if (poly[i] != 0) {
            int exp = i - STARTLEVEL;
            int c = poly[i];
            if (!first && c > 0) printf(" + ");
            else if (!first && c < 0) printf(" - ");
            else if (c < 0) printf("-");
            if (c < 0) c = -c;
            if (exp == 0) printf("%d", c);
            else if (exp == 1) { if (c == 1) printf("t"); else printf("%dt", c); }
            else if (exp == -1) { if (c == 1) printf("t^-1"); else printf("%dt^-1", c); }
            else { if (c == 1) printf("t^%d", exp); else printf("%dt^%d", c, exp); }
            first = 0;
        }
    }
    if (first) printf("0");
}

int main(int argc, char **argv) {
    int mod = 3;
    int start_total = 4;
    int end_total = 200;
    int target_ld = 0;  /* 0 = try all from 1 to 6 */

    for (int a = 1; a < argc; a++) {
        if (!strcmp(argv[a], "--mod")) mod = atoi(argv[++a]);
        else if (!strcmp(argv[a], "--start")) start_total = atoi(argv[++a]);
        else if (!strcmp(argv[a], "--end")) end_total = atoi(argv[++a]);
        else if (!strcmp(argv[a], "--ld")) target_ld = atoi(argv[++a]);
    }

    printf("Level delta search: mod %d, total %d-%d, ld=%s\n",
           mod, start_total, end_total,
           target_ld ? "fixed" : "all(1-6)");
    fflush(stdout);

    int found_v_by_ld[7] = {0};
    int found_p_by_ld[7] = {0};

    for (int total = start_total; total <= end_total; total++) {
        for (int w0 = 1; w0 <= total; w0++) {
            for (int w1 = 0; w0 + w1 <= total; w1++) {
                int w2 = total - w0 - w1;
                int width[3] = {w0, w1, w2};
                int nextcut[HUMPS], flcut[HUMPS];
                int gcdwidths = 0;

                for (int i = 0; i < HUMPS; i++) {
                    gcdwidths = gcd(width[i], gcdwidths);
                    nextcut[i] = (i ? nextcut[i-1] : 0) + width[i];
                    flcut[i] = (i ? nextcut[i-1] : 0) + nextcut[i] - 1;
                }

                int poly_buf[LEVELS];
                int plo, phi;

                for (int leftend = total / 2; leftend >= 0; leftend--) {
                    if (gcd(gcdwidths, leftend) != 1) continue;

                    int ld_lo = target_ld ? target_ld : 1;
                    int ld_hi = target_ld ? target_ld : 6;

                    for (int ld = ld_lo; ld <= ld_hi; ld++) {
                        if (ld == HUMPS + 1) continue;  /* skip default, already searched */

                        char code = example_ld(nextcut, flcut, leftend, mod,
                                              ld, poly_buf, &plo, &phi);
                        if (code == 'V') {
                            found_v_by_ld[ld]++;
                            int total_v = 0;
                            for (int k = 1; k <= 6; k++) total_v += found_v_by_ld[k];
                            if (total_v <= 50) {
                                printf("VICTORY [t=%d ld=%d]: w[%d,%d,%d] le=%d poly=",
                                       total, ld, w0, w1, w2, leftend);
                                print_poly(poly_buf, plo, phi);
                                printf("\n");
                                fflush(stdout);
                            }
                        } else if (code == 'v') {
                            found_p_by_ld[ld]++;
                        }
                    }
                }
            }
        }

        if (total % 20 == 0) {
            printf("  [total=%d]", total);
            for (int ld = 1; ld <= 6; ld++) {
                if (ld == HUMPS + 1) continue;
                if (found_v_by_ld[ld] || found_p_by_ld[ld])
                    printf(" ld%d:V=%d,v=%d", ld, found_v_by_ld[ld], found_p_by_ld[ld]);
            }
            printf("\n");
            fflush(stdout);
        }
    }

    printf("\nSummary by level_delta:\n");
    for (int ld = 1; ld <= 6; ld++) {
        if (ld == HUMPS + 1) { printf("  ld=%d: (default, skipped)\n", ld); continue; }
        printf("  ld=%d: %d victories, %d partials\n", ld, found_v_by_ld[ld], found_p_by_ld[ld]);
    }
    return 0;
}
