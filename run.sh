#
gcc -o psim psim.c dynSpecSim.c T2toolkit.c tempo2pred.c cheby2d.c t1polyco.c -L/usr/local/lib/cfitsio -I/usr/include/cfitsio/ -lcfitsio -lfftw3 -lm -fopenmp


