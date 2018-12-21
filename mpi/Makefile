CC = mpicc
CFLAGS = -std=c11 -Wall -Wextra -Wstrict-prototypes -Wredundant-decls -Wshadow -pedantic -pedantic -fno-strict-aliasing -D_XOPEN_SOURCE=700 -O2 -I./
LDFLAGS = 
BIN = matmult-mpi matmult-mpi-omp

all: $(BIN)
	
matmult-mpi: matmult-mpi.c
	$(CC) $(CFLAGS) -D_REENTRANT -o $@ $? $(LDFLAGS) -lmpi -lpthread

matmult-mpi-omp: matmult-mpi.c
	$(CC) $(CFLAGS) -D_REENTRANT -fopenmp -o $@ $? $(LDFLAGS) -lmpi -lgomp

clean:
	rm -f $(BIN)
	rm -f *.o 

