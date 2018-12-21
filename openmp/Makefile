CFLAGS = -std=c11 -Wall -Wextra -Wstrict-prototypes -Wredundant-decls -Wshadow -pedantic -pedantic -fno-strict-aliasing -D_XOPEN_SOURCE=700 -O2 -I./
LDFLAGS = 
BIN = matmult-omp

all: $(BIN)
	
matmult-omp: matmult-omp.c
	$(CC) $(CFLAGS) -fopenmp -o $@ $? $(LDFLAGS) -lgomp

clean:
	rm -f $(BIN)
	rm -f *.o 

