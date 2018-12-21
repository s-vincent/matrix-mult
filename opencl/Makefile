CFLAGS = -std=c11 -Wall -Wextra -Wstrict-prototypes -Wredundant-decls -Wshadow -pedantic -pedantic -fno-strict-aliasing -D_XOPEN_SOURCE=700 -O2 -I./
LDFLAGS = 
BIN = matmult-cl

all: $(BIN)
	
matmult-cl: matmult-cl.c util_opencl.c
	$(CC) $(CFLAGS) -o $@ $? $(LDFLAGS) -lOpenCL

clean:
	rm -f $(BIN) *.o

