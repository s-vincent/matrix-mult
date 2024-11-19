CFLAGS = -std=c11 -Wall -Wextra -Wstrict-prototypes -Wredundant-decls -Wshadow -pedantic -pedantic \
				 -fno-strict-aliasing -D_XOPEN_SOURCE=700 -O2 -I./ \
				 -DCL_TARGET_OPENCL_VERSION=200
LDFLAGS =
BIN = matmult-cl

all: $(BIN)

matmult-cl: matmult-cl.o util_opencl.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) -lOpenCL

clean:
	rm -f $(BIN) *.o

