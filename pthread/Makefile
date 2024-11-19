CFLAGS = -std=c11 -Wall -Wextra -Wstrict-prototypes -Wredundant-decls -Wshadow -pedantic -pedantic -fno-strict-aliasing -D_XOPEN_SOURCE=700 -O2 -I./
LDFLAGS =
BIN = matmult-pthread

all: $(BIN)

matmult-pthread: matmult-pthread.c
	$(CC) $(CFLAGS) -D_REENTRANT -o $@ $? $(LDFLAGS) -lpthread

clean:
	rm -f $(BIN)
	rm -f *.o

