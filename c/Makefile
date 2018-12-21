CFLAGS = -std=c11 -Wall -Wextra -Wstrict-prototypes -Wredundant-decls -Wshadow -pedantic -pedantic -fno-strict-aliasing -D_XOPEN_SOURCE=700 -O2 -I./
LDFLAGS = 
BIN = matmult

all: $(BIN)
	
matmult: matmult.c
	$(CC) $(CFLAGS) -o $@ $? $(LDFLAGS)

clean:
	rm -f $(BIN)
	rm -f *.o 

