# Makefile for Cross Interleaved Reed Solomon encoder/decoder
#
# (c) Henry Minsky, Universal Access 1991-1996
#

CC = gcc

DEBUG_FLAGS = -g
CFLAGS = -shared -fPIC -Wall -Wstrict-prototypes  $(OPTIMIZE_FLAGS) $(DEBUG_FLAGS) -I..

CFILES = rs.c galois.c berlekamp.c crcgen.c

TARGET_LIB = libecc.so

all: $(CFILES)
	$(CC) $(CFLAGS) -o $(TARGET_LIB) $(CFILES)

clean:
	rm -f *.o libecc.so

# DO NOT DELETE THIS LINE -- make depend depends on it.
