CC=gcc
CFLAGS=-lm -lpthread

all: single multi

single:
	$(CC) -o bin/psrs-single src/psrs.c -lm -lpthread

multi:
	$(CC) -o bin/psrs-multi src/psrs.c -lm -lpthread -DPTHREADS
