
all: test.out

test.out: main.c read_mnist.c
	gcc -o $@ $+

