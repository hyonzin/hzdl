
all: test.out

test.out: main.c read_mnist.c dnn.c activation.c util.c
	gcc -o $@ $+ -lm

clean:
	rm -f test.out

