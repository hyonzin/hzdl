
SOURCES=hzdl/read_mnist.c hzdl/dnn.c hzdl/activation.c hzdl/util.c

all: test.out

test.out: main.c $(SOURCES)
	gcc -o $@ $+ -lm

clean:
	rm -f test.out

