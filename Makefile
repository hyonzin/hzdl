CC := gcc
CFLAGS := -I. -O3 -lm -fPIC -fopenmp

SRC := hzdl/
BUILD := build/
LIB := $(BUILD)/libhzdl.so

SOURCES := $(wildcard $(SRC)/*.c) \
		   $(wildcard $(SRC)/layer/*.c) \
		   $(wildcard $(SRC)/dataset/*.c)
OBJECTS := $(addprefix $(BUILD)/, $(patsubst %.c, %.o, $(notdir $(SOURCES))))


all: dir $(BUILD)/test_mnist.out

$(BUILD)/test_mnist.out: $(SRC)/test/test_mnist.c $(OBJECTS)
	$(CC) $^ -o $@ $(CFLAGS)


$(BUILD)/%.o: $(SRC)/%.c
	$(CC) -c $< -o $@ $(CFLAGS)

$(BUILD)/%.o: $(SRC)/layer/%.c
	$(CC) -c $< -o $@ $(CFLAGS)

$(BUILD)/%.o: $(SRC)/dataset/%.c
	$(CC) -c $< -o $@ $(CFLAGS)

dir:
	mkdir -p $(BUILD)/

clean:
	rm -fr $(BUILD)

