CC := gcc
CFLAGS := -I. -O3 -lm -fPIC -fopenmp

SRC := hzdl
BUILD := build
LIB := $(BUILD)/libhzdl.so

.PRECIOUS: $(BUILD)/%.o
SOURCES := $(wildcard $(SRC)/*.c) \
		   $(wildcard $(SRC)/layer/*.c) \
		   $(wildcard $(SRC)/dataset/*.c)
OBJECTS := $(addprefix $(BUILD)/, $(patsubst %.c, %.o, $(notdir $(SOURCES))))
TEST_SOURCES := $(wildcard $(SRC)/test/*.c)
TEST_BINARIES := $(addprefix $(BUILD)/, $(patsubst %.c, %.out, $(notdir $(TEST_SOURCES))))

all: dir tests

dir: $(BUILD)

tests: $(TEST_BINARIES)

$(BUILD)/%.out: $(SRC)/test/%.c $(OBJECTS)
	$(CC) $^ -o $@ $(CFLAGS)

$(BUILD)/%.o: $(SRC)/%.c
	$(CC) -c $< -o $@ $(CFLAGS)

$(BUILD)/%.o: $(SRC)/layer/%.c
	$(CC) -c $< -o $@ $(CFLAGS)

$(BUILD)/%.o: $(SRC)/dataset/%.c
	$(CC) -c $< -o $@ $(CFLAGS)

$(BUILD):
	mkdir -p $@

clean:
	rm -fr $(BUILD)

