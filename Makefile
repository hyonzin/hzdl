CC := gcc
CFLAGS := -I$(shell pwd) -O3 -lm -fPIC -fopenmp
#CFLAGS := -I$(shell pwd) -O3 -lm -fPIC

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

default: tests

all: tests lib

tests: dir $(TEST_BINARIES)

lib: dir $(LIB)

dir: $(BUILD)

$(BUILD)/%.out: $(SRC)/test/%.c $(OBJECTS)
	$(CC) $^ -o $@ $(CFLAGS)

$(LIB): $(OBJECTS)
	$(CC) $^ --shared -o $@ $(CFLAGS)

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

