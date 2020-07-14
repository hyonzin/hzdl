CC := gcc
CFLAGS := -O3 -lm -fPIC

SRC := hzdl/
BUILD := build/
LIB := $(BUILD)/libhzdl.so
TEST := test.out

SOURCES := $(wildcard $(SRC)/*.c) \
		   $(wildcard $(SRC)/layer/*.c) \
		   $(wildcard $(SRC)/example/*.c)
OBJECTS := $(addprefix $(BUILD)/, $(patsubst %.c, %.o, $(notdir $(SOURCES))))


test: dir $(TEST)

lib: dir $(LIB)

all: dir $(TEST) $(LIB)
	
$(LIB): $(BUILDECTS)
	$(CC) $^ -o $@ $(CFLAGS) --shared

$(BUILD)/%.o: $(SRC)/%.c
	$(CC) -c $< -o $@ $(CFLAGS)

$(BUILD)/%.o: $(SRC)/layer/%.c
	$(CC) -c $< -o $@ $(CFLAGS)

$(BUILD)/%.o: $(SRC)/example/%.c
	$(CC) -c $< -o $@ $(CFLAGS)

$(BUILD)/main.o: ./main.c
	$(CC) -c $< -o $@ $(CFLAGS)

$(TEST): $(BUILD)/main.o $(OBJECTS)
	$(CC) $^ -o $@ $(CFLAGS)


dir:
	mkdir -p $(BUILD)/

clean:
	rm -fr $(TEST) $(BUILD)

