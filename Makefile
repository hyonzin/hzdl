CC := gcc
CFLAGS=-O3 -lm

SRC := hzdl/
OBJ := build/
BIN := bin/

SOURCES := $(wildcard $(SRC)/*.c) $(wildcard $(SRC)/layer/*.c) $(wildcard $(SRC)/example/*.c) $(wildcard ./*.c)  
OBJECTS := $(patsubst %.c, $(OBJ)/%.o, $(SOURCES))


all: dir $(BIN)/test.out
	
$(BIN)/test.out: $(OBJECTS)
	$(CC) $^ -o $@ $(CFLAGS)

$(OBJ)/%.o: ./%.c
	$(CC) -c $< -o $@ $(CFLAGS)

$(OBJ)/%.o: $(SRC)/%.c
	$(CC) -c $< -o $@ $(CFLAGS)

$(OBJ)/layer/%.o: $(SRC)/layer/%.c
	$(CC) -c $< -o $@ $(CFLAGS)

$(OBJ)/example/%.o: $(SRC)/example/%.c
	$(CC) -c $< -o $@ $(CFLAGS)

dir:
	mkdir -p $(BIN) $(OBJ)/hzdl/layer $(OBJ)/hzdl/example

clean:
	rm -fr $(BIN) $(OBJ)
