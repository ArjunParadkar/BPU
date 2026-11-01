# PS firmware: host build (C model of the PL) and board build (mmap HAL).
#   make host            -> build/host/{runner,write_verify,selftest}
#   make board           -> build/board/{runner,write_verify,measure}   (CC=aarch64-linux-gnu-gcc or native on the Kria)
CC      ?= gcc
CFLAGS  ?= -O2 -std=c11 -Wall -Wextra -Iinclude
SRC_COMMON = src/bpu_tile.c src/bpu_tables.c src/hal.c src/hal_model.c
src/bpu_tables.c: ../scripts/gen_ps_tables.py
	python3 ../scripts/gen_ps_tables.py $@
host: src/bpu_tables.c
	mkdir -p build/host
	$(CC) $(CFLAGS) -o build/host/runner       src/runner.c $(SRC_COMMON)
	$(CC) $(CFLAGS) -o build/host/write_verify src/write_verify_main.c $(SRC_COMMON)
	$(CC) $(CFLAGS) -o build/host/selftest     src/selftest.c src/bpu_tables.c
board: src/bpu_tables.c
	mkdir -p build/board
	$(CC) $(CFLAGS) -DBPU_HAVE_MMAP -o build/board/runner       src/runner.c $(SRC_COMMON) src/hal_mmap.c
	$(CC) $(CFLAGS) -DBPU_HAVE_MMAP -o build/board/write_verify src/write_verify_main.c $(SRC_COMMON) src/hal_mmap.c
	$(CC) $(CFLAGS) -DBPU_HAVE_MMAP -o build/board/measure      src/measure.c $(SRC_COMMON) src/hal_mmap.c
clean:
	rm -rf build
.PHONY: host board clean
