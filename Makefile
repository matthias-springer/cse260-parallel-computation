# This makefile is intended for the GNU C compiler.
# Your code must compile (with GCC) with the given CFLAGS.
# You may experiment with the OPT variable to invoke additional compiler options.
atlas := 1
include $(PUB)/Arch/arch.gnu-4.7_c99.generic
WARNINGS += -Wall -pedantic


#
# If you want to add symbol table information for gdb/cachegrind
# specify dbg=1 on the "make" command line
ifeq ($(dbg), 1)
        CFLAGS += -g
        LDFLAGS += -g
        C++FLAGS += -g
endif



# Compiler for gprof
ifeq ($(gprof), 1)
        CFLAGS += -g -pg
        C++FLAGS += -g -pg
        LDFLAGS += -g -pg
endif

# If you want to copy data blocks to contiguous storage
# This applies to the hand code version
ifeq ($(copy), 1)
    C++FLAGS += -DCOPY
    CFLAGS += -DCOPY
endif


# If you want to use restrict pointers, make restrict=1
# This applies to the hand code version
ifeq ($(restrict), 1)
    C++FLAGS += -D__RESTRICT
    CFLAGS += -D__RESTRICT
ifneq ($(CARVER), 0)
    C++FLAGS += -restrict
    CFLAGS += -restrict
endif
endif

ifeq ($(NO_BLAS), 1)
    C++FLAGS += -DNO_BLAS
    CFLAGS += -DNO_BLAS
endif

CFLAGS += -DVALIDATION=$(VALIDATION) -O4 -msse -msse2 -msse3 -m3dnow -mfpmath=sse
#CFLAGS += -DVALIDATION=$(VALIDATION)

#DEBUG += -DDEBUG



targets = benchmark-naive benchmark-blocked benchmark-blas
objects = benchmark.o dgemm-naive.o dgemm-blocked.o dgemm-blas.o
UTIL   = wall_time.o cmdLine.o

.PHONY : default
default : all

.PHONY : all
all : clean $(targets)

benchmark-naive : benchmark.o dgemm-naive.o  $(UTIL)
	$(CC) -o $@ $^ $(LDLIBS)
benchmark-blocked : benchmark.o dgemm-blocked.o $(UTIL)
	$(CC) -o $@ $^ $(LDLIBS)
benchmark-blas : benchmark.o dgemm-blas.o $(UTIL)
	$(CC) -o $@ $^ $(LDLIBS)

%.o : %.c
	$(CC) -c $(CFLAGS) $<


.PHONY : clean
clean:
	rm -f $(targets) $(objects) $(UTIL) core
