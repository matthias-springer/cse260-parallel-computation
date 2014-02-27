include $(PUB)/Arch/arch.gnu-4.7_c++11.generic

# If you want to add symbol table information for gdb/cachegrind
# specify gdb=1 on the "make" command line
ifeq ($(gdb), 1)
        CFLAGS += -g
        LDFLAGS += -g
        C++FLAGS += -g
endif   


# If you want to add symbol table information for gprof
# specify gprof=1 on the "make" command line
ifeq ($(gprof), 1)
        CFLAGS += -g -pg
        C++FLAGS += -g -pg
        LDFLAGS += -g -pg
endif

WARNINGS += -Wall -pedantic

C++FLAGS 	+= $(WARNINGS)

C++LINK		= $(C++)

C++FLAGS += $(OPTIMIZATION)

# Enables MPI compilation
mpi = 1

# Use the parallel print facility
# ppf = 1

# Enable MPI compilation
# omp = 1

TARGETS = nbody

all:	$(TARGETS)

SOBJ = nbody.o common.o cmdLine.o particlesHelper.o Plotting.o Bin.o World.o

nbody: $(SOBJ)
	$(C++LINK) $(C++FLAGS) -o $@ $(LDLIBS) $(SOBJ)

# ===========================
#
.PHONY: clean
clean:
	rm -f *.o $(TARGETS)

