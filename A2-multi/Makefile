HOST = $(shell hostname)
LILLIPUT   =  $(shell expr match `hostname` lilliput)
CSECLASS   =  $(shell expr match `hostname` cseclass)
BANG   =  $(shell expr match `hostname` ccom-bang)
CARVER =  $(shell expr match `hostname` cvrsvc)
multi = 1


NO_BLAS = 0
ifneq ($(LILLIPUT), 0)
PUB	= /class/public/cse260-wi14
include $(PUB)/Arch/arch.intel-mkl
else
ifneq ($(CSECLASS), 0)
PUB	= /class/public/cse260-wi14
include $(PUB)/Arch/arch.intel
NO_BLAS = 1
else
# No parallel target using ATLAS
ifneq ($(BANG), 0)
atlas = 1
multi = 0
include $(PUB)/Arch/arch.gnu-4.7_c++11.generic
# PUB = /share/class/public/cse260-wi14
# include $(PUB)/Arch/arch.gnu.atlas
else
ifneq ($(CARVER), 0)
PUB = /global/u2/b/baden/cse260-wi14
C++FLAGS += -DUSE_MKL
CFLAGS += -DUSE_MKL
include $(PUB)/Arch/arch.intel-mkl
else
PUB = /Path/To/Your/Own/machine/here
include $(PUB)/Arch/arch.gnu
endif
endif
endif
endif

# XTRAFLAGS += -fsource-asm -S

# Add symbol table information for gdb/cachegrind
ifeq ($(debug), 1)
        CFLAGS += -g
        LDFLAGS += -g
        C++FLAGS += -g
endif   


# Add symbol table information for gprof
ifeq ($(gprof), 1)
        CFLAGS += -g -pg
        C++FLAGS += -g -pg
        LDFLAGS += -g -pg
endif

# If you want to compile for single precision,
# specify single=1 on the "make" command line
ifeq ($(single), 1)
else
    C++FLAGS += -D_DOUBLE
    CFLAGS += -D_DOUBLE
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


#DEBUG += -DDEBUG


UTIL	= Timer.o
ifeq ($(NO_BLAS), 0)
	MM_BLAS	= mm_blas.o
endif
OBJ	= mmpy.o $(MM_BLAS) cmdLine.o verify.o $(UTIL)

mmpy:	        $(OBJ)
		$(C++LINK) $(LDFLAGS) -o $@ $(OBJ)  $(LDLIBS)



clean:	
	$(RM) *.o mmpy;
	$(RM) core;

