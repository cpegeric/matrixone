# C++ compiler (use NVCC for CUDA-related compilation and linking)
NVCC := $(CUDA_HOME)/bin/nvcc
CXX := g++ # Still keep for reference, but won't be used for brute_force_c.cpp

# Paths from parent Makefile
CUDA_HOME ?= /usr/local/cuda
GOCUVS ?= /home/eric/miniconda3/envs/go # Assuming GOCUVS base path if not specified
CONDA_PREFIX ?= /home/eric/miniconda3/envs/go # Assuming CONDA_PREFIX for other headers

# Common include flags for C++ compilation
CLFLAGS := -I. -I$(CUDA_HOME)/include -I$(CONDA_PREFIX)/include -I$(GOCUVS)/include/rapids -I$(GOCUVS)/include/raft -I$(GOCUVS)/include/cuvs -I../cpp

# Compiler flags for C++ source files (brute_force_c.cpp)
# Use -x cu to force nvcc to treat it as a CUDA C++ file
# -Xcompiler passes flags to the host C++ compiler (g++)
NVCC_COMPILER_FLAGS := -std=c++17 -x cu -Xcompiler "-Wall -Wextra -fPIC -O2" $(CLFLAGS) -DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE

# Linker flags for creating a shared library (still use NVCC as linker driver)
NVCC_LDFLAGS := -L$(CUDA_HOME)/lib64/stubs -lcuda -L$(CUDA_HOME)/lib64 -lcudart -L$(GOCUVS)/lib -lcuvs -lcuvs_c -ldl -lrmm
HOST_LDFLAGS := -lpthread -lm

LDFLAGS := $(NVCC_LDFLAGS) $(addprefix -Xlinker ,$(HOST_LDFLAGS))

# Output library names
TARGET_SHARED_LIB := libbrute_force_c.so
TARGET_STATIC_LIB := libbrute_force_c.a
SOURCE_FILE := brute_force_c.cpp
OBJECT_FILE := $(SOURCE_FILE:.cpp=.o)

.PHONY: all clean

all: $(TARGET_SHARED_LIB) $(TARGET_STATIC_LIB)

$(TARGET_SHARED_LIB): $(OBJECT_FILE)
	@echo "Linking shared library $@"
	$(NVCC) -shared $(OBJECT_FILE) $(LDFLAGS) -o $@

$(TARGET_STATIC_LIB): $(OBJECT_FILE)
	@echo "Creating static library $@"
	ar rcs $@ $<

$(OBJECT_FILE): $(SOURCE_FILE)
	@echo "Compiling $< with NVCC (as CUDA C++)"
	$(NVCC) $(NVCC_COMPILER_FLAGS) -c $< -o $@

clean:
	@echo "Cleaning up..."
	rm -f $(TARGET_SHARED_LIB) $(TARGET_STATIC_LIB) $(OBJECT_FILE)
