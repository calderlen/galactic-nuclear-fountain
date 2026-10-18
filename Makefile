CXX ?= c++
CPPFLAGS ?= -Imodel -Iobservations -Iio
CXXFLAGS ?= -std=c++23 -O3 -Wall -Wextra -Wpedantic
AR ?= ar
ARFLAGS := rcs

BUILD_DIR := build
MODEL_SOURCES := model/profiles.cpp model/physics.cpp model/models.cpp model/rk4.cpp model/prescribed_gas.cpp \
	model/potential.cpp model/orbits.cpp model/landing_kernel.cpp model/launch_distribution.cpp \
	model/cgm_amd.cpp model/cgm_io.cpp
MODEL_OBJECTS := $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(MODEL_SOURCES))
HEADERS := model/units.h model/profiles.h model/physics.h model/models.h model/rk4.h model/prescribed_gas.h model/prescribed_gas_io.h \
	model/potential.h model/orbits.h model/landing_kernel.h model/launch_distribution.h \
	model/cgm_amd.h model/cgm_io.h observations/comparison.h io/csv.h
LIBRARY := $(BUILD_DIR)/libgalactic-nuclear-fountain.a
RUNNER := $(BUILD_DIR)/galactic-nuclear-fountain-model
COMPARISON := $(BUILD_DIR)/galactic-nuclear-fountain-compare
KERNEL := $(BUILD_DIR)/galactic-nuclear-fountain-kernel
TRAJECTORIES := $(BUILD_DIR)/galactic-nuclear-fountain-trajectories
CALCULATIONS := $(BUILD_DIR)/galactic-nuclear-fountain-calculations
RUNNER_OBJECTS := $(BUILD_DIR)/model/model_runner.o $(BUILD_DIR)/model/prescribed_gas_io.o $(BUILD_DIR)/io/csv.o
COMPARISON_OBJECTS := $(BUILD_DIR)/observations/comparison_runner.o \
	$(BUILD_DIR)/observations/comparison.o $(BUILD_DIR)/io/csv.o

.PHONY: all kernel trajectories

all: $(RUNNER) $(COMPARISON) $(KERNEL) $(TRAJECTORIES) $(CALCULATIONS)

kernel: $(KERNEL)

trajectories: $(TRAJECTORIES)

$(LIBRARY): $(MODEL_OBJECTS)
	$(AR) $(ARFLAGS) $@ $^

$(RUNNER): $(RUNNER_OBJECTS) $(LIBRARY)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(COMPARISON): $(COMPARISON_OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(CALCULATIONS): $(BUILD_DIR)/model/calculation_runner.o $(BUILD_DIR)/observations/comparison.o $(BUILD_DIR)/io/csv.o $(LIBRARY)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(KERNEL): $(BUILD_DIR)/model/kernel_runner.o $(BUILD_DIR)/io/csv.o $(LIBRARY)
	$(CXX) $(CXXFLAGS) $^ -o $@ -lz

$(TRAJECTORIES): $(BUILD_DIR)/model/trajectory_runner.o $(BUILD_DIR)/io/csv.o $(LIBRARY)
	$(CXX) $(CXXFLAGS) $^ -o $@ -lz

$(BUILD_DIR)/%.o: %.cpp $(HEADERS)
	mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@
