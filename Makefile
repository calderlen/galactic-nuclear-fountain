CXX ?= c++
CPPFLAGS ?= -Imodel -Iobservations -Iio
CXXFLAGS ?= -std=c++17 -O3 -Wall -Wextra -Wpedantic
AR ?= ar
ARFLAGS := rcs

BUILD_DIR := build
MODEL_SOURCES := model/profiles.cpp model/physics.cpp model/models.cpp model/rk4.cpp
MODEL_OBJECTS := $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(MODEL_SOURCES))
HEADERS := model/profiles.h model/physics.h model/models.h model/rk4.h \
	observations/comparison.h io/csv.h
LIBRARY := $(BUILD_DIR)/libgalacticwind.a
RUNNER := $(BUILD_DIR)/galacticwind_model
COMPARISON := $(BUILD_DIR)/galacticwind_compare
RUNNER_OBJECTS := $(BUILD_DIR)/model/model_runner.o $(BUILD_DIR)/io/csv.o
COMPARISON_OBJECTS := $(BUILD_DIR)/observations/comparison_runner.o \
	$(BUILD_DIR)/observations/comparison.o $(BUILD_DIR)/io/csv.o

.PHONY: all

all: $(RUNNER) $(COMPARISON)

$(LIBRARY): $(MODEL_OBJECTS)
	$(AR) $(ARFLAGS) $@ $^

$(RUNNER): $(RUNNER_OBJECTS) $(LIBRARY)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(COMPARISON): $(COMPARISON_OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BUILD_DIR)/%.o: %.cpp $(HEADERS)
	mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@
